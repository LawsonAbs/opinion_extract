import collections
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.model_selection import train_test_split
import sys
from sklearn.metrics import f1_score
from visdom import Visdom
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import json
import time
import pickle
import random
import warnings
import numpy as np
from torch.optim import Optimizer
from collections import defaultdict
from lookahead import Lookahead
from torch.nn import BCELoss
import torch
torch.set_printoptions(profile="full")
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.backends import cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BertTokenizerFast
)
from modeling.modeling_nezha.modeling import NeZhaPreTrainedModel, NeZhaModel
from modeling.bert.modeling_bert import BertModel, BertPreTrainedModel
from sparse_max.sparsemax import SparsemaxLoss
from sklearn.model_selection import StratifiedKFold,KFold

label_dict = {'5-24': 0, '6-34': 1, '1-1': 2, '6-8': 3, '10-26': 4, '2-3': 5, '5-22': 6, '6-28': 7, '8-18': 8, '1-4': 9, '2-6': 10, '6-21': 11, 
              '7-16': 12, '6-29': 13, '6-20': 14, '6-15': 15, '6-13': 16, '9-23': 17, '5-35': 18, '2-33': 19, '5-30': 20, '1-9': 21, '8-27': 22, 
              '1-10': 23, '6-19': 24, '3-5': 25, '2-2': 26, '4-7': 27, '2-17': 28, '5-12': 29, '6-32': 30, '6-31': 31, '2-25': 32, '2-11': 33, '2-14': 34}

id2label = {0: '5-24', 1: '6-34', 2: '1-1', 3: '6-8', 4: '10-26', 5: '2-3', 6: '5-22', 7: '6-28', 8: '8-18', 9: '1-4', 10: '2-6', 11: '6-21', 12: '7-16', 13: '6-29', 14: '6-20', 15: '6-15', 16: '6-13', 17: '9-23', 18: '5-35', 19: '2-33', 20: '5-30', 21: '1-9', 22: '8-27', 23: '1-10', 24: '6-19', 25: '3-5', 26: '2-2', 27: '4-7', 28: '2-17', 29: '5-12', 30: '6-32', 31: '6-31', 32: '2-25', 33: '2-11', 34: '2-14'}

            
label_list = ['5-24', '6-34', '1-1', '6-8', '10-26', '2-3', '5-22', '6-28', '8-18', '1-4', '2-6', '6-21', '7-16', '6-29', '6-20', 
              '6-15', '6-13', '9-23', '5-35', '2-33', '5-30', '1-9', '8-27', '1-10', '6-19', '3-5', '2-2', '4-7', '2-17', '5-12', 
              '6-32', '6-31', '2-25', '2-11', '2-14']
bad_clz = ['6-20','5-24','1-1','6-8','5-22','6-13',
            '1-9','6-31','2-11','2-14','6-19','6-28','4-7',  # 13

        # less sample                 
            '3-5','8-27','7-16','8-18','9-23','10-26','2-17' # 7
            ]


viz = Visdom()
win_1 = "0.36_train_loss"
win_2= "0.36_macro_f1_eval"
win_3 = "3-5 loss_2"
opts = {
    "xlabel":'step',
    "ylabel":"value",
    "title":"0.36 loss"
}
step_3_5 = 0

# 分割成train - dev 
# 其实完全没有必要手写这个方法，可以使用 sklearn.model_selection 中的 StratifiedKFold 来实现 => 
# 后来发现还是得手写，函数分割得到的结果还是不行
def split_data_by_class(x,y,rate,seed=22):
    '''
    x,y 表示训练数据的输入和标签
    如果random = True, 则随机分割~
    rate表示dev所占数据的比例，如果不足1条，则按1条处理

    returns:
        train_idx,dev_idx
    '''    
    # 先shuffle一下再说，保持相同的shuffle  
    random.seed(seed)
    random.shuffle(y)
    random.seed(seed)
    random.shuffle(x)

    cont_id = {} # 每个类别都放到一个list中    
    # 保持每个类别划分相同
    for i in range(len(y)):
        y_idx = y[i] # y_idx是个分类值
        if y_idx not in cont_id.keys():
            cont_id[y_idx] = []
        cont_id[y_idx].append(i) # 将该类别的下标放到其中
    
    train_idx,dev_idx = [],[] # 最后返回的值，是一个下标
    # 保持按照类别均匀抽取
    for item in cont_id.items():
        key,value = item # key 是类别信息，value是该类别的所有（在y中）下标
        if len(value) >= 2: # 如果当前类别的样本数大于2，才分成dev
            mid = int(len(value)*rate)
            if mid == 0: # 如果只有0，那么就得将其改成1
                mid = 1
            for i in range(mid): # 按照个数取前 rate * len(value) 个
                dev_idx.append(value[i])         
            for i in range(mid,len(value)): # 从上次的开始，作为train 集合
                train_idx.append(value[i])
        else:            
             train_idx.extend(value) # 将整个值都作为train
    
    # 得到值后再次shuffle，避免输出连续相同的值
    random.seed(seed)
    random.shuffle(train_idx)
    random.seed(seed)
    random.shuffle(dev_idx)
    
    # 需要判断一下最后返回的dev_idx 的长度是否符合要求，如果不合要求，则要随机再抽取一点儿到其中
    while len(dev_idx) < len(train_idx) * rate:
        idx = random.choice(train_idx)        
        dev_idx.append(idx) # 放到dev中
        train_idx.remove(idx) # 删除这个idx

    return train_idx,dev_idx # 返回train/dev 数据所在的下标


class NeZhaSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 35
        # self.num_labels = 20
        self.bert = BertModel(config) 
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.multi_drop = 5
        self.multi_dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(self.multi_drop)])
        # self.loss_fct = FocalLoss(35,
        # # alpha=torch.tensor([6.4558,2.5173,0.3665,1.3661,22.2365,
        # #                     0.4059,1.6205,1.7949,20.0129,0.5009,
        # #                     0.4911,2.9870,12.5080,1.2392,12.1290,
        # #                     0.6274,2.1636,21.0662,1.0617,0.6773,
        # #                     0.3105,0.2073,8.1685,0.6435,2.2486,
        # #                     25.0161,0.5438,0.6414,2.2742,1.9525,
        # #                     3.8860,2.5658,1.6747,1.1180,1.4244]))


        # alpha = torch.tensor([2.8650,1.9232,0.3665,1.3119,4.1017,
        #                     0.4059,1.4827,1.5849,3.9964,0.5009,
        #                     0.4911,2.0943,3.5264,1.2145,3.4956,
        #                     0.6274,1.7718,4.0477,1.0599,0.6773,
        #                     0.3105,0.2073,3.1003,0.6435,1.8103,
        #                     4.2195,0.5438,0.6414,1.8216,1.6691,
        #                     2.3574,1.9423,1.5156,1.1116,1.3538])
        #                     )
        self.loss_fct = LabelSmoothingLoss(smoothing=0.01)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)
        for j, dropout in enumerate(self.multi_dropouts):
            if j == 0:
                logits = self.classifier(dropout(pooled_output)) / self.multi_drop
            else:
                logits += self.classifier(dropout(pooled_output)) / self.multi_drop
        outputs = (logits,) + outputs[2:]

        
        if labels is not None:            
            # loss_fct = nn.CrossEntropyLoss()
            # loss_fct = SparsemaxLoss()
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
            
            # print("当前正在处理的数据是：",input_ids)
            # print("对应的标签是",labels)
            # print("损失是：",loss)
        return outputs


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

# 这里for得到的是一个yield函数，但是感觉这么做是不是有点儿低效？
# 为什么想着用这种方式来获取一个batch，不都是直接遍历获取的吗？
def batch_loader(config, src, tgt, seg, mask):
    ins_num = src.size()[0]
    batch_size = config['batch_size']
    for i in range(ins_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        mask_batch = mask[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch, mask_batch
    if ins_num > ins_num // batch_size * batch_size:
        src_batch = src[ins_num // batch_size * batch_size:, :]
        tgt_batch = tgt[ins_num // batch_size * batch_size:]
        seg_batch = seg[ins_num // batch_size * batch_size:, :]
        mask_batch = mask[ins_num // batch_size * batch_size:, :]
        yield src_batch, tgt_batch, seg_batch, mask_batch

def read_single_dataset(config, tokenizer, path):
    start = time.time()
    dataset = []
    seq_length = config['max_seq_len']
    label_list = ['不匹配', '部分匹配', '完全匹配']
    print('>> load data:', config[path])
    with open(config[path], 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            sent_a, sent_b, tgt = line.strip().split('\t')
            src_a = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(sent_a) + ['[SEP]'])
            src_b = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent_b) + ['[SEP]'])
            src = src_a + src_b
            seg = [0] * len(src_a) + [1] * len(src_b)
            mask = [1] * len(src)
            tgt = int(tgt)
            if len(src) > seq_length:
                src = src[: seq_length]
                seg = seg[: seq_length]
                mask = mask[: seq_length]
            while len(src) < seq_length:
                src.append(0)
                seg.append(0)
                mask.append(0)
            dataset.append((src, tgt, seg, mask))
    return dataset

def read_dataset(config, tokenizer):
    start = time.time()
    dataset = []
    seq_length = config['max_seq_len']
    print('>> load data:', config['data_path'])
    with open(config['data_path'], 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            id, sent_a, tgt = line.strip().split('\t')
            src_a = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(sent_a) + ['[SEP]'])
            src = src_a
            seg = [0] * len(src_a)
            mask = [1] * len(src)
            tgt = int(tgt)
            if len(src) > seq_length:
                src = src[: seq_length]
                seg = seg[: seq_length]
                mask = mask[: seq_length]
            while len(src) < seq_length:
                src.append(0)
                seg.append(0)
                mask.append(0)
            dataset.append((src, tgt, seg, mask))
    
    # data_cache_path = config['normal_data_cache_path']
    # if not os.path.exists(os.path.dirname(data_cache_path)):
    #     os.makedirs(os.path.dirname(data_cache_path))
    # with open(data_cache_path, 'wb') as f:
    #     pickle.dump(dataset, f)
    
    print("\n>> loading sentences from {},Time cost:{:.2f}".
          format(config['data_path'], ((time.time() - start) / 60.00)))

    return dataset


class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.
    weight = t.ones[]
    """
    # 这里的smoothing 应该再往大一点儿调，0.01 太小了。
    def __init__(self, smoothing=0.01,weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.step = 0

    def forward(self, x, target):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        # step1. 得到NLLLoss，也就是CrossEntropy 计算的值
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        
        # 使用weight计算损失
        nll_loss = nll_loss.squeeze(1)
        a = nll_loss.tolist()
        # print(f"当前各个样本的损失是：nll_loss={a}")
        temp = [id2label[i] for i in target.tolist()]
        # print(f"当前各个样本的标签是：label = {temp}")
        # step2. => 取均值
        smooth_loss = -log_probs.mean(dim=-1)
        if '3-5' in temp:
            idx = temp.index('3-5')
            special_loss = nll_loss[idx].item()
            viz.line( [special_loss],[self.step],win=win_3,update='append')
            self.step += 1
        # 为什么有个self.confidence * nll_loss?
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


'''
实现FocalLoss
01.考虑二分类=> 多分类扩展是否逻辑通顺？
'''
class FocalLoss(nn.Module):
    r'''
    alpha(1D Tensor, Variable) : the scalar factor for this criterion
    gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                            putting more focus on hard, misclassiﬁed examples
    size_average(bool): By default, the losses are averaged over observations for each minibatch.
                        However, if the field size_average is set to False, the losses are
                        instead summed for each minibatch.
    '''
    # size_average 是什么参数？
    def __init__(self, class_num,alpha=None,gamma=2,size_average=True):
        super().__init__()
        self.gamma = 2
        self.class_num = class_num
        self.size_average = size_average
        if alpha is None: # 这里的self.alpha 是torch
            self.alpha = torch.ones(class_num, 1).cuda() # size = [class_num,1]
        else:
            self.alpha = alpha

    def forward(self, inputs,targets):
        # 得到batch 和 class_num
        N = inputs.size(0) # batch_size 
        C = inputs.size(1) # class_num
        P = F.softmax(inputs) # 对输入做softmax，得到每个类别的logits
        # sf = torch.nn.Softmax()
        class_mask = inputs.data.new(N, C).fill_(0) # 按照inputs的shape创建一个初始值为0【可指定】的tensor
        # class_mask = Variable(class_mask)
        ids = targets.view(-1, 1) # 由一维变二维
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)] # 将alpha 的形状适配成inputs的 size = > (batch_size,1)

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

# 用于对抗训练
class FGM:
    def __init__(self, config, model):
        self.model = model
        self.backup = {}
        self.emb_name = config['emb_name']
        self.epsilon = config['epsilon']

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, config, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = config['epsilon']
        self.emb_name = config['emb_name']
        self.alpha = config['alpha']

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def block_shuffle(config, train_set):
    bs = config['batch_size'] * 100 # block 大小
    num_block = int(len(train_set) / bs) # 计算出一个block中的train data 个数
    slice_ = num_block * bs
    # 不一定能整除，所以需要使用切片找出到最后的一位下标
    train_set_to_shuffle = train_set[:slice_]
    train_set_left = train_set[slice_:]

    # 按照长度排序，这样有利于在后面tokenizer 的时候减少padding 
    sorted_train_set = sorted(train_set_to_shuffle, key=lambda i: len(i[0]))
    shuffled_train_set = []

    # 先将排好序的放到temp数组中，然后排序temp。 => 可以保证在shuffle的时候尽可能保证batch间的平衡
    tmp = []
    for i in range(len(sorted_train_set)):
        tmp.append(sorted_train_set[i])
        if (i+1) % bs == 0:
            random.shuffle(tmp)
            shuffled_train_set.extend(tmp)
            tmp = []

    # 对剩下的元素shuffle 
    random.shuffle(train_set_left)
    shuffled_train_set.extend(train_set_left)

    return shuffled_train_set


def build_model_and_tokenizer(config):
    tokenizer_path = config['model_path'] + '/vocab.txt'
    if config['tokenizer_fast']:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    if config['use_model'] == 'nezha':
        model = NeZhaSequenceClassification.from_pretrained(config['model_path'])
    # if config['use_model'] == 'bert':
    #     model = BertModel.from_pretrained(config['model_path'])
    return tokenizer, model

# 模型准确率
# 我这个计算方式是有问题的，因为评测脚本算的是macro F1，而不是单纯的precision，况且也不能只使用precision来衡量，而是用f1
def cal_precision(preds, labels):
    # torch.max dim=1找出每行最大值 返回value和index [1]为index
    # torch.eq 比较两个tensor是否相等
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()
    # acc = correct.sum().item() / len(correct)
    precision = correct.sum().item()
    # 返回precision 
    return precision

# 将获取到的所有标签计算出最后的macro_f1 值
def cal_f1(preds, trues):
    macro_f1 = f1_score(trues,preds,average='macro')    
    return macro_f1

def build_optimizer(config, model, train_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': config['weight_decay']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'], eps=1e-8)
    optimizer = Lookahead(optimizer, k=3, alpha=0.5)#TODO 读一下原论文
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * config['warmup_ratio'],
                                     t_total=train_steps)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    return optimizer, scheduler



def train():
    config = {
        'use_model': 'nezha',
        'normal_data_cache_path': '',  # 保存训练数据 下次加载更快
        'data_path': '/home/lawson/program/daguan/risk_data_grand/data/train_balance_90.txt', # 训练数据
        'output_path': '/home/lawson/program/daguan/risk_data_grand/model', # fine-tuning后保存模型的路径
        'model_path': '/home/lawson/program/daguan/pretrain_model/bert-base-fgm/2.4G+4.8M_large_10000_128_40000_checkpoint-50000', # your pretrain model path
        'shuffle_way': 'block_shuffle',  # block_shuffle 还是 random shuffle
        'use_swa': True, # 目前没有用到？？？
        'tokenizer_fast': False,
        'batch_size': 8,
        'num_epochs': 10,
        'max_seq_len': 100,
        'learning_rate': 2e-5,
        'alpha': 0.3,  # PGD的alpha参数设置 
        'epsilon': 1.0, # FGM的epsilon参数设置 
        'adv_k': 3, # PGD的训练次数
        'emb_name': 'word_embeddings.', 
        'adv': 'fgm', # 对抗训练方式
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'device': 'cuda',
        'logging_step': 50, # 每500步打印logger
        'seed': 124525601, # 随机种子
        'fold': 5, # k-flod  => 相当于使用0.2 的样本作为dev集
        'dev': False # 是否采用验证集
        }
    # 得到pretrain_model_name
    pretrain_model_name = config["model_path"].split("/")[-1]
    warnings.filterwarnings('ignore')
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))
    print("\n>> loading model from :{}".format(config['model_path']))
    
    tokenizer, model = build_model_and_tokenizer(config)
    # if not os.path.exists(config['normal_data_cache_path']):
    #     train_set = read_dataset(config, tokenizer)
    #     # train_set = read_single_dataset(config, tokenizer, 'data_path2')
    # else:
    #     with open(config['normal_data_cache_path'], 'rb') as f:
    #         train_set = pickle.load(f)
    # 确保每次读取的数据都是最新的
    train_set = read_dataset(config, tokenizer)
    seed_everything(config['seed'])

    if config['shuffle_way'] == 'block_shuffle':
        train_set = block_shuffle(config, train_set)
    else:
        random.shuffle(train_set)  # 这种shuffle效率不高

    train_num = len(train_set)

    # +1 大概率是为了 range() 函数使用
    train_steps = int(train_num * config['num_epochs'] / config['batch_size']) + 1
    
    optimizer, scheduler = build_optimizer(config, model, train_steps)
    model.to(config['device'])
    swa_model = AveragedModel(model)
    swa_start = 5
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    src = [example[0] for example in train_set]
    y = [example[1] for example in train_set]
    seg = [example[2] for example in train_set]
    mask = [example[3] for example in train_set]
    
    # train_data = TensorDataset(src, tgt, seg, mask)
    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config['batch_size'], num_workers=4)
    cudnn.benchmark = True
    
    if config['adv'] == '':
        print('\n>> start normal training ...')
    elif config['adv'] == 'fgm':
        print('\n>> start fgm training ...')
    elif config['adv'] == 'pgd':
        print('\n>> start pgd training ...')

    start = time.time()
    
    # 添加使用StratifiedKFold 的实现
    # skf = StratifiedKFold(n_splits=config['fold'],shuffle=True,random_state=config['seed'])    
    x = [] 
    for input_ids, type_ids, attention_masks in zip(src, seg, mask):
        x.append((input_ids, type_ids, attention_masks))    
    
    # 分割得到的结果是[X_train,y_train]为一对，为了方便前后对比，这里使用了一个random_state 保持每次划分一致
    # X表示的是输入，y表示的是标签；train表示的是训练集，test表示的验证集
    # 这里不使用 train_test_split() 是因为它是随机分割对模型最后的评测没有帮助  
    # X_train,X_test,y_train,y_test = train_test_split(kfold_dataset,tgt_numpy,test_size=0.15,random_state=22)
    
    x = np.array(x)
    y = np.array(y)
    # tgt_numpy 传入到函数中，因为shuffle，其值会被改变，所以这里要把x,y都传入【即保持同频的shuffle】
    if config['dev']:
        train_idx, dev_idx = split_data_by_class(x,y,rate=0.2,seed=config['seed'])
        # 转成numpy的格式        
        x_train , x_test = x[train_idx],x[dev_idx]
        y_train ,y_test = y[train_idx],y[dev_idx]
        
        # train
        src = torch.LongTensor([example[0] for example in x_train])
        seg = torch.LongTensor([example[1] for example in x_train])
        mask = torch.LongTensor([example[2] for example in x_train])
        tgt = torch.LongTensor(y_train)        
        
        # eval
        eval_src = torch.LongTensor([example[0] for example in x_test])
        eval_seg = torch.LongTensor([example[1] for example in x_test])
        eval_mask = torch.LongTensor([example[2] for example in x_test])
        eval_tgt = torch.LongTensor(y_test)   
        
    else:        
        src = torch.LongTensor([example[0] for example in x])
        seg = torch.LongTensor([example[1] for example in x])
        mask = torch.LongTensor([example[2] for example in x])
        tgt = torch.LongTensor(y) 
     

    # start train !!
    total_loss = 0.0
    global_steps = 0
    for epoch in range(1, config['num_epochs'] + 1):
        cur_avg_loss = 0.0
        model.train()        
        i = 1
        for (src_batch, tgt_batch, seg_batch, mask_batch) \
                in tqdm(batch_loader(config, src, tgt, seg, mask)):
            src_batch = src_batch.to(config['device'])
            tgt_batch = tgt_batch.to(config['device'])
            seg_batch = seg_batch.to(config['device'])
            mask_batch = mask_batch.to(config['device'])

            # lookahead.zero_grad()
            output = model(input_ids=src_batch, labels=tgt_batch,
                        token_type_ids=seg_batch, attention_mask=mask_batch)
            loss = output[0]
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            cur_avg_loss += loss.item()

            # 需要学习一下这个对抗训练的方法
            if config['adv'] == 'fgm':
                fgm = FGM(config, model)
                fgm.attack()
                adv_loss = model(input_ids=src_batch, labels=tgt_batch,
                                token_type_ids=seg_batch, attention_mask=mask_batch)[0]
                adv_loss.backward()
                fgm.restore()

            if config['adv'] == 'pgd':
                pgd = PGD(config, model)
                K = config['adv_k']
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    adv_loss = model(input_ids=src_batch, labels=tgt_batch,
                                    token_type_ids=seg_batch, attention_mask=mask_batch)[0]
                    adv_loss.backward()
                pgd.restore()
            optimizer.step()
            # lookahead.step()
            # if epoch > swa_start:
            #     swa_model.update_parameters(model)
            #     swa_scheduler.step() 
            # else:
            #     scheduler.step()
            scheduler.step()
            model.zero_grad()

            # 输出loss并画图 
            if (i + 1) % config['logging_step'] == 0:
                print("\n>> epoch - {}, epoch steps - {}, global steps - {}, "
                    "epoch avg loss - {:.4f}, global avg loss - {:.4f}, time cost - {:.2f} min".format
                    (epoch, i + 1, global_steps + 1, cur_avg_loss / config['logging_step'],
                    total_loss / (global_steps + 1),
                    (time.time() - start) / 60.00))
                viz.line([cur_avg_loss / config['logging_step']],[global_steps],win=win_1,update='append')
                cur_avg_loss = 0.0
            i +=1
            global_steps += 1
        
        if config['dev']:
            # train结束之后，在验证集上找一个最好的模型
            avg_loss ,all_label = [],[] # 其实这里precision,recall 是同一个值
            best_f1 = 0 # 保存最好的f1 值
            model.eval()
            # 因为是遍历单个batch，所以需要使用all_preds来记录每次得到的结果        
            all_preds = [] 
            for i, (src_batch,tgt_batch, seg_batch, mask_batch, ) in enumerate(tqdm(batch_loader(config, eval_src, eval_tgt, eval_seg, eval_mask))):
                src_batch = src_batch.to(config['device'])
                tgt_batch = tgt_batch.to(config['device'])
                seg_batch = seg_batch.to(config['device'])
                mask_batch = mask_batch.to(config['device'])
                with torch.no_grad():
                    output = model(input_ids=src_batch, labels=tgt_batch,
                                    token_type_ids=seg_batch, attention_mask=mask_batch)
                loss = output[0]
                avg_loss.append(loss.item())
                logits = torch.softmax(output[1], 1)            
                preds = torch.argmax(logits,-1)
                all_preds.extend(preds.tolist())    
                all_label.extend(tgt_batch.tolist())

            # 使用 f1_score 函数来计算 macro_f1 值
            macro_f1 = cal_f1(all_preds, all_label)      
            print(classification_report(all_label,all_preds,
                                        target_names=label_list
                                        # target_names=bad_clz
                                        )
                                        ) # 打印详细报告
            print("macro_f1 = ",macro_f1)
            viz.line([macro_f1],[epoch],win=win_2,update='append')
            if macro_f1 > best_f1 :# 如果效果好，则保存模型
                best_f1 = macro_f1
        macro_f1 = 0
        model_save_path = os.path.join(config['output_path'], f'{pretrain_model_name}_epoch_{epoch}_90')
        # if os.path.exists(model_save_path):
        #     os.remove(model_save_path)
        print('model_save_path:', model_save_path)
        # hasattr 用于判断对象是包含对应的属性，是返回true，否则返回false
        model_to_save = model.module if hasattr(model, 'module') else model
        print('\n>> model saved ... ...')
        model_to_save.save_pretrained(model_save_path)
        conf = json.dumps(config)
        out_conf_path = os.path.join(config['output_path'], f'{pretrain_model_name}_epoch_{epoch}_90' +
                                    '/train_config.json')
        with open(out_conf_path, 'w', encoding='utf-8') as f:
            f.write(conf)
            # else:
            #     early_stopping += 1
            #     print(f"Counter {early_stopping} of {config['early_stopping']}")
            #     if early_stopping > config['early_stopping']:
            #         print("Early stopping with best_f1: ", best_f1, "and val_f1 for this epoch: ", avg_f1, "...")
            #         break
        
    localtime_end = time.asctime(time.localtime(time.time()))
    print("\n>> program end at:{}".format(localtime_end))

# 验证函数
def eval():
    config = {
        'use_model': 'nezha',
        'normal_data_cache_path': '',  # 保存训练数据 下次加载更快
        'data_path': '/home/lawson/program/daguan/risk_data_grand/data/train_2000.txt', # 训练数据
        'output_path': '/home/lawson/program/daguan/risk_data_grand/model', # fine-tuning后保存模型的路径
        # 'model_path': '/home/lawson/program/daguan/risk_data_grand/model/best', # your pretrain model path => 使用large
        'model_path': '/home/lawson/program/daguan/risk_data_grand/model/checkpoint-0.692197906755471', 
        'shuffle_way': 'block_shuffle',  # block_shuffle 还是 random shuffle         
        'use_swa': True, # 目前没有用到？？？
        'tokenizer_fast': False, 
        'batch_size': 8,        
        'max_seq_len': 128,
        'alpha': 0.3,  # PGD的alpha参数设置 
        'epsilon': 1.0, # FGM的epsilon参数设置 
        'adv_k': 3, # PGD的训练次数
        'emb_name': 'word_embeddings.', 
        'device': 'cuda',        
        'seed': 124525601, # 随机种子         
        }

    warnings.filterwarnings('ignore')
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))
    print("\n>> loading model from :{}".format(config['model_path']))
    
    tokenizer, model = build_model_and_tokenizer(config)    
    # 确保每次读取的数据都是最新的
    train_set = read_dataset(config, tokenizer)
    seed_everything(config['seed']) # 这里对所有的都随机数都取了

    if config['shuffle_way'] == 'block_shuffle':
        train_set = block_shuffle(config, train_set)
    else:
        random.shuffle(train_set)  # 这种shuffle效率不高

    train_num = len(train_set)          
    model.to(config['device'])
    src = [example[0] for example in train_set]
    tgt = [example[1] for example in train_set]
    seg = [example[2] for example in train_set]
    mask = [example[3] for example in train_set]
    kfold_dataset = [] # 放到？？
    for input_ids, type_ids, attention_masks in zip(src, seg, mask):
        kfold_dataset.append((input_ids, type_ids, attention_masks))
    
    cudnn.benchmark = True
    
    kfold_dataset = np.array(kfold_dataset)
    tgt_numpy = np.array(tgt)
    # 分割得到的结果是[X_train,y_train]为一对，为了方便前后对比，这里使用了一个random_state 保持每次划分一致
    # X表示的是输入，y表示的是标签；train表示的是训练集，test表示的验证集
    X_train,X_test,y_train,y_test = train_test_split(kfold_dataset,tgt_numpy,test_size=0.15,random_state=22)
    print(X_test[1][0])
    # eval
    eval_src = torch.LongTensor([example[0] for example in X_test])
    eval_seg = torch.LongTensor([example[1] for example in X_test])
    eval_mask = torch.LongTensor([example[2] for example in X_test])
    eval_tgt = torch.LongTensor(y_test)    

    all_label = [] 
    model.eval()
    # 因为是遍历单个batch，所以需要记录每次得到的结果
    # 为什么这里的batch_size = 3？？？
    all_preds = [] # 存储所有的预测结果
    for i, (src_batch,tgt_batch, seg_batch, mask_batch, ) in enumerate(tqdm(batch_loader(config, eval_src, eval_tgt, eval_seg, eval_mask))):
        src_batch = src_batch.to(config['device'])
        tgt_batch = tgt_batch.to(config['device'])
        seg_batch = seg_batch.to(config['device'])
        mask_batch = mask_batch.to(config['device'])
        with torch.no_grad():
            output = model(input_ids=src_batch, labels=tgt_batch,
                            token_type_ids=seg_batch, attention_mask=mask_batch)
        loss = output[0]        
        logits = torch.softmax(output[1], 1)
        preds = torch.argmax(logits,-1)
        all_preds.extend(preds.tolist())    
        all_label.extend(tgt_batch.tolist())

    # 使用 f1_score 函数来计算值
    macro_f1 = cal_f1(all_preds, all_label)
    print(classification_report(all_preds, all_label)) # 打印报告
    print("macro_f1 = ",macro_f1)
    localtime_end = time.asctime(time.localtime(time.time()))
    print("\n>> program end at:{}".format(localtime_end))

if __name__ == '__main__':
    train()
    # eval() 
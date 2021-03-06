# coding:utf-8
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import csv
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import jionlp as jio
import torch
import torch.nn as nn
from transformers import BertTokenizer

from modeling.modeling_nezha.modeling import NeZhaPreTrainedModel, NeZhaModel
from modeling.bert.modeling_bert import BertForMaskedLM, BertModel, BertPreTrainedModel
#from model_utils.ensemble_model import EnsembleModel

label_dict = {'5-24': 0, '6-34': 1, '1-1': 2, '6-8': 3, '10-26': 4, '2-3': 5, '5-22': 6, '6-28': 7, '8-18': 8, '1-4': 9, '2-6': 10, '6-21': 11, 
              '7-16': 12, '6-29': 13, '6-20': 14, '6-15': 15, '6-13': 16, '9-23': 17, '5-35': 18, '2-33': 19, '5-30': 20, '1-9': 21, '8-27': 22, 
              '1-10': 23, '6-19': 24, '3-5': 25, '2-2': 26, '4-7': 27, '2-17': 28, '5-12': 29, '6-32': 30, '6-31': 31, '2-25': 32, '2-11': 33, '2-14': 34}
label_list = ['5-24', '6-34', '1-1', '6-8', '10-26', '2-3', '5-22', '6-28', '8-18', '1-4', '2-6', '6-21', '7-16', '6-29', '6-20', 
              '6-15', '6-13', '9-23', '5-35', '2-33', '5-30', '1-9', '8-27', '1-10', '6-19', '3-5', '2-2', '4-7', '2-17', '5-12', 
              '6-32', '6-31', '2-25', '2-11', '2-14']

bad_clz2id={'6-20': 0, '5-24': 1, '1-1': 2, '6-8': 3, '5-22': 4, '6-13': 5, '1-9': 6, '6-31': 7, '2-11': 8, '2-14': 9, '6-19': 10, '6-28': 11, '4-7': 12, '3-5': 13, '8-27': 14, '7-16': 15, '8-18': 16, '9-23': 17, '10-26': 18, '2-17': 19}

bad_id2clz = {0: '6-20', 1: '5-24', 2: '1-1', 3: '6-8', 4: '5-22', 5: '6-13', 6: '1-9', 7: '6-31', 8: '2-11', 9: '2-14', 10: '6-19', 11: '6-28', 12: '4-7', 13: '3-5', 14: '8-27', 15: '7-16', 16: '8-18', 17: '9-23', 18: '10-26', 19: '2-17'}

bad_clz = ['6-20','5-24','1-1','6-8','5-22','6-13',
            '1-9','6-31','2-11','2-14','6-19','6-28','4-7',  # 13

        # less sample                 
            '3-5','8-27','7-16','8-18','9-23','10-26','2-17' # 7
            ]
class NeZhaSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 35
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.multi_drop = 5
        self.multi_dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(self.multi_drop)])
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
        # pooled_output = outputs.logits
        logits = self.classifier(pooled_output)
        for j, dropout in enumerate(self.multi_dropouts):
            if j == 0:
                logits = self.classifier(dropout(pooled_output)) / self.multi_drop
            else:
                logits += self.classifier(dropout(pooled_output)) / self.multi_drop
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = LabelSmoothingLoss(smoothing=0.01)
            # loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


def argmax(res):
    k, tmp = 0, 0
    for i in range(len(res)):
        if res[i] > tmp:
            tmp = res[i]
            k = i

    return k

def batch_loader(config, src, seg, mask):
    ins_num = src.size()[0]
    batch_size = config['batch_size']
    for i in range(ins_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        mask_batch = mask[i * batch_size: (i + 1) * batch_size, :]
        yield src_batch, seg_batch, mask_batch
    if ins_num > ins_num // batch_size * batch_size:
        src_batch = src[ins_num // batch_size * batch_size:, :]
        seg_batch = seg[ins_num // batch_size * batch_size:, :]
        mask_batch = mask[ins_num // batch_size * batch_size:, :]
        yield src_batch, seg_batch, mask_batch

class LabelSmoothingLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """
    
    def __init__(self, smoothing=0.01):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def read_dataset(config):
    start = time.time()
    tokenizer = BertTokenizer.from_pretrained(config['load_model_path'])
    dataset, r_dataset = [], []
    seq_length = config['max_seq_len']

    with open(config['test_path'], 'r', encoding='utf-8') as f:
        for line_id, line in enumerate(f):
            id, sent_a = line.strip().split('\t')
            src_a = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(sent_a) + ["[SEP]"])
            src = src_a 
            seg = [0] * len(src_a)
            mask = [1] * len(src)

            if len(src) > seq_length:
                src = src[: seq_length]
                seg = seg[: seq_length]
                mask = mask[: seq_length]

            while len(src) < seq_length:
                src.append(0)
                seg.append(0)
                mask.append(0)
            dataset.append((src, seg, mask, id))

        # data_cache_path = config['normal_data_cache_path']
        # if not os.path.exists(os.path.dirname(data_cache_path)):
        #     os.makedirs(os.path.dirname(data_cache_path))
        # with open(data_cache_path, 'wb') as f:
        #     pickle.dump(dataset, f)

    print("\n>>> loading sentences from {}, time cost:{:.2f}".
          format(config['test_path'], (time.time() - start) / 60.00))

    return dataset


def predict(dataset, pre_model, config):
    
    src = torch.LongTensor([sample[0] for sample in dataset])
    seg = torch.LongTensor([sample[1] for sample in dataset])
    mask = torch.LongTensor([sample[2] for sample in dataset])
    id = [sample[3] for sample in dataset]
    predict_all = np.array([], dtype=int)
    for (src_batch, seg_batch, mask_batch) in \
            tqdm(batch_loader(config, src, seg, mask)):
        src_batch = src_batch.to(config['device'])
        seg_batch = seg_batch.to(config['device'])
        mask_batch = mask_batch.to(config['device'])
        with torch.no_grad():
            output = pre_model(input_ids=src_batch, token_type_ids=seg_batch, attention_mask=mask_batch)

        logits = output[0]
        logits = torch.softmax(logits, 1)
        pred = torch.max(logits, dim=1)[1].cpu().data.numpy()
        predict_all = np.append(predict_all, pred)

    predict_all = predict_all.tolist()
    label = [label_list[label] for label in predict_all]
    # label = [bad_clz[label] for label in predict_all]
    res = pd.DataFrame({'id':id,
                        'label':label})
    res.to_csv(config['submit_path'],index=False)


def main():
    config = {
        'model_type': 'bert-base-fgm', # ?????????????????????                
        # 'vocab_path': '/home/lawson/program/daguan/risk_data_grand/model/best/vocab.txt',
        'test_path': '/home/lawson/program/daguan/risk_data_grand/data/test.txt', # ????????????
        'load_model_path': '/home/lawson/program/daguan/risk_data_grand/model/2.4G+4.8M_large_10000_128_40000_checkpoint-50000_epoch_10_90', # ????????????????????????
        'submit_path': 'submission_90.csv', # ????????????????????????
        'batch_size': 8,
        'max_seq_len': 100, 
        'device': 'cuda',
    }

    warnings.filterwarnings('ignore')
    start_time = time.time()
    localtime_start = time.asctime(time.localtime(time.time()))
    print(">> program start at:{}".format(localtime_start))
        
    # config['normal_data_cache_path'] = '/home/lawson/program/daguan/user_data/processed/' + config['model_type'] + '/test_data.pkl'
    
    # if not os.path.exists(config['normal_data_cache_path']):
    test_set = read_dataset(config)
    # else:
    #     with open(config['normal_data_cache_path'], 'rb') as f:
    #         test_set = pickle.load(f)

    print("\n>> start predict ... ...")
    
    # max_step = 0
    # for checkpoint_name in os.listdir('/home/lawson/program/daguan/risk_data_grand/output_model'):
    #     checkpoint, step = checkpoint_name.split('-')
    #     if int(step) > max_step:
    #         max_step= int(step)
    print('>> load model: ',config['load_model_path'])
    
    model = NeZhaSequenceClassification.from_pretrained(config['load_model_path'])
    # model = BertForMaskedLM.from_pretrained(config['load_model_path'])
    model.to(config['device'])
    model.eval()

    predict(dataset=test_set, pre_model=model, config=config)
    
    localtime_end = time.asctime(time.localtime(time.time()))
    print("\n>> program end at : {}, total cost time : {:.2f}".
          format(localtime_end, (time.time() - start_time) / 60.00))


if __name__ == '__main__':
    main()

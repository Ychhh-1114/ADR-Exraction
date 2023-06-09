# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/11 15:20
# @File    : process_rel.py

"""
file description:：

"""
import json
import torch
from config_rel import ConfigRel, USE_CUDA
from process import process_data
import copy
from transformers import BertTokenizer
import codecs
from collections import defaultdict


class DataPreparationRel:
    def __init__(self, config):
        self.config = config
        # self.get_token2id()
        self.rel_cnt = defaultdict(int)
    
    def get_data(self, file_path, is_test=False):
        if is_test == False:
            data = []
            datas = process_data(file_path)

            for data_item in datas:
                spo_list = data_item['spo_list']
                text = data_item['text']
                text = text.lower()
                for spo_item in spo_list:
                    subject = spo_item["subject"]
                    subject = subject.lower()
                    object = spo_item["object"]
                    object = object.lower()
                    # 增加位置信息
                    # index_s = text.index(subject)
                    # index_o = text.index(object)
                    # position_s, position_o = [], []
                    # for i, word in enumerate(text):
                    #     if word not in self.word2id:
                    #         continue
                    #     position_s.append(i-index_s+self.config.max_seq_length*2)
                    #     position_o.append(i-index_o+self.config.max_seq_length*2)
                    if not is_test:
                        relation = spo_item['predicate']
                        if self.rel_cnt[relation] > self.config.rel_num:
                            continue
                        self.rel_cnt[relation] += 1

                    else:
                        relation = []
                    # sentence_cls = '$'.join([subject, object, text.replace(subject, '#'*len(subject)).replace(object, '#'*len(object))])
                    sentence_cls = ''.join([subject, object, text])
                    # sentence_cls = text
                    item = {'sentence_cls': sentence_cls, 'relation': relation, 'text': text,
                            'subject': subject, 'object': object}  # 'position_s': position_s, 'position_o': position_o}
                    data.append(item)
                # 添加负样本
                sentence_neg = '$'.join([object, subject, text])
                # sentence_neg = '$'.join(
                #     [object, subject, text.replace(subject, '#' * len(subject)).replace(object, '#' * len(object))])
                item_neg = {'sentence_cls': sentence_neg, 'relation': 'N', 'text': text,
                            'subject': object, 'object': subject}
                data.append(item_neg)

            dataset = Dataset(data)
            if is_test:
                dataset.is_test = True

            data_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.config.batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                drop_last=False
            )

            return data_loader


    def get_data(self,datas,is_test = True):
        data = []
        for data_item in datas:
            spo_list = data_item['spo_list']
            text = data_item['text']
            text = text.lower()
            for spo_item in spo_list:

                subject = spo_item["subject"]
                subject = subject.lower()
                object = spo_item["object"]
                object = object.lower()
                # 增加位置信息
                # index_s = text.index(subject)
                # index_o = text.index(object)
                # position_s, position_o = [], []
                # for i, word in enumerate(text):
                #     if word not in self.word2id:
                #         continue
                #     position_s.append(i-index_s+self.config.max_seq_length*2)
                #     position_o.append(i-index_o+self.config.max_seq_length*2)
                if not is_test:
                    relation = spo_item['predicate']
                    if self.rel_cnt[relation] > self.config.rel_num:
                        continue
                    self.rel_cnt[relation] += 1

                else:
                    relation = []
                # sentence_cls = '$'.join([subject, object, text.replace(subject, '#'*len(subject)).replace(object, '#'*len(object))])
                sentence_cls = ''.join([subject, object, text])
                # sentence_cls = text
                item = {'sentence_cls': sentence_cls, 'relation': relation, 'text': text,
                        'subject': subject, 'object': object}  # 'position_s': position_s, 'position_o': position_o}
                data.append(item)
            # 添加负样本
            sentence_neg = '$'.join([object, subject, text])
            # sentence_neg = '$'.join(
            #     [object, subject, text.replace(subject, '#' * len(subject)).replace(object, '#' * len(object))])
            item_neg = {'sentence_cls': sentence_neg, 'relation': 'N', 'text': text,
                        'subject': object, 'object': subject}
            data.append(item_neg)

        dataset = Dataset(data)
        if is_test:
            dataset.is_test = True

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            drop_last=True
        )

        return data_loader






    def get_token2id(self):
        self.word2id = {}
        with codecs.open('../data/vec.txt', 'r', encoding='utf-8') as f:
            cnt = 0
            for line in f.readlines():
                self.word2id[line.split()[0]] = cnt
                cnt += 1

    def get_train_dev_data(self, path_train=None, path_dev=None, path_test=None, is_test=False):
        train_loader, dev_loader, test_loader = None, None, None
        if path_train is not None:
            train_loader = self.get_data(path_train)
        if path_dev is not None:
            dev_loader = self.get_data(path_dev)
        if path_test is not None:
            test_loader = self.get_data(path_test, is_test=True)
    
        return train_loader, dev_loader, test_loader

    def get_train_dev_data(self,data, is_test=True):
        test_loader = self.get_data(data)

        return  test_loader
    
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.is_test = False
        with open('rel2id.json', 'r', encoding='utf-8') as f:
            self.rel2id = json.load(f)
        vocab_file = 'biobert-base-cased-v1.2'
        self.bert_tokenizer = BertTokenizer.from_pretrained(vocab_file)

        # vocab_file = '../data/vec.txt'
        # self.get_token2id()

    # def get_token2id(self):
    #     self.word2id = {}
    #     with codecs.open('../data/vec.txt', 'r', encoding='utf-8') as f:
    #         cnt = 0
    #         for line in f.readlines():
    #             self.word2id[line.split()[0]] = cnt
    #             cnt += 1

    def __getitem__(self, index):
        sentence_cls = self.data[index]['sentence_cls']
        relation = self.data[index]['relation']
        text = self.data[index]['text']
        subject = self.data[index]['subject']
        object = self.data[index]['object']
        # position_s = self.data[index]['position_s']
        # position_o = self.data[index]['position_o']
        
        data_info = {}
        for key in self.data[0].keys():
            if key in locals():
                data_info[key] = locals()[key]
        
        return data_info
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, data_batch):

        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            max_length = max(lengths)
            # padded_seqs = torch.zeros(len(sequences), max_length)
            padded_seqs = torch.zeros(len(sequences), max_length)
            tmp_pad = torch.ones(1, max_length)
            mask_tokens = torch.zeros(len(sequences), max_length)
            for i, seq in enumerate(sequences):
                end = lengths[i]
                seq = torch.LongTensor(seq)
                if len(seq) != 0:
                    padded_seqs[i, :end] = seq[:end]
                    mask_tokens[i, :end] = tmp_pad[0, :end]
            
            return padded_seqs, mask_tokens
        
        item_info = {}
        for key in data_batch[0].keys():
            item_info[key] = [d[key] for d in data_batch]

        # 转化为数值
        sentence_cls = [self.bert_tokenizer.encode(sentence, add_special_tokens=True) for sentence in item_info['sentence_cls']]
        # sentence_cls = [[] for _ in range(len(item_info['sentence_cls']))]
        # for i, sentence in enumerate(item_info['sentence_cls']):
        #     tmp = []
        #     for c in sentence:
        #         if c in self.word2id:
        #             tmp.append(self.word2id[c])
        #     sentence_cls[i] = tmp

        if not self.is_test:
            relation = torch.Tensor([self.rel2id[rel] for rel in item_info['relation']]).to(torch.int64)

        
        # 批量数据对齐
        sentence_cls, mask_tokens = merge(sentence_cls)
        sentence_cls = sentence_cls.to(torch.int64)
        mask_tokens = mask_tokens.to(torch.int64)


        # relation = relation.to(torch.int64)
        # position_s, _ = merge(item_info['position_s'])
        # position_o, _ = merge(item_info['position_o'])
        if USE_CUDA:
            sentence_cls = sentence_cls.contiguous().cuda()
            mask_tokens = mask_tokens.contiguous().cuda()
            # position_s = position_s.contiguous().cuda()
            # position_o = position_o.contiguous().cuda()
        else:
            sentence_cls = sentence_cls.contiguous()
            mask_tokens = mask_tokens.contiguous()
            # position_s = position_s.contiguous()
            # position_o = position_o.contiguous()
        if not self.is_test:
            if USE_CUDA:
                relation = relation.contiguous().cuda()
            else:
                relation = relation.contiguous()

        data_info = {"mask_tokens": mask_tokens.to(torch.uint8)}
        data_info['text'] = item_info['text']
        data_info['subject'] = item_info['subject']
        data_info['object'] = item_info['object']
        for key in item_info.keys():
            if key in locals():
                data_info[key] = locals()[key]
        
        return data_info
        
        
if __name__ == '__main__':
    config = ConfigRel()
    process = DataPreparationRel(config)
    train_loader, dev_loader, test_loader = process.get_train_dev_data('adr-train.csv')
    
    for item in train_loader:
        print(item)
        exit()
    


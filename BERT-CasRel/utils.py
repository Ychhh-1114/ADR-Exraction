import torch.utils.data as data
import pandas as pd
import random
from config import *
import json
from process import process_data
from transformers import BertTokenizerFast
import numpy as np

#返回一个rel2id 和id2rel
def get_rel():
    df = pd.read_csv(REL_PATH, names=['rel', 'id'])
    return df['rel'].tolist(), dict(df.values)

#生成长度为len，hot_pos位置为1其余位置为0的独热编码
def multihot(length, hot_pos):
    return [1 if i in hot_pos else 0 for i in range(length)]



class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        _, self.rel2id = get_rel()
        # 加载文件
        # 加载文件

        if type == 'train':
            file_path = TRAIN_PATH
        elif type == 'test':
            file_path = TEST_PATH

        self.lines = process_data(file_path)

        # 加载bert
        self.tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        info = self.lines[index]
        # print(type(info))
        # exit()
        tokenized = self.tokenizer(info['text'], return_offsets_mapping=True)
        info['input_ids'] = tokenized['input_ids']
        info['offset_mapping'] = tokenized['offset_mapping']
        return self.parse_json(info)


    def get_pos_id(self, source, elem):
        for head_id in range(len(source)):
            tail_id = head_id + len(elem)
            if source[head_id:tail_id] == elem:
                return head_id, tail_id - 1


    def collate_fn(self,batch):
        #获得最长的句子长度
        batch.sort(key=lambda x : len(x["input_ids"]),reverse=True)
        max_len = len(batch[0]["input_ids"])
        batch_text = {
            'text': [],
            'input_ids': [],
            'offset_mapping': [],
            'triple_list': [],
        }
        batch_mask = []
        batch_sub = {
            'heads_seq': [],
            'tails_seq': [],
        }
        batch_sub_rnd = {
            'head_seq': [],
            'tail_seq': [],
        }
        batch_obj_rel = {
            'heads_mx': [],
            'tails_mx': [],
        }

        #对batch中的每一个item进行处理
        for item in batch:
            input_ids = item["input_ids"]  #对元素进行pad填充
            item_len = len(input_ids)
            pad_len = max_len - item_len
            input_ids = input_ids + [0] * pad_len
            mask = [1] * item_len + [0] * pad_len
            # print(mask)
            # exit()

            #对subject的位置进行填充
            sub_head_seq = multihot(max_len,item["sub_head_ids"])
            sub_tail_seq = multihot(max_len, item["sub_tail_ids"])
            # print(item["sub_head_ids"])
            # print(sub_head_seq)
            # exit()
            #随机选择一个subject
            if len(item['triple_id_list']) == 0:  #如果没有三元组则continue
                continue
            sub_rnd = random.choice(item['triple_id_list'])[0]
            sub_rnd_head_seq = multihot(max_len, [sub_rnd[0]])
            sub_rnd_tail_seq = multihot(max_len, [sub_rnd[1]])

            #根据随机subject计算relations矩阵
            obj_head_mx = [[0] * REL_SIZE for _ in range(max_len)]   #生成两个二维全0矩阵（一个head矩阵一个tail矩阵）
            obj_tail_mx = [[0] * REL_SIZE for _ in range(max_len)]

            for triple in item["triple_id_list"]:                    #对全0矩阵进行填充获得obj的head和tail的rel矩阵
                rel_id = triple[1]
                head_id, tail_id = triple[2]
                if triple[0] == sub_rnd:
                    obj_head_mx[head_id][rel_id] = 1
                    obj_tail_mx[tail_id][rel_id] = 1

            #重新组装batch，一条item压入一组信息
            batch_text["text"].append(item["text"])
            batch_text["input_ids"].append(input_ids)
            batch_text["offset_mapping"].append(item["offset_mapping"])
            batch_text["triple_list"].append(item["triple_list"])

            batch_mask.append(mask)

            batch_sub["heads_seq"].append(sub_head_seq)
            batch_sub["tails_seq"].append(sub_tail_seq)

            batch_sub_rnd["head_seq"].append(sub_rnd_head_seq)
            batch_sub_rnd["tail_seq"].append(sub_rnd_tail_seq)

            batch_obj_rel["heads_mx"].append(obj_head_mx)
            batch_obj_rel["tails_mx"].append(obj_tail_mx)

        return batch_mask,(batch_text,batch_sub_rnd),(batch_sub,batch_obj_rel)




    def parse_json(self, info):  #对json串进行解析

        text = info['text']
        input_ids = info['input_ids']
        dct = {
            'text': text,
            'input_ids': input_ids,
            'offset_mapping': info['offset_mapping'],
            'sub_head_ids': [],
            'sub_tail_ids': [],
            'triple_list': [],
            'triple_id_list': []
        }

        spo = info["spo_list"]
        subject = spo['subject']
        object = spo['object']
        predicate = spo['predicate']


        dct['triple_list'].append((subject, predicate, object))

        tokenized = self.tokenizer(subject, add_special_tokens=False)
        sub_token = tokenized['input_ids']
        sub_pos_id = self.get_pos_id(input_ids, sub_token)

        sub_head_id, sub_tail_id = sub_pos_id
        # 计算 object 实体位置
        tokenized = self.tokenizer(object, add_special_tokens=False)
        obj_token = tokenized['input_ids']
        obj_pos_id = self.get_pos_id(input_ids, obj_token)

        obj_head_id, obj_tail_id = obj_pos_id
        # 数据组装
        dct['sub_head_ids'].append(sub_head_id)
        dct['sub_tail_ids'].append(sub_tail_id)

        dct['triple_id_list'].append((
            [sub_head_id, sub_tail_id],
            self.rel2id[predicate],
            [obj_head_id, obj_tail_id],
        ))

        return dct

if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, shuffle=False, batch_size=2, collate_fn=dataset.collate_fn)
    print(next(iter(loader)))
    exit()
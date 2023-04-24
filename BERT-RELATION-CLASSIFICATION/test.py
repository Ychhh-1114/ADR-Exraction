import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
import warnings
import torch
import time
import argparse

import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

setup_seed(44)

from transformers import BertModel

from loader import map_id_rel, load_dev

rel2id, id2rel = map_id_rel()

print(len(rel2id))
print(id2rel)

USE_CUDA = torch.cuda.is_available()

def get_train_args():
    labels_num=len(rel2id)
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=1,help = '每批数据的数量')
    parser.add_argument('--nepoch',type=int,default=30,help = '训练的轮次')
    parser.add_argument('--lr',type=float,default=0.001,help = '学习率')
    parser.add_argument('--gpu',type=bool,default=True,help = '是否使用gpu')
    parser.add_argument('--num_workers',type=int,default=2,help='dataloader使用的线程数量')
    parser.add_argument('--num_labels',type=int,default=len(id2rel),help='分类类数')
    parser.add_argument('--data_path',type=str,default='./data',help='数据路径')
    opt=parser.parse_args()
    print(opt)
    return opt




# def test(net,text_list,ent1_list,ent2_list,result):
#     net.eval()
#     max_length=128
#
#     net=torch.load('bert-fc.pth')
#     rel_list=[]
#     with torch.no_grad():
#         for text,ent1,ent2,label in zip(text_list,ent1_list,ent2_list,result):
#             sent = ent1 + ent2+ text
#             tokenizer = BertTokenizer.from_pretrained('./bert-model/biobert-base-cased-v1.2')
#             indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
#             avai_len = len(indexed_tokens)
#             while len(indexed_tokens) < max_length:
#                 indexed_tokens.append(0)  # 0 is id for [PAD]
#             indexed_tokens = indexed_tokens[: max_length]
#             indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
#
#             # Attention mask
#             att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
#             att_mask[0, :avai_len] = 1
#             if USE_CUDA:
#                 indexed_tokens = indexed_tokens.cuda()
#                 att_mask = att_mask.cuda()
#
#             if USE_CUDA:
#                 indexed_tokens=indexed_tokens.cuda()
#                 att_mask=att_mask.cuda()
#             outputs = net(indexed_tokens, attention_mask=att_mask)
#             # print(y)
#             logits = outputs[0]
#             _, predicted = torch.max(logits.data, 1)
#             result=predicted.cpu().numpy().tolist()[0]
#             print("Source Text: ",text)
#             print("Entity1: ",ent1," Entity2: ",ent2," Predict Relation: ",id2rel[result]," True Relation: ",label)
#             print('\n')
#             rel_list.append(id2rel[result])
#     return rel_list
# opt = get_train_args()
#
# model=get_model(opt)


from random import choice

text_list=[]
ent1=[]
ent2=[]
result=[]

data=load_dev()
dev_text=data['text']
dev_mask=data['mask']
dev_label=data['label']

dev_text = [ t.numpy() for t in dev_text]
dev_mask = [ t.numpy() for t in dev_mask]

dev_text=torch.tensor(dev_text)
dev_mask=torch.tensor(dev_mask)
dev_label=torch.tensor(dev_label)

batch_size = 20
dev_dataset = torch.utils.data.TensorDataset(dev_text,dev_mask,dev_label)

dev_iter = torch.utils.data.DataLoader(dev_dataset, batch_size, shuffle=True)

def get_model():
    labels_num=len(rel2id)
    from model import BERT_Classifier
    model = BERT_Classifier(labels_num)
    return model


class BERT_Classifier(nn.Module):
    def __init__(self,label_num):
        super().__init__()
        self.encoder = BertModel.from_pretrained('./bert-model/biobert-base-cased-v1.2')
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.1,inplace=False)
        self.fc = nn.Linear(768, label_num)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, attention_mask ,label=None):
        x = self.encoder(x, attention_mask=attention_mask)[0]
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)
        if label == None:
            return None,x
        else:
            return self.criterion(x,label),x


net = torch.load('bert-fc.pth', map_location=torch.device('cpu'))
correct = 0
total=0
iter = 0

for text, mask, y in dev_iter:
    iter += 1
    # print(type(y))
    # print(y)
    if text.size(0) != batch_size:
        break
    text = text.reshape(batch_size, -1)
    mask = mask.reshape(batch_size, -1)
    if USE_CUDA:
        text = text.cuda()
        mask = mask.cuda()
        y = y.cuda()
    # print(text.shape)
    print("hello")
    loss, logits = net(text, mask, y)
    # print(y)
    # print(loss.shape)
    # print("predicted",predicted)
    # print("answer", y)
    # print(outputs[1].shape)
    # print(output)
    # print(outputs[1])
    _, predicted = torch.max(logits.data, 1)
    total += text.size(0)
    correct += predicted.data.eq(y.data).cpu().sum()

    if iter % 5 == 0:
        print("iter: ",iter)

loss = loss.detach().cpu()
print( " loss: ", loss.mean().numpy().tolist(), "right", correct.cpu().numpy().tolist(),
      "total", total, "Acc:", correct.cpu().numpy().tolist() / total)




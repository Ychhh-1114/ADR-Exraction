from utils import *
from predict import predict
from torch.utils import data
import torch
import json
import sys
import numpy as np
import torch.nn as nn
from nets.gpNet import RawGlobalPointer, sparse_multilabel_categorical_crossentropy
from transformers import BertTokenizerFast, BertModel,AutoTokenizer, AutoModelForMaskedLM
from utils.dataloader import data_generator, load_name,data_generatorTest
from torch.utils.data import DataLoader
import configparser
from torch.utils.tensorboard import SummaryWriter
from utils.bert_optimization import BertAdam


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# 忽略 transformers 警告
from transformers import logging
logging.set_verbosity_error()


device = 'cuda' if torch.cuda.is_available() else 'cpu'



if __name__ == '__main__':
    # model = torch.load(MODEL_DIR + 'model_48.pth', map_location=DEVICE)
    # dataset = Dataset('test')

    con = configparser.ConfigParser()
    con.read('./config.ini', encoding='utf8')
    args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
    tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)
    encoder = BertModel.from_pretrained(args_path["model_path"])

    with open(args_path["schema_data"], 'r', encoding='utf-8') as f:
        schema = {}
        for idx, item in enumerate(f):
            item = json.loads(item.rstrip())
            schema[item["subject_type"] + "_" + item["predicate"] + "_" + item["object_type"]] = idx
    id2schema = {}
    for k, v in schema.items(): id2schema[v] = k


    train_data = data_generatorTest(load_name(args_path["test_file"]))

    correct_num, predict_num, gold_num = 0, 0, 0
    pred_triple_list = []
    true_triple_list = []
    pred_y = []
    true_y = []
    print(len(train_data))
    # exit()
    # train_data = train_data[:10]
    for index,item in enumerate(train_data):
        pred_items = predict(item["text"])
        for index,each in enumerate(pred_items):
            pred_y.append((each["subject"],"causes",each["object"]))
        true_y.append((item["spo_list"][0],item["spo_list"][1],item["spo_list"][2]))


    correct_num += len(set(true_y) & set(pred_y))
    predict_num += len(set(pred_y))
    gold_num += len(set(true_y))
    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    print('\tcorrect_num:', correct_num, 'predict_num:', predict_num, 'gold_num:', gold_num)
    print('\tprecision:%.3f' % precision, 'recall:%.3f' % recall, 'f1_score:%.3f' % f1_score)


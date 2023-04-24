# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/11 19:34
# @File    : demo.py

"""
file description:

"""
import torch
from model_ner import SeqLabel
from model_rel import AttBiLSTM
from config_ner import ConfigNer, USE_CUDA
from config_rel import ConfigRel

from process_ner import ModelDataPreparation
from process_rel import DataPreparationRel

import trainer_ner, trainer_rel
import json
from transformers import BertForSequenceClassification

import torch.nn as nn


def merge_spo(so, pre):
    data = []
    for index, each in enumerate(so):

        spolist = each["spo_list"]
        for each_spo in spolist:
            spo = {
                "subject": "",
                "object": [],
                "predicate": ""
            }
            # print("111",each)
            # if each["subject"] not in
            spo["subject"] = each_spo["subject"]
            spo["object"].append(each_spo["object"])
            spo["predicate"] = pre[index]
            print(spo)
            data.append(spo)



    return merge_list(data)


def merge_list(datalist):
    text_list = []
    data_list = []
    for each in datalist:
        print(each)
        if each["predicate"] == "causes":
            if each["subject"] not in text_list:
                text_list.append(each["subject"])
                data_list.append(each)
            else:
                index = text_list.index(each["subject"])
                data_list[index]["object"].append(each["object"][0])

    return data_list


def set_data(datalist):
    text_list = []
    data_list = []
    for each in datalist:
        spo = {
            "text": "",
            "spo_list": []
        }
        if each["text"] not in text_list:
            spo["text"] = each["text"]
            spo["spo_list"].append(each["spo_list"])
            text_list.append(each["text"])
            data_list.append(spo)
        else:
            index = text_list.index(each["text"])
            data_list[index]["spo_list"].append(each["spo_list"])

    return data_list


def get_entities(pred_ner, text):
    token_types = [[] for _ in range(len(pred_ner))]
    entities = [[] for _ in range(len(pred_ner))]
    for i in range(len(pred_ner)):
        token_type = []
        entity = []
        j = 0
        word_begin = False
        while j < len(pred_ner[i]):

            if pred_ner[i][j][0] == 'B':
                if word_begin:
                    token_type = []  # 防止多个B出现在一起
                    entity = []
                token_type.append(pred_ner[i][j])
                if j > len(text[i]):
                    continue
                entity.append(text[i][j])
                word_begin = True
            elif pred_ner[i][j][0] == 'I':
                if word_begin:
                    token_type.append(pred_ner[i][j])
                    entity.append(text[i][j])
            else:
                if word_begin:
                    token_types[i].append(''.join(token_type))
                    token_type = []
                    entities[i].append(''.join(entity))
                    entity = []
                word_begin = False
            j += 1
    return token_types, entities


def test():
    test_path = 'test.csv'
    PATH_NER = 'ner.pth'

    #
    config_ner = ConfigNer()
    # config_ner.batch_size = 5
    ner_model = SeqLabel(config_ner)
    ner_model_dict = torch.load(PATH_NER, map_location=torch.device('cpu'))
    ner_model.load_state_dict(ner_model_dict['state_dict'])

    ner_data_process = ModelDataPreparation(config_ner)
    _, _, test_loader = ner_data_process.get_train_dev_data(path_test=test_path)
    trainerNer = trainer_ner.Trainer(ner_model, config_ner, test_dataset=test_loader)
    pred_ner = trainerNer.predict()
    # print("haha")
    print(pred_ner)
    text = None
    for data_item in test_loader:
        text = data_item['text']
    token_types, entities = get_entities(pred_ner, text)
    print(token_types)
    print(entities)

    rel_list = []
    with open('./rel_predict.json', 'w', encoding='utf-8') as f:
        for i in range(len(pred_ner)):
            texti = text[i]
            for j in range(len(entities[i])):
                for k in range(len(entities[i])):
                    if j == k:
                        continue
                    rel_list.append({"text": texti, "spo_list": {"subject": entities[i][j], "object": entities[i][k]}})
        json.dump(rel_list, f, ensure_ascii=False)
    print(rel_list)


    rel_list = set_data(rel_list)
    print(rel_list)
    PATH_REL = 'rel.pth'

    config_rel = ConfigRel()
    config_rel.batch_size = 1
    rel_model = BertForSequenceClassification.from_pretrained('biobert-base-cased-v1.2',
                                                              num_labels=config_rel.num_relations)
    # rel_model = AttBiLSTM(config_rel)
    rel_model_dict = torch.load(PATH_REL, map_location=torch.device('cpu'))
    rel_model.load_state_dict(rel_model_dict['state_dict'])
    rel_test_path = './rel_predict.json'

    rel_data_process = DataPreparationRel(config_rel)
    # test_loader = rel_data_process.get_train_dev_data(path_test=rel_test_path,is_test=True)
    test_loader = rel_data_process.get_train_dev_data(data=rel_list)
    #
    trainREL = trainer_rel.Trainer(rel_model, config_rel, test_dataset=test_loader)
    rel_pred = trainREL.bert_predict()
    res_data = merge_spo(rel_list, rel_pred)
    print(res_data)
    #


if __name__ == '__main__':
    test()

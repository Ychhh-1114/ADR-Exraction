import torch.nn as nn
from transformers import BertModel
from config import *
import torch
import torch.nn.functional as F

# 忽略 transformers 警告
from transformers import logging
logging.set_verbosity_error()


class CasRel(nn.Module):

    #初始化model
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        # 冻结Bert参数，只训练下游模型
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        #定义网络
        self.sub_head_linear = nn.Linear(BERT_DIM, 1)   #sub只需要一维
        self.sub_tail_linear = nn.Linear(BERT_DIM, 1)

        self.obj_head_linear = nn.Linear(BERT_DIM, REL_SIZE)  #预测的obj矩阵需要REL_SIZE维
        self.obj_tail_linear = nn.Linear(BERT_DIM, REL_SIZE)


    #subject头尾标记预测
    def get_encoded_text(self, input_ids, mask):
        return self.bert(input_ids, attention_mask=mask)[0]

    def get_subs(self, encoded_text):
        pred_sub_head = torch.sigmoid(self.sub_head_linear(encoded_text))
        pred_sub_tail = torch.sigmoid(self.sub_tail_linear(encoded_text))
        return pred_sub_head, pred_sub_tail

    def get_objs_for_specific_sub(self, encoded_text, sub_head_seq, sub_tail_seq):
        # sub_head_seq.shape (b, c) -> (b, 1, c)
        sub_head_seq = sub_head_seq.unsqueeze(1).float()
        sub_tail_seq = sub_tail_seq.unsqueeze(1).float()

        # encoded_text.shape (b, c, 768)
        sub_head = torch.matmul(sub_head_seq, encoded_text)   #获得head和tail的编码并加在encoded_text中
        sub_tail = torch.matmul(sub_tail_seq, encoded_text)

        encoded_text = encoded_text + (sub_head + sub_tail) / 2

        # encoded_text.shape (b, c, 768)
        pred_obj_head = torch.sigmoid(self.obj_head_linear(encoded_text))
        pred_obj_tail = torch.sigmoid(self.obj_tail_linear(encoded_text))

        # shape (b, c, REL_SIZE)
        return pred_obj_head, pred_obj_tail

    def forward(self, input, mask):
        input_ids, sub_head_seq, sub_tail_seq = input
        encoded_text = self.get_encoded_text(input_ids, mask)
        pred_sub_head, pred_sub_tail = self.get_subs(encoded_text)
        # print(pred_sub_head.shape)  # (b, c, 1)
        # print(pred_sub_tail.shape)
        # exit()
        input_ids, sub_head_seq, sub_tail_seq = input
        encoded_text = self.get_encoded_text(input_ids, mask)

        # 预测subject首尾序列
        pred_sub_head, pred_sub_tail = self.get_subs(encoded_text)

        # 预测relation-object矩阵
        pred_obj_head, pred_obj_tail = self.get_objs_for_specific_sub(encoded_text, sub_head_seq, sub_tail_seq)

        return encoded_text, (pred_sub_head, pred_sub_tail, pred_obj_head, pred_obj_tail)

    def loss_fn(self, true_y, pred_y, mask):
        def calc_loss(pred, true, mask):
            true = true.float()
            

            # pred.shape (b, c, 1) -> (b, c)
            pred = pred.squeeze(-1)
            weight = torch.where(true > 0, CLS_WEIGHT_COEF[1], CLS_WEIGHT_COEF[0])

            # print(pred.shape)
            # print(true.shape)
            # exit()

            loss = F.binary_cross_entropy(pred, true, weight=weight, reduction='none')

            if loss.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            return torch.sum(loss * mask) / torch.sum(mask)  #通过与mask相乘将pad补充的元素损失进行归0

        pred_sub_head, pred_sub_tail, pred_obj_head, pred_obj_tail = pred_y
        true_sub_head, true_sub_tail, true_obj_head, true_obj_tail = true_y
        return calc_loss(pred_sub_head, true_sub_head, mask) * SUB_WEIGHT_COEF + \
               calc_loss(pred_sub_tail, true_sub_tail, mask) * SUB_WEIGHT_COEF + \
               calc_loss(pred_obj_head, true_obj_head, mask) + \
               calc_loss(pred_obj_tail, true_obj_tail, mask)












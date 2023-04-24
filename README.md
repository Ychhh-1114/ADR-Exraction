# ADR-Exraction
BERT-CasRel | Roberta-GPlinker | BERT-BILSTM-CRF

## 模型

- [ ] joint模型：
  1. BioERT-CasRel
  2. RoBERTa-GPlinker
- [ ] pipeline模型：
  1. BERT-BiLSTM-CRF
- [ ] BERT模型在纯关系分类问题中的检验

## 

## 数据集

```
text	drug	effect	indexes
We present two children with acute lymphocytic leukemia who developed leukoencephalopathy following administration of a combination of intravenous ara = C and methotrexate during the consolidation phase of chemotherapy.	ara = C	leukoencephalopathy	{'drug': {'start_char': [147], 'end_char': [154]}, 'effect': {'start_char': [70], 'end_char': [89]}}

```

数据集链接: https://pan.baidu.com/s/1s-70ZJxVmP9Rg5fnV5VWDQ?pwd=bert 提取码: bert 



## 效果分析

- CasRel在准确率和召回率以及F1值综合效果是最好的
- GPlinker可能是调参问题，准确率和召回率一般且模型参数量较大，模型大小也为这三者之中最大
- BERT-BILSTM-CRF准确率和召回率较高和CasRel差不多
- 单独检测了BERT模型对于adr关系分类问题的准确率，其效果高达95%以上，侧面说明了若在pipeline模型中发生误差，主要是由于NER模型的效果有误差

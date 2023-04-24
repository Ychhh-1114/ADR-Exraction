import torch


REL_SIZE = 2

REL_PATH = "./data/input/rel.csv"
TRAIN_PATH = './data/input/adr-train.csv'
TEST_PATH = './data/input/adr-test.csv'

BERT_MODEL_NAME = './bert-model/biobert-base-cased-v1.2'

MODEL_DIR = './data/output/'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 3  #作为demo batch设为2，在训练时调味100
BERT_DIM = 768  #BERT的输出维数
LR = 1e-3       #学习率
EPOCH = 50

# sub和obj的head与tail的判断阈值
SUB_HEAD_BAR = 0.6
SUB_TAIL_BAR = 0.6

OBJ_HEAD_BAR = 0.6
OBJ_TAIL_BAR = 0.6


#降权
CLS_WEIGHT_COEF = [0.3, 1.0]
SUB_WEIGHT_COEF = 3


EPS = 1e-10
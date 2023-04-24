import torch


REL_SIZE = 2

REL_PATH = "./data/output/rel.csv"
TRAIN_PATH = './data/input/adr-train.csv'
TEST_PATH = './data/input/adr-test.csv'

BERT_MODEL_NAME = './bert-model/biobert-base-cased-v1.2'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 2
BERT_DIM = 768 #BERT的输出维数
LR = 1e-3      #学习率
EPOCH = 50
MODEL_DIR = './data/output/'

SUB_HEAD_BAR = 0.5
SUB_TAIL_BAR = 0.5

OBJ_HEAD_BAR = 0.5
OBJ_TAIL_BAR = 0.5


#降权
CLS_WEIGHT_COEF = [0.3, 1.0]
SUB_WEIGHT_COEF = 3


EPS = 1e-10
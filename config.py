import os
import torch

#data_dir = os.getcwd() + '/data/clue/'
#data_dir = os.getcwd() + '/data/resume/'
data_dir = os.getcwd() + '\\data\\newdz_plus\\'
#data_dir = os.getcwd() + '\\data\\newdz\\'
label_dir=data_dir+'label.npz'
label_plus_dir = data_dir+'label_plus.npz'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
dev_dir = data_dir + 'dev.npz'
unlabel_dir = data_dir + 'unlabel.npz'
label_dir = data_dir + 'label.npz'
# files = ['train', 'test','unlabel']
files=['label']
bert_model = 'pretrained_bert_models\\bert-base-chinese\\'
roberta_model = 'pretrained_bert_models\\chinese_roberta_wwm_large_ext\\'
#model_dir = os.getcwd() + '/experiments/resume/'
model_dir = os.getcwd() + '\\experiments\\newdz_plus\\'
log_dir = model_dir + 'train.log'
case_dir = os.getcwd() + '\\case\\newdz_plus\\bad_case.txt'
test_result_dir = os.getcwd() + '\\experiments\\newdz_plus\\result.txt'
# 训练集、验证集划分比例
dev_split_size = 0.1
test_split_size=0.5

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 2e-5  #3e-5
weight_decay = 0.01
clip_grad = 5   #梯度裁剪

batch_size = 4
epoch_num = 50
min_epoch_num = 5
patience = 0.00002
patience_num = 20

best_val_f1=0.0
best_val_loss=1e18
test_batch_size=1

gpu = '0'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")

'''labels = ['address', 'book', 'company', 'game', 'government',
          'movie', 'name', 'organization', 'position', 'scene'] 
    '''#clue
'''labels = ['CONT', 'EDU', 'LOC', 'NAME', 'ORG',
          'PRO', 'RACE', 'TITLE']'''#resume
labels = ['灾害地点', '灾害类型', '伤亡情况','灾害规模','经济损失','致灾原因']
label2id = {
    "O": 0,
    "B-灾害地点": 1,
    "B-灾害类型": 2,
    "B-伤亡情况": 3,
    "B-灾害规模": 4,
    "B-经济损失": 5,
    "B-致灾原因": 6,
    "M-灾害地点": 7,
    "M-灾害类型": 8,
    "M-伤亡情况": 9,
    "M-灾害规模": 10,
    "M-经济损失": 11,
    "M-致灾原因": 12,
    'E-灾害地点': 13,
    "E-灾害类型": 14,
    'E-伤亡情况': 15,
    "E-灾害规模": 16,
    "E-经济损失": 17,
    "E-致灾原因": 18
}
'''label2id = {
    "O": 0,
    "B-CONT": 1,
    "B-EDU": 2,
    "B-LOC": 3,
    'B-NAME': 4,
    'B-ORG': 5,
    'B-PRO': 6,
    'B-RACE': 7,
    'B-TITLE': 8,
    'M-CONT': 9,
    'M-EDU': 10,
    "M-LOC": 11,
    "M-NAME": 12,
    "M-ORG": 13,
    'M-PRO': 14,
    'M-RACE': 15,
    'M-TITLE': 16,
    'E-CONT': 17,
    'E-EDU': 18,
    'E-LOC': 19,
    'E-NAME': 20,
    'E-ORG':21,
    "E-PRO": 22,
    "E-RACE": 23,
    "E-TITLE": 24,
    'S-CONT': 25,
    'S-EDU': 26,
    'S-LOC': 27,
    'S-NAME': 28,
    'S-ORG': 29,
    'S-PRO': 30,
    'S-RACE': 31,
    'S-TITLE': 32
}''' #resume 
'''label2id = {
    "O": 0,
    "B-address": 1,
    "B-book": 2,
    "B-company": 3,
    'B-game': 4,
    'B-government': 5,
    'B-movie': 6,
    'B-name': 7,
    'B-organization': 8,
    'B-position': 9,
    'B-scene': 10,
    "I-address": 11,
    "I-book": 12,
    "I-company": 13,
    'I-game': 14,
    'I-government': 15,
    'I-movie': 16,
    'I-name': 17,
    'I-organization': 18,
    'I-position': 19,
    'I-scene': 20,
    "S-address": 21,
    "S-book": 22,
    "S-company": 23,
    'S-game': 24,
    'S-government': 25,
    'S-movie': 26,
    'S-name': 27,
    'S-organization': 28,
    'S-position': 29,
    'S-scene': 30
}
''' #clue

id2label = {_id: _label for _label, _id in list(label2id.items())}

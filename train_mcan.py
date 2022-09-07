import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import numpy as np
from tqdm import tqdm
import torch
from model_mcan import Net
from sklearn.model_selection import train_test_split
import csv
import base64
from nltk.tokenize import wordpunct_tokenize
import glob
import random

torch.random.manual_seed(42)
np.random.seed(42)
random.seed(42)


csv.field_size_limit(1000000)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
device = 'cuda:0'

labels = json.load(open('label.json', 'r', encoding='utf-8'))
label2id = {key: i for i, (key, value) in enumerate(labels.items())}

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)


vocab2id = {vocab: i for i, vocab in enumerate(open('vocabs.txt', encoding='utf-8').read().splitlines())}
def tokenize(text):
    text = wordpunct_tokenize(text.lower())
    return [vocab2id[word] if word in vocab2id else 0 for word in text]

with open(r'D:\open_data\VQA\CLEVR_v1.0\questions\CLEVR_val_questions.json', encoding='utf-8') as f:
    CLVER = json.load(f)
    np.random.shuffle(CLVER['questions'])

train_clver, dev_clver = train_test_split(CLVER['questions'], test_size=0.1, random_state=42)
image_features = np.zeros((15000, 196, 1024), dtype='float32')

# with open('vec.tsv', "r") as tsv_in_file:
#     reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
#     for item in tqdm(reader):
#         # break
#         item['image_id'] = int(item['image_id'])
#         item['image_h'] = int(item['image_h'])
#         item['image_w'] = int(item['image_w'])
#         item['num_boxes'] = int(item['num_boxes'])
#         for field in ['features']:
#             data = item[field]
#             # buf = base64.decodestring(data)
#             buf = base64.b64decode(data[1:])
#             temp = np.frombuffer(buf, dtype=np.float32)
#             item[field] = temp.reshape((item['num_boxes'],-1))
#         image_features[item['image_id'], :item['features'].shape[0], :] = item['features']
#
# with open('Faster-R-CNN-with-model-pretrained-on-Visual-Genome-master/path2id.json', 'r', encoding='utf-8') as f:
#     path2id = json.load(f)


feature_files = glob.glob(r'D:\PekingInfoOtherSearch\openvqa-master\data\clevr\feats\val\*.npz')
for file in tqdm(feature_files):
    # break
    file = file.replace('\\', '/')
    iid = file.split('/')[-1].split('.')[0]
    feature = np.load(file)['x']
    image_features[int(iid)] = feature

def dataloader(clevrset, batch_size):
    batch_img_feature, batch_question, batch_ques_ans, labels = [], [], [], []
    for data in clevrset:
        path = data['image_filename']
        question = data['question']
        ques_ans = data['question'] + ' ' + data['answer']
        batch_img_feature.append(int(os.path.splitext(path)[0].split('_')[-1]))
        # batch_img_feature.append(image_features[path2id[path]])
        batch_question.append(tokenize(question))
        batch_ques_ans.append(tokenize(ques_ans))
        labels.append(label2id[data['answer']])
        if len(batch_img_feature) == batch_size:
            batch_img_feature = image_features[batch_img_feature]
            # batch_img_feature = np.array(batch_img_feature)
            batch_question = sequence_padding(batch_question)
            batch_ques_ans = sequence_padding(batch_ques_ans)
            labels = np.array(labels)
            yield batch_img_feature, batch_question, batch_ques_ans, labels
            batch_img_feature, batch_question, batch_ques_ans, labels = [], [], [], []


class Params:
    HIDDEN_SIZE = 512
    DROPOUT_R = 0.1
    MULTI_HEAD = 8
    HIDDEN_SIZE_HEAD = 64
    FF_SIZE = 2048
    LAYER = 6
    FLAT_MLP_SIZE = 512
    FLAT_GLIMPSES = 1
    FLAT_OUT_SIZE = 1024
    WORD_EMBED_SIZE = 300
    IMG_FEAT_SIZE = 1024
    USE_GLOVE = True
    OPT_BETAS = (0.9, 0.98)
    OPT_EPS = 1e-9
    BATCH_SIZE = 64
    LR_BASE = 0.00004

class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size


    def step(self):
        self._step += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * 1):
            r = self.lr_base * 1/4.
        elif step <= int(self.data_size / self.batch_size * 2):
            r = self.lr_base * 2/4.
        elif step <= int(self.data_size / self.batch_size * 3):
            r = self.lr_base * 3/4.
        else:
            r = self.lr_base

        return r


def get_optim(__C, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = __C.LR_BASE

    return WarmupOptimizer(
        lr_base,
        torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0,
            betas=__C.OPT_BETAS,
            eps=__C.OPT_EPS
        ),
        data_size,
        __C.BATCH_SIZE
    )

pretrain_emd = np.load('embedding.npz', allow_pickle=True)['embedding']
model = Net(Params(), pretrain_emd, token_size=96, answer_size=28)
params = model.parameters()
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = get_optim(Params(), model, len(train_clver))
# optimizer = torch.optim.Adam(lr=Params().LR_BASE, params=params)
model.to(device)
mselosses = []

for _ in range(20):
    model.train()
    pbar = tqdm(enumerate(dataloader(train_clver, Params().BATCH_SIZE)))
    # model.load_state_dict(torch.load('model.pt'))
    accs = []
    losses = []
    for idx, (image_feature, ques_feature, ques_ans_feature, labels) in pbar:
        optimizer.zero_grad()
        image_feature = torch.tensor(image_feature).float().to(device)
        ques_feature = torch.tensor(ques_feature).long().to(device)
        ques_ans_feature = torch.tensor(ques_ans_feature).long().to(device)
        labels = torch.tensor(labels).long().to(device)
        pred = model(image_feature, ques_feature)
        loss = ce_loss(pred, labels)
        losses.append(loss.item())
        pred = torch.argmax(pred, dim=-1)
        acc = torch.sum((labels == pred)) / len(labels)
        accs.append(acc.item())
        pbar.set_description(f"train set: loss {np.mean(losses)}, acc {np.mean(accs)}, lr {optimizer._rate}")
        loss = loss
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'model.pt')

    model.eval()
    pbar = tqdm(enumerate(dataloader(dev_clver, Params().BATCH_SIZE)))
    accs = []
    losses = []
    for idx, (image_feature, ques_feature, ques_ans_feature, labels) in pbar:
        image_feature = torch.tensor(image_feature).float().cuda()
        ques_feature = torch.tensor(ques_feature).long().cuda()
        ques_ans_feature = torch.tensor(ques_ans_feature).long().cuda()
        labels = torch.tensor(labels).long().cuda()
        pred = model(image_feature, ques_feature)
        loss = ce_loss(pred, labels)
        losses.append(loss.item())
        pred = torch.argmax(pred, dim=-1)
        acc = torch.sum((labels == pred)) / len(labels)
        accs.append(acc.item())
        pbar.set_description(f"dev set: loss {np.mean(losses)}, acc {np.mean(accs)}, lr {optimizer._rate}")
        loss = loss

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

import time
import datetime

from torch.utils.data import Dataset, TensorDataset, DataLoader

from cantokenizer import CanTokenizer

import random

'''
PYTHON=/opt/conda/bin/python3
TOTAL_NUM_UPDATES=2
WARMUP_UPDATES=0.1 
UPDATE_FREQ=1 
BATCH_SIZE=2 # per gpu 
SEQ_LENGTH=96 
LR=1e-4 
MODEL=electra_nli4_orig
MODEL_SIZE=base 
VOCAB_FILE=cantokenizer-vocab.txt 
DATA_DIR=./ 
SAVE_DIR=$MODEL"_"$MODEL_SIZE"_"$BATCH_SIZE"_"$SEQ_LENGTH
TOKENIZERS_PARALLELISM=false $PYTHON train.py \
  $DATA_DIR \
  --eval-dir $DATA_DIR \
  --model $MODEL \
  --restore-file pt_model_d.pt \
  --save-dir "$SAVE_DIR" \
  --log-interval 5  --weight-decay 0.01 \
  --seq-length $SEQ_LENGTH \
  --lr $LR \
  --gpus 8 \
  --warmup-updates $WARMUP_UPDATES \
  --total-num-update $TOTAL_NUM_UPDATES \
  --batch-size $BATCH_SIZE \
  --num-workers=0 \
  --model-size=$MODEL_SIZE \
  --vocab-file=$VOCAB_FILE \
  --shuffle



'''

def convert(data, tokenizer, max_seq_length, prefix):
    '''
        data = [
            ["First sentence", "Second Sentence", 0],
            ...
        ]
    '''
    encodeds = tokenizer.encode_batch([(e[0], e[1]) for e in data])
    data_original = []
    data_attn_mask = []
    data_labels = []
    data_type_ids = []
    for e, t in zip(encodeds, data):
        label = t[2]
        if len(e.ids) > max_seq_length:
            continue
        e.pad(max_seq_length)
        data_original.append(e.ids)
        data_attn_mask.append(e.attention_mask)
        data_labels.append(label)
        data_type_ids.append(e.type_ids)

    indices = list(range(len(data_original)))
    random.shuffle(indices)

    data_original = [data_original[i] for i in indices]
    data_attn_mask = [data_attn_mask[i] for i in indices]
    data_labels = [data_labels[i] for i in indices]
    data_type_ids = [data_type_ids[i] for i in indices]

    ids = np.array(data_original).astype(np.int16)
    attn = np.array(data_attn_mask).astype(np.int8)
    labels = np.array(data_labels).astype(np.int8)
    type_ids = np.array(data_type_ids).astype(np.int8)
    
    with open(prefix+"_ids", 'wb') as f:
        f.write(ids.tobytes())
    with open(prefix+"_mask", 'wb') as f:
        f.write(attn.tobytes())
    with open(prefix+"_type_ids", 'wb') as f:
        f.write(type_ids.tobytes())
    with open(prefix+"_label", 'wb') as f:
        f.write(labels.tobytes())

'''

import json

from cantokenizer import CanTokenizer
tokenizer = CanTokenizer('cantokenizer-vocab.txt', 
                         add_special=True, 
                         add_special_cls='<s>', 
                         add_special_sep='</s>'
                        )

max_seq_length = 96

with open('yuenli-train.json') as f:
    train = json.load(f)
convert([e for e in train if e[0] and e[1]], tokenizer, 96, 'train')

with open('yuenli-test.json') as f:
    test = json.load(f)
convert([e for e in test if e[0] and e[1]], tokenizer, 96, 'test')

'''



#####################################
## 
##            Data Utils
## 
#####################################

import numpy as np
import os
class textDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 seq_length,
                 batch_size, 
                 eval=False, 
                 eval_num_samples=0):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_path = data_path
        
        self.eval_num_samples = (eval_num_samples // batch_size)
        self.eval = eval

        prefix = 'test' if self.eval else 'train'
        self.length = os.stat(data_path+prefix+"_ids").st_size//(seq_length*2) // batch_size

        self.ids_bin_buffer = None
        self.dataset = None

    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        if i >= self.length:
            raise StopIteration
        return self.__getbatch__(i*self.batch_size, self.batch_size)

    def __getbatch__(self, i, size):
        seq_length = self.seq_length
        if self.ids_bin_buffer is None:
            data_path = self.data_path
            prefix = 'test' if self.eval else 'train'

            path = data_path+prefix+"_ids" 
            self.ids_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.ids_bin_buffer = self.ids_bin_buffer_mmap

            path = data_path+prefix+"_mask"
            self.attention_mask_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.attention_mask_bin_buffer = self.attention_mask_bin_buffer_mmap 

            path = data_path+prefix+"_label"
            self.labels_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.labels_bin_buffer = self.labels_bin_buffer_mmap

            path = data_path+prefix+"_type_ids"
            self.type_ids_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.type_ids_bin_buffer = self.type_ids_bin_buffer_mmap 
        
        start = seq_length*i*2
        shape = (size,self.seq_length)
        
        ids_buffer = np.frombuffer(self.ids_bin_buffer, dtype=np.int16, count=seq_length*size, offset=start).reshape(shape)
        labels_buffer = np.frombuffer(self.labels_bin_buffer, dtype=np.int8, count=size, offset=i).reshape((size))
        attention_mask_buffer = np.frombuffer(self.attention_mask_bin_buffer, dtype=np.int8, count=seq_length*size, offset=start // 2).reshape(shape)
        type_ids_buffer = np.frombuffer(self.type_ids_bin_buffer, dtype=np.int8, count=seq_length*size, offset=start // 2).reshape(shape)
        return (
            torch.LongTensor(ids_buffer),
            torch.LongTensor(attention_mask_buffer), 
            torch.LongTensor(labels_buffer), 
            torch.LongTensor(type_ids_buffer), 
        )

#####################################
## 
##             Modelling
## 
#####################################



from transformers import (
    ElectraForSequenceClassification as ElectraForSequenceClassification_, 
    ElectraModel,
    ElectraPreTrainedModel
)
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn

def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))



class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ElectraForSequenceClassification(ElectraForSequenceClassification_):
    def __init__(self, config):
        ElectraPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels
        self.electra = ElectraModel(config)
        self.classifier = ElectraClassificationHead(config)

        self.init_weights()
        
        


from transformers import ElectraConfig

def get_model(args, tokenizer):
    config = ElectraConfig.from_pretrained('google/electra-base-discriminator')
    config.num_labels = 4
    config.vocab_size = tokenizer.get_vocab_size()
    model = ElectraForSequenceClassification(config)
    return model

def get_loss(model, sample, args, device, gpus=0, report=False):

    ids, mask, labels, type_ids = sample

    if gpus:
        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        type_ids = type_ids.to(device)

    loss, pred = model(ids, mask, type_ids, labels=labels)

    log = None
    if report:
        log = OrderedDict()
        log['acc'] = (pred.argmax(-1) == labels).sum()
        log['acc_tot'] = torch.LongTensor([labels.shape[0]]).sum().cuda()

    return loss, log


def evaluate(model, sample, args, device, record, gpus=0, report=False):
    ids, mask, labels, type_ids = sample

    if gpus:
        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        type_ids = type_ids.to(device)
        
    pred = model(ids, mask, type_ids, labels=labels)[1]

    if 'correct_tot' not in record:
        record['correct_tot'] = 0
    if 'correct' not in record:
        record['correct'] = 0
    
    record['correct'] += (pred.argmax(-1) == labels).sum()
    record['correct_tot'] += torch.LongTensor([labels.shape[0]]).sum().cuda()

def post_evaluate(record, args):
    record['accuracy'] = float(record['correct']) / float(record['correct_tot'])
    record['correct'] = record['correct']
    record['correct_tot'] = record['correct_tot']

def log_formatter(log, tb, step_i):
    log['acc'] = float(log['acc'] / log['acc_tot'])

def get_tokenizer(args):
    return None
    return CanTokenizer(vocab_file = args.vocab_file)

def set_parser(parser):
    parser.add_argument('--base-model', help='model file to import')
    parser.add_argument('--model-size', default='small',
                        help='model size, '
                                'e.g., "small", "base", "large" (default: small)')

def get_dataset(args):
    return textDataset(
        args.data, 
        args.seq_length, 
        args.batch_size,
        eval_num_samples=0,
    )

def get_eval_dataset(args):
    return textDataset(
        args.data, 
        args.seq_length, 
        args.batch_size,
        eval_num_samples=0,
        eval=True
    )
    pass

get_model
get_loss
log_formatter 
get_tokenizer
evaluate
post_evaluate
set_parser
get_dataset

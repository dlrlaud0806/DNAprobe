import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F 
import random
import json
import os, sys

#### file paths
dirs = {
    #'raw' : os.path.join('..', '201711'),
    'raw' : os.path.join('..', '202207'),
    'processed': os.path.join('..', 'processed_202207'),
    'folds': os.path.join('..', 'processed_202207', 'folds'),
    'LOO': os.path.join('..', 'processed_202207', 'LOO'),
    'LOCO': os.path.join('..', 'processed_202207', 'LOCO'),
    'logs': os.path.join('..', 'logs')
}

for d in dirs.values():
    if not os.path.exists(d):
        os.makedirs(d)

def get_graphs_filepath(dataset_flag):
    return os.path.join(dirs['processed'], '%s_graphs.json' % dataset_flag)

def get_stats_filepath(dataset_flag):
    return os.path.join(dirs['processed'], '%s_feature_stats.json' % dataset_flag)
    
def get_fold_filepath(dataset_flag, n_folds, fold_id):
    return os.path.join(dirs['folds'], '%s_%i_%i.json' % (dataset_flag, n_folds, fold_id))

def get_LOO_filepath(dataset_flag, id):
    return os.path.join(dirs['LOO'], '%s_%s.json' % (dataset_flag, id))

def get_LOCO_filepath(id):
    return os.path.join(dirs['LOCO'], 'class_%s.json' % (id))
####


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def manage_model_restoration(checkpoint_file_name, model):
    if checkpoint_file_name is not None:
        checkpoint = torch.load(checkpoint_file_name)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, (complex, np.complex)):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  # pragma: py3
            return obj.decode()
        return json.JSONEncoder.default(self, obj)

def dump_log(log_to_save, hypers, file_name):
    if file_name is not None:
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))
        with open(file_name, 'w') as f:
            json.dump(log_to_save, f, indent=4, cls=NumpyEncoder)
        hypers_file = '%s.hypers' % os.path.splitext(file_name)[0]
        with open(hypers_file, 'w') as f:
            json.dump(hypers, f, indent=4)
        print('log dumped to %s' % file_name)


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hid_sizes):
        super().__init__() # 모델 연산 정의
        self.fc1 = nn.Linear(in_size, hid_sizes, bias=True)
        self.fc2 = nn.Linear(hid_sizes, out_size, bias=True)
        # self.dropout = nn.Dropout(0)

    def forward(self, x): # 모델 연산의 순서를 정의
        x = self.fc1(x) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        x = F.relu(self.fc2(x)) # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        return x

def print_temporary(s):
    print('%s     \r' % s, end='')
    sys.stdout.flush()
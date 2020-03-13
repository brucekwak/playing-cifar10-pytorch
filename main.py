# [Reference] https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

import os
import sys
# import logging
from pprint import pprint

from easydict import EasyDict
import numpy as np
import random

import torch

import argparse
import json

from data_loader import get_train_valid_loader_CIFAR10_cutout, get_train_valid_loader_CIFAR10_cutmix, get_test_loader_CIFAR10
from model import SimpleConvNet
from train import train_and_test

#############################################
# 1. 실험을 위한 파라미터 셋팅
#############################################
print('==> Setting parameters..')
parser = argparse.ArgumentParser()
parser.add_argument('--params', type=str, help='Parameter Json file')
args = parser.parse_args()

param_dir = './parameters/' + args.params
f = open(param_dir)
params = json.load(f)
pprint(params)

args_train = EasyDict(params['ARGS_TRAIN'])
RUN_CODE = params['RUN_CODE']
DATA_PATH = params['DATA_PATH']
CUTMIX = params['CUTMIX']


#############################################
# 2. 데이터셋 준비
#############################################
print('==> Preparing data..')
if args_train.data == "CIFAR10":
    if CUTMIX == True:
        train_loader, valid_loader = get_train_valid_loader_CIFAR10_cutmix(
                                                                        DATA_PATH, 
                                                                        batch_size = args_train.batch_size,
                                                                        valid_size = 0.2,
                                                                        num_workers = args_train.workers)
    else:
        train_loader, valid_loader = get_train_valid_loader_CIFAR10_cutmix(
                                                                        DATA_PATH, 
                                                                        batch_size = args_train.batch_size,
                                                                        valid_size = 0.2,
                                                                        num_workers = args_train.workers)

    test_loader = get_test_loader_CIFAR10(
                                        DATA_PATH, 
                                        batch_size = args_train.batch_size,
                                        num_workers = args_train.workers)


#############################################
# 3. 모델 선언
#############################################
print('==> Building model..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SimpleConvNet(args_train.input_dim, args_train.num_classes)
model.to(device)


#############################################
# 4. 학습
#############################################
print('==> Training..')
best_val_prec1, test_prec1, params = train_and_test(model, train_loader, valid_loader, test_loader, args_train, CUTMIX, RUN_CODE)

#############################################
# 5. 결과 출력
#############################################
print('############################')
print('best_val_prec1:', best_val_prec1)
print('test_prec1:', test_prec1)
print('params:', params)
print('############################')
print('Finished!')

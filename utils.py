import pickle,os
from PIL import Image
import scipy.io
import time
from tqdm import tqdm
import pandas as pd
import shutil
from random import randint
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# Description: < This function calculates accuracy on given data. >
# Input: < Takes output and target variable. >
# Output: < Returns accuracy and precision. >
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # print("output target topk in accuracy: ", output, target, topk)
    maxk = max(topk)
    batch_size = target.size(0)
    # print("max in accuracy: ", maxk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print("pred in accuracy: ", pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print("correct in accuracy: ", correct)
    res = []
    for k in topk:
        # print("correct[:k]", correct[:k])
        correct_k = correct[:k].view(-1).float().sum(0)
        # print("currect_k and k in accuracy: ", correct_k, k)
        res.append(correct_k.mul_(100.0 / batch_size))
    # print('res is ', res)
    return res
# Descripton: < This function resets and updates values as per need. >
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    #updates value as per given
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# Descripton: < This function saves the best model >
def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best) #saving checkpoint by copying the best model

# Descripton: < This function takes info and prints it. >

def record_info(info,filename,mode):

    if mode =='train':

        result = (
              'Time {batch_time} '
              'Data {data_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5}\n'
              'LR {lr}\n'.format(batch_time=info['Batch Time'],
               data_time=info['Data Time'], loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5'],lr=info['lr']))
        print result

        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Data Time','Loss','Prec@1','Prec@5','lr']

    if mode =='test':
        result = (
              'Time {batch_time} \n'
              'Loss {loss} '
              'Prec@1 {top1} '
              'Prec@5 {top5} \n'.format( batch_time=info['Batch Time'],
               loss=info['Loss'], top1=info['Prec@1'], top5=info['Prec@5']))
        print result
        df = pd.DataFrame.from_dict(info)
        column_names = ['Epoch','Batch Time','Loss','Prec@1','Prec@5']

    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)

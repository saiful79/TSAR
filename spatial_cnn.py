# """
# Author --> Semanticslab
# Writter --> Hasan Ali Emon, Saiful Islam
# Target --> Recognizing action from video
#
#
# """

import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader
from utils import *
from network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0" #it will detect cuda autometically

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=5000, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
# add finetune and last layer flag argument
#when use --fine-tune-flag default true then it pretrain+findtuning accuracy find
#by default fine tune is true, means by default it will run. if we want to run last layer then
#we have to make fine tune false and last layer true
parser.add_argument('--fine-tune-flag', default = True, type = bool, help = 'set fine tune flag. default is True')
#when use --fine-tune-flag default False and --last-layer-flag default true then it pretrain+last layer accuracy find
parser.add_argument('--last-layer-flag', default = False, type = bool, help = 'set last layer flag. default is False')

def main():
    global arg
    arg = parser.parse_args()
    print arg


    #Preparing DataLoader
    data_loader = dataloader.spatial_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=12,

                        # path='/home/ubuntu/data/UCF101/spatial_no_sampled/'
                        path='/media/semanticslab11/hdd1/data/two-stream-action/data/backup/ucf101/', #path to your ucf 101 data
                        # ucf_list ='/home/ubuntu/lab/pytorch/ucf101_two_stream/github/UCF_list/',
                        ucf_list ='UCF_list/',#here use ucf_list path
                        ucf_split ='01',
                        )

    train_loader, test_loader, test_video = data_loader.run()
    #initiating our model
    model = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test_video=test_video,
                        pretrain_finetune = arg.fine_tune_flag,
                        pretrain_last_layer = arg.last_layer_flag
    )
    #Calling run function to run the model
    model.run()

class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, test_video, pretrain_finetune, pretrain_last_layer):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video
        self.pretrain_finetune = pretrain_finetune
        self.pretrain_last_layer = pretrain_last_layer

# Description: < This function takes a pretrained model and mainpulates its last layer according
#     to our action classes.>
# Output: < returns a nural network model>

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        # self.model = resnet101(pretrained= True, channel=3)#.cuda()

        self.model = models.resnet101(pretrained= True).cuda() #use pretraing model resnet101 for feature extraction
        # pretrain + fine tune
        if self.pretrain_finetune:
            print('pretrain + fine tuning...')
            model_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(model_ftrs, 101).cuda()
        elif self.pretrain_last_layer:
            print('pretrain + last layer...')
            # pretrain + last layear
            # self.model = models.resnet101(pretrained= True)
            # freeze all except last layer
            for param in self.model.parameters():
                param.requires_grad = False
            model_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(model_ftrs, 101).cuda()

        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)

# Description: < This function resumes from the last saved checkpoint.>

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return
# Description: < This function runs some function for train and validate datasets, saves the
#     best prediction in a 'pickle' file, checkpoint and model in a '**.pth.tar' file >


    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True

        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1 #if prec1 is best then is best is True
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best: #if is_best is True then we save prediction in a pickle file
                self.best_prec1 = prec1
                with open('record/spatial/spatial_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()

            save_checkpoint({ #and also checkpoint in a '**.pth.tar' file
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/spatial/checkpoint.pth.tar','record/spatial/model_best.pth.tar')

# Description: < This function trains our model >

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter() #initializing everything to zero
        data_time = AverageMeter() #initializing everything to zero
        losses = AverageMeter() #initializing everything to zero
        top1 = AverageMeter() #initializing everything to zero
        top5 = AverageMeter() #initializing everything to zero
        self.model.train() #switch to train mode
        end = time.time() #taking starting time
        progress = tqdm(self.train_loader) #taking our progress bar
        for i, (data_dict,label) in enumerate(progress):   # mini-batch training
            #in every iteration its taking data whose length is equal to batch size
            #we have to access data according to magic function '__getitem__' returns data
            #that's why we take a tuple as we are getting a tuple from '__getitem__'
            # measure data loading time
            data_time.update(time.time() - end)

            label = label.cuda(async=True)
            target_var = Variable(label).cuda()

            # Initiating output, output's dimension should be equal to model's output's dimension
            output = Variable(torch.zeros(len(data_dict['img1']),101).float()).cuda() #here should be number of classes
            for i in range(len(data_dict)):
                key = 'img'+str(i)
                data = data_dict[key]
                input_var = Variable(data).cuda()
                output += self.model(input_var)

            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label, topk=(1, 5)) #topk=(1, 5)
            # losses.update(loss.data[0], data.size(0))
            # top1.update(prec1[0], data.size(0))
            # top5.update(prec5[0], data.size(0))

            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train') #saving train information

# Description: < This function validate our model >

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys,data,label) in enumerate(progress):

            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction
            preds = output.data.cpu().numpy()
            # print('preds is ', preds)
            # np.savetxt('result_of_pred.txt', preds)
            # print('len of preds ', len(preds))
            # print('type of pred ', type(preds))
            # print('preds.shape[0] ', preds.shape[0])
            # with open('result_of_pred.txt', 'w') as f:
            #     f.write(preds)
            nb_data = preds.shape[0]
            # break
            for j in range(nb_data):
                # print('keys are ', keys[j])
                videoName = keys[j].split('/',1)[0]
                # print('video names are ', videoName)
                # exit()
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j,:]
                else:
                    self.dic_video_level_preds[videoName] += preds[j,:]

        video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()



        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(video_loss,5)],
                'Prec@1':[round(video_top1,3)],
                'Prec@5':[round(video_top5,3)]}
        record_info(info, 'record/spatial/rgb_test.csv','test')
        return video_top1, video_loss

# Description: < This function calculates validation accuracy. This final value of 'correct'
#     is how many times it has correctly predicted.>

    def frame2_video_level_accuracy(self):

        correct = 0
        # video_level_preds will contain prediction of each video
        video_level_preds = np.zeros((len(self.dic_video_level_preds),101)) # use 101 instead of 2
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):

            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name])-1

            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1
            if np.argmax(preds) == (label): #if it predicts correctly then we increase 'correct' by one
                correct+=1

        #top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()

        top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5)) #calculating accuracy and precision
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda()) #use .cuda() each varibales like as Variable(video_level_preds).cuda()

        top1 = float(top1.numpy())
        top5 = float(top5.numpy())

        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1,top5,loss.data.cpu().numpy()



if __name__=='__main__':
    main()
    # model = resnet101(pretrained = False, channel=3)
    # model.load_state_dict(torch.load('record/spatial/model_best.pth.tar')['state_dict'])
    # original saved file with DataParallel
    # state_dict = torch.load('record/spatial/model_best.pth.tar')
    # print(state_dict)
    # # create new OrderedDict that does not contain `module.`
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    # model.load_state_dict(new_state_dict)

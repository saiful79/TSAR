import numpy as np
import pickle
from PIL import Image
import time
import shutil
import random
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from split_train_test_video import *

class motion_dataset(Dataset):
    def __init__(self, dic, in_channel, root_dir, mode,bidirectional, transform=None):
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.bidirectional=bidirectional
        self.in_channel = in_channel
        self.img_rows=224
        self.img_cols=224

# Description: < This function stacks 20 images; 5 horizontal and vertical for backward flow
    # and other 5 horizontal and vertical for forward flow.
    # this function also transforms a image according to "self.transform" >
# Output: < This function returns a list of 20 images. >

    def stackopf(self):
        name = 'v_'+self.video
        u = self.root_dir+ 'u/' + name
        v = self.root_dir+ 'v/'+ name

        flow = torch.FloatTensor(2*self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)

        if self.bidirectional:
            count = 19
            for j in range(int(self.in_channel/2), 0, -1):
                idx = i + j
                idx = str(idx)
                frame_idx = 'frame'+ idx.zfill(6)
                h_image = u +'/' + frame_idx +'.jpg'
                v_image = v +'/' + frame_idx +'.jpg'

                imgH=(Image.open(h_image))
                imgV=(Image.open(v_image))

                H = self.transform(imgH)
                V = self.transform(imgV)
                # flow[2*(j-1),:,:] = H
                # flow[2*(j-1)+1,:,:] = V
                flow[count -1,:,:] = H
                flow[count,:,:] = V
                count -=2

                imgH.close()
                imgV.close()

            count = 9
            for j in range(int(self.in_channel/2) + 1, self.in_channel+1):
                idx = i + j
                idx = str(idx)
                frame_idx = 'frame'+ idx.zfill(6)
                h_image = u +'/' + frame_idx +'.jpg'
                v_image = v +'/' + frame_idx +'.jpg'

                imgH=(Image.open(h_image))
                imgV=(Image.open(v_image))

                H = self.transform(imgH)
                V = self.transform(imgV)

                # flow[2*(j-1),:,:] = H
                # flow[2*(j-1)+1,:,:] = V
                flow[count - 1,:,:] = H
                flow[count,:,:] = V
                count -=2

                imgH.close()
                imgV.close()
        else:
            for j in range(int(self.in_channel/2), 0, -1):
                idx = i + j
                idx = str(idx)
                frame_idx = 'frame'+ idx.zfill(6)
                h_image = u +'/' + frame_idx +'.jpg'
                v_image = v +'/' + frame_idx +'.jpg'

                imgH=(Image.open(h_image))
                imgV=(Image.open(v_image))

                H = self.transform(imgH)
                V = self.transform(imgV)
                flow[2*(j-1),:,:] = H
                flow[2*(j-1)+1,:,:] = V


                imgH.close()
                imgV.close()


            for j in range(int(self.in_channel/2) + 1, self.in_channel+1):
                idx = i + j
                idx = str(idx)
                frame_idx = 'frame'+ idx.zfill(6)
                h_image = u +'/' + frame_idx +'.jpg'
                v_image = v +'/' + frame_idx +'.jpg'

                imgH=(Image.open(h_image))
                imgV=(Image.open(v_image))

                H = self.transform(imgH)
                V = self.transform(imgV)



                flow[2*(j-1),:,:] = H
                flow[2*(j-1)+1,:,:] = V


                imgH.close()
                imgV.close()
        return flow
    #returns length of keys
    def __len__(self):
        return len(self.keys)


# Description: < This is a magic function and to preprocess data we need to overload this function.
#     For training data we select randomly a image idx, takes 10 cosecutive frames and preprocessing it. >
# Output: < This function returns tuple of data and label if it is in training stage and if
#     it is in validation stage it returns tuple of video_name, data and label.>
# Note: < We have to access data during train or validation just the way this function returns
#     data. i.e. for training first we need to access the "data" and then "label" as it returns
#     "sample = (data, label)" and for validatoin first "video_name" then "data" and at the
#     last "label" >



    def __getitem__(self, idx):
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)

        if self.mode == 'train':
            self.video, nb_clips = self.keys[idx].split('-')
            self.clips_idx = random.randint(1,int(nb_clips))
        elif self.mode == 'val':
            self.video,self.clips_idx = self.keys[idx].split('-')
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1
        data = self.stackopf()

        if self.mode == 'train':
            sample = (data,label)
        elif self.mode == 'val':
            sample = (self.video,data,label)
        else:
            raise ValueError('There are only train and val mode')
        return sample





class Motion_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel,  path, ucf_list, ucf_split,bidirectional):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count={}
        self.in_channel = in_channel
        self.data_path=path
        self.bidirectional=bidirectional
        # split the training and testing videos
        splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
        self.train_video, self.test_video = splitter.split_video()

# """
# Description: <This function opens a pickle file from "/dataloader/dic/" directory which is a dictionary
# containing folder name as key and how many frame that folder contains as value. example, dic_frame["v_ApplyEyeMakeup_g01_c01.avi"]
# has the value 163 as that directory contains 163 image frame.And makes a new dictionay names frame_count whose key is extracted
# from the key of dic_frame and value is the same as dic_frame has. >
#
# """


    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        with open('/home/semanticslab11/development/two_stream_bidirectional_pytorch/two-stream-action-original_bidirection/dataloader/dic/frame_count_opf_101.pickle','rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame :
            videoname = line.split('_',1)[1].split('.',1)[0]
            n,g = videoname.split('_',1)
            if n == 'HandStandPushups':
                videoname = 'HandstandPushups_'+ g
            self.frame_count[videoname]=dic_frame[line]

# """
# Description: < This function runs some functions >
# Output: < Returns Train_loader, val_loader and test_video
# """

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample19()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video

# Descripton: <This function increases(19 times) the amount of test data and returns a dictionary
# whose key is the same as "dic_training" and contains value same as "test_video">

    def val_sample19(self):
        self.dic_test_idx = {}
        #print len(self.test_video)
        for video in self.test_video:
            n,g = video.split('_',1)

            sampling_interval = int((self.frame_count[video]-10)/19)
            for index in range(19):
                clip_idx = index*sampling_interval
                key = video + '-' + str(clip_idx+1)
                self.dic_test_idx[key] = self.test_video[video]

# '''
# Description: This function reads the "train_video" dictionary and makes a new dictionary whoes value
# is 10 less than train_video's value because we will select image randomly and this will make sure
# that our randomly selected frame is within the bound. And we are making new key for "dic_training"
# dictionary which is our output dictionary. and the key is (key of train_video) + "-" +"number of frame -10"
# example, if dic_training["ApplyEyeMakeup_g01_c01"]=163 then key of dic_training is "ApplyEyeMakeup_g01_c01 153"
# and
# '''


    def get_training_dic(self):
        self.dic_video_train={}
        for video in self.train_video:

            nb_clips = self.frame_count[video]-10#+1
            key = video +'-' + str(nb_clips)
            self.dic_video_train[key] = self.train_video[video]

# Description: < This function initiate "spatial_dataset" to preprocess our data. This function passes
# "transforms.Compose" which includes how our data will be scaled and convert into tensor.
# Then this "training_set" is passes to "Dataloader" class where data will be loaded
# to process further. >
# Output: < "train_loader" which contains training data >

    def train(self):
        training_set = motion_dataset(dic=self.dic_video_train, in_channel=self.in_channel, root_dir=self.data_path,
            mode='train',
            bidirectional=self.bidirectional,
            transform = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ]))
        print '==> Training data :',len(training_set),' videos',training_set[1][0].size()

        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
            )

        return train_loader

# Description: < This function initiate "spatial_dataset" to preprocess our data. This function passes
# "transforms.Compose" which includes how our data will be scaled and convert into tensor.
# Then this "validation_set" is passes to "Dataloader" class where data will be loaded
# to process further. >
# Output: < "val_loader" which contains training data >

    def val(self):
        validation_set = motion_dataset(dic= self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path ,
            mode ='val',
            bidirectional=self.bidirectional,

            transform = transforms.Compose([
            transforms.Scale([224,224]),
            transforms.ToTensor(),
            ]))
        print '==> Validation data :',len(validation_set),' frames',validation_set[1][1].size()
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader

if __name__ == '__main__':
    data_loader =Motion_DataLoader(BATCH_SIZE=1,num_workers=1,in_channel=10,
                                        path='/home/semanticslab3/development/python/two-stream-action/optical_flow/tvl1_flow',
                                        ucf_list='UCF_list/',
                                        ucf_split='01'
                                        )
    train_loader,val_loader,test_video = data_loader.run()
    #print train_loader,val_loader

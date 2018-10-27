import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from split_train_test_video import *
from skimage import io, color, exposure

class spatial_dataset(Dataset):
    def __init__(self, dic, root_dir, mode, transform=None):

        self.keys = dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.mode =mode
        self.transform = transform
    #returns length of keys
    def __len__(self):
        return len(self.keys)

# Description: < This function transforms a image according to "self.transform", for this first it
#     goes to that image directory and reads the image and then it transforms it. >
#     Output: < returns transformed image >

    def load_ucf_image(self,video_name, index):
        #going into that directory
        if video_name.split('_')[0] == 'HandstandPushups':
            n,g = video_name.split('_',1)
            name = 'HandstandPushups_'+g
            path = self.root_dir + 'HandstandPushups'+'/separated_images/v_'+name+'/v_'+name+'_'
        else:
            path = self.root_dir + video_name.split('_')[0]+'/separated_images/v_'+video_name+'/v_'+video_name+'_'

        img = Image.open(path +str(index)+'.jpg') #opening image
        transformed_img = self.transform(img) #transforming image
        img.close()

        return transformed_img

# Description: < This is a magic function and to preprocess data we need to overload this function.
#     For training data we select randomly 3 image and preprocessing it. >
# Output: < This function returns tuple of data and label if it is in training stage and if
#     it is in validation stage it returns tuple of video_name, data and label.>
# Note: < We have to access data during train or validation just the way this function returns
#     data. i.e. for training first we need to access the "data" and then "label" as it returns
#     "sample = (data, label)" and for validatoin first "video_name" then "data" and at the
#     last "label" >


    def __getitem__(self, idx):

        if self.mode == 'train':
            video_name, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            clips = []
            clips.append(random.randint(1, nb_clips/3))            #selecting
            clips.append(random.randint(nb_clips/3, nb_clips*2/3)) #three random
            clips.append(random.randint(nb_clips*2/3, nb_clips+1)) #

        elif self.mode == 'val':
            video_name, index = self.keys[idx].split(' ')
            index =abs(int(index))
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1

        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_ucf_image(video_name, index)

            sample = (data, label)
        elif self.mode=='val':
            data = self.load_ucf_image(video_name,index)
            sample = (video_name, data, label)
        else:
            raise ValueError('There are only train and val mode')

        return sample

class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, ucf_list, ucf_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.frame_count ={}
        #getting train_video and test_video from split_train_test_video.py which are dictionary
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
        with open('/home/semanticslab11/development/two_stream_bidirectional_pytorch/two-stream-action-original_bidirection/dataloader/dic/frame_count101.pickle','rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame :
            # if line is "v_ApplyEyeMakeup_g01_c01.avi" then videoname is "ApplyEyeMakeup_g01_c01"
            videoname = line.split('_',1)[1].split('.',1)[0]
            n,g = videoname.split('_',1)
            if n == 'HandstandPushups':
                videoname = 'HandstandPushups_'+ g
            self.frame_count[videoname]=dic_frame[line]
# """
# Description: < This function runs some functions >
# Output: < Returns Train_loader, val_loader and test_video
# """
    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample20()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video
# '''
# Description: This function reads the "train_video" dictionary and makes a new dictionary whoes value
# is 10 less than train_video's value because we will select image randomly and this will make sure
# that our randomly selected frame is within the bound. And we are making new key for "dic_training"
# dictionary which is our output dictionary. and the key is (key of train_video) + " " +"number of frame -10"
# example, if dic_training["ApplyEyeMakeup_g01_c01"]=163 then key of dic_training is "ApplyEyeMakeup_g01_c01 153"
# and
# '''

    def get_training_dic(self):
        self.dic_training={}
        for video in self.train_video:
            #print videoname
            nb_frame = self.frame_count[video]-10
            key = video+' '+ str(nb_frame)

            self.dic_training[key] = self.train_video[video]

# Descripton: <This function increases(19 times) the amount of test data and returns a dictionary
# whose key is the same as "dic_training" and contains value same as "test_video">

    def val_sample20(self):
        print '==> sampling testing frames'
        self.dic_testing={}
        for video in self.test_video:
            nb_frame = self.frame_count[video]-10
            interval = int(nb_frame/19)
            for i in range(19):
                frame = i*interval
                key = video+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video]

# Description: < This function initiate "spatial_dataset" to preprocess our data. This function passes
# "transforms.Compose" which includes how our data will be cropped, fliped, convert into tensor and
# normalized. Then this "training_set" is passes to "Dataloader" class where data will be loaded
# to process further. >
# Output: < "train_loader" which contains training data >

    def train(self):
        # print("train data dic, path: ", len(self.dic_training), self.data_path)
        training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print '==> Training data :',len(training_set),'frames'
        print training_set[1][0]['img1'].size()

        train_loader = DataLoader(
            dataset=training_set,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

# Description: < This function initiate "spatial_dataset" to preprocess our data. This function passes
# "transforms.Compose" which includes how our data will be scaled, convert into tensor and
# normalized. Then this "validation_set" is passes to "Dataloader" class where data will be loaded
# to process further. >
# Output: < "val_loader" which contains training data >

    def validate(self):
        # print("validation data dic, path: ", len(self.dic_testing), self.data_path)
        validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Scale([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))

        print '==> Validation data :',len(validation_set),'frames'
        print validation_set[1][1].size()

        val_loader = DataLoader(
            dataset=validation_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader





if __name__ == '__main__':

    dataloader = spatial_dataloader(BATCH_SIZE=1, num_workers=1,
                                path='/media/semanticslab11/hdd/data/two-stream-action/data/backup/ucf101/',
                                ucf_list='../UCF_list/',
                                ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()

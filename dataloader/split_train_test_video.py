import os, pickle


class UCF101_splitter():
    def __init__(self, path, split):
        self.path = path #path of ucf list
        self.split = split #which split we are taking
# """
# Description: <This function take the file "classInd.txt", reads it and strips new line from it.
#     then it generates a dictionary whose key is action class and value is corrosponding serial number.
#     such as action_label[ApplyEyeMakeup] has value "1" and action_label[ApplyLipstick] has value 2.
#     it means ApplyEyeMakeup is first action, and ApplyLipstick is second action.
#Output: < returns a dictionary. >
# """


    def get_action_index(self):
        self.action_label={}
        #the file classInd.txt contains all(101) actions and a serialized key.
        #such as 1 ApplyEyeMakeup, 2 ApplyLipstick. it means ApplyEyeMakeup is first action,
        #ApplyLipstick is second action and so on
        with open(self.path+'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            #spliting line "1 ApplyEyeMakeup", means label value is "1" and action value is ApplyEyeMakeup
            label,action = line.split(' ')
            #making dictionary where each class is hashed by its serial number
            if action not in self.action_label.keys():
                self.action_label[action]=label

# """ Description: <This function selects one txt file of three testlist*.txt and trainlist*.txt
# and this files are located in /../UCF_list/ >
# Input: < >
# Output: < Dictionary containing action subfolders and action number as its value.
# example dic["ApplyEyeMakeup_g01_c01"] contains value "1", as ApplyEyeMakeup is first action in our serial. """
    def split_video(self):
        self.get_action_index()
        for path,subdir,files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist'+self.split:
                    train_video = self.file2_dic(self.path+filename)
                if filename.split('.')[0] == 'testlist'+self.split:
                    test_video = self.file2_dic(self.path+filename)
        print '==> (Training video, Validation video):(', len(train_video),len(test_video),')'
        self.train_video =train_video# self.name_HandstandPushups(train_video)
        self.test_video =test_video# self.name_HandstandPushups(test_video)
        print '==> (After convert HandStandPushup_g Training video, Validation video):(', len(self.train_video),len(self.test_video),')'

        return self.train_video, self.test_video
# """
# Description: <This function generates a dictionary whose key is the name of the folder and value is its action
# class's serial number. example
# dic["ApplyEyeMakeup_g01_c01"] contains value "1", as ApplyEyeMakeup is first action in our serial >
# Input: < Takes a file, reads its line and strips new line from it. >
# Output:< A dictionary >
# """
    def file2_dic(self,fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic={}
        for line in content:
            #if line = "ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi", then video = "v_ApplyEyeMakeup_g01_c01.avi"
            #and key = "ApplyEyeMakeup_g01_c01"
            video = line.split('/',1)[1].split(' ',1)[0]
            key = video.split('_',1)[1].split('.',1)[0]
            label = self.action_label[line.split('/')[0]]
            dic[key] = int(label)
            #print key,label
        return dic

    # def name_HandstandPushups(self,dic):
    #     dic2 = {}
    #     for video in dic:
    #         n,g = video.split('_',1)
    #         if n == 'HandstandPushups':
    #             videoname = 'HandstandPushups_'+ g
    #         else:
    #             videoname=video
    #         dic2[videoname] = dic[video]
    #     return dic2


if __name__ == '__main__':

    path = '../UCF_list/'
    split = '01'
    splitter = UCF101_splitter(path=path,split=split)
    train_video,test_video = splitter.split_video()
    print len(train_video),len(test_video)

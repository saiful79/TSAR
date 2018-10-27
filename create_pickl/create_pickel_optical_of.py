import pickle
import os
root = "/media/semanticslab11/hdd/data/two-stream-action/data/tvl1_flow" # use it your optical flow dataset directory
txt_file = open('train_and_test.txt') #copy testlist01.txt and trainlist01.txt of all content and create train_and_test.txt.
#put train_and_test.txt directory into text_file
#find testlist01.txt and  trainlist01 file that are given into UCF_list
dic={}
for line in txt_file:
    # print(line[:-1])
    line = line[:-1]
    folder, video = line.split(" ")[0].split("/")
    video_folder=video.split(".")[0]
    org_folder = root +"/u/"+video_folder
    
    if not os.path.isdir(org_folder):
    	continue

    frame_count = len(os.listdir(org_folder))

    dic[video]=frame_count
    print(video,frame_count)


pickle_file = open("../dataloader/dic/frame_count_motion_101.pickle","wb")
pickle.dump(dic, pickle_file)
pickle_file.close()


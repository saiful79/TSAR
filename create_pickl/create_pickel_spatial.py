import pickle
import os
root = "/media/semanticslab11/hdd/data/two-stream-action/data/backup/ucf101/" #use ucf101 dataset directory
txt_file = open('/home/semanticslab11/development/two_stream_bidirectional_pytorch/two-stream-action-original_bidirection/UCF_list/test_train.txt')
#copy testlist01.txt and trainlist01.txt of all content and create train_and_test.txt.
#put train_and_test.txt directory into text_file
#find testlist01.txt and  trainlist01 file that are given into UCF_list
dic = {}
for line in txt_file:
    # print(line[:-1])
    line = line[:-1]
    folder, sub = line.split("/",1)
    sub=sub.split(".")[0]
#     print(folder, sub)
    org_folder = root + folder
    for frame_count in os.listdir(org_folder):
        org_folder = root + folder+"/"+frame_count+"/"+sub
#         print(org_folder)
        
        frame_total=len(os.listdir(org_folder))
        print(sub,frame_total)
        
        if not os.path.isdir(org_folder):
            continue

        sub=sub+".avi"
        dic[sub]=frame_total
        print(sub, frame_total)
        
pickle_file = open("../dataloader/dic/frame_count_spatial_101.pickle","wb")
pickle.dump(dic, pickle_file)
pickle_file.close()


# import pickle
# import os
# root = "/media/semanticslab11/hdd/data/two-stream-action/data/backup/ucf101/"
# txt_file = open('/home/semanticslab11/development/two_stream_bidirectional_pytorch/two-stream-action-original_bidirection/UCF_list/test_train.txt')
# dic = {}
# for line in txt_file:
#     # print(line[:-1])
#     line = line[:-1]
#     folder, video = line.split(" ")[0].split("/")
#     org_folder = root + "/" + folder + "/separated_images/" + video[:-4]
#     if not os.path.isdir(org_folder):
#         continue
#     # if org_folder[-1]==".":
#     #   org_folder = org_folder[:-1]
#     frame_count = len(os.listdir(org_folder))
#     dic[video]=frame_count
#     print(video, frame_count)
# pickle_file = open("frame_count_5.pickle","wb")
# pickle.dump(dic, pickle_file)
# pickle_file.close()



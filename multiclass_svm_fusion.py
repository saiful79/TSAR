from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
from utils import *
import dataloader
from sklearn import svm


if __name__ == '__main__':


    rgb_preds='record/spatial/spatial_video_preds.pickle'
    opf_preds = 'record/motion/motion_video_preds.pickle'

    with open(rgb_preds,'rb') as f:
        rgb =pickle.load(f)
    f.close()
    with open(opf_preds,'rb') as f:
        opf =pickle.load(f)
    f.close()

    dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1,
                                    path='/media/semanticslab11/hdd1/data/two-stream-action/data/backup/ucf101/',
                                    ucf_list='UCF_list/',
                                    ucf_split='01')
    train_loader,val_loader,test_video = dataloader.run()

    video_level_preds = np.zeros((len(rgb.keys()),2))
    video_level_labels = np.zeros(len(rgb.keys()))
    correct=0
    ii=0
    for optkey in sorted(opf.keys()):
        print(optkey)


    for name in sorted(rgb.keys()):
        r = rgb[name]
        o = opf[name]

        label = int(test_video[name])

        video_level_preds[ii,:] = (r+o)
        video_level_labels[ii] = label
        ii+=1
        if np.argmax(r+o) == (label):
            correct+=1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()


    # top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,2)) #use 5 instead of 2

    # print top1,top5


    #use multiclass svm
    svm_model=svm.SVC(kernel="rbf")
    
    # svm_model.fit(video_level_preds,video_level_labels)

    # print("multiclass svm prediction: {}".format(svm_model.predict(video_level_preds)))

	start_time = dt.datetime.now()
	print('Start param searching at {}'.format(str(start_time)))

	svm_classifier=svm_model.fit(video_level_preds,video_level_labels)
	print("print the svm mode score ..............................")
	print(svm_classifier.score(video_level_preds,video_level_labels))

	predic_report=svm_classifier.predict(video_level_preds)

	with open('/home/semanticslab3/development/python/saiful/nahid/two-stream-action-original_bidirection/multiclass_svm/my_dumped_classifier.pkl', 'wb') as fid:
	    pickle.dump(svm_classifier, fid)
	# print("the accuracy is :")
	print("Accuracy={}".format(metrics.accuracy_score(video_level_labels,predic_report)))

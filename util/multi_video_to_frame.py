import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='PyTorch Cats vs Dogs fine-tuning example')
parser.add_argument('--data', metavar='DIR', help='path to dataset')

def frame_extraction(i, video, real_path):
	out_dir = "data"
	video_frame=video.split('.')[0]
	#separated_images
	make_dir = out_dir + "/" + i + "/separated_images/" + video_frame
	if not os.path.exists(make_dir):
		os.makedirs(make_dir)

	cap = cv2.VideoCapture(real_path)
	# in this code not o bite frame extract 
	success,image = cap.read()
	count = 0
	success = True
	while success:
	    
	    print(i + '_' + video_frame + '_frame_%d.jpg'% count)

	    cv2.imwrite(make_dir+"/"+i+video_frame+'_%d.jpg'%count,image)
	    success,image = cap.read()
	    count+=1

def data_folder(data_path):
    for i in os.listdir(data_path):
        class_video = len(os.listdir(data_path + "/" + i))
#         print(class_video)

        for video in os.listdir(data_path + "/" + i):
#             print("file name: {0} and video: {1}".format(i,video))
            real_path=data_path + "/" + i + "/" + video
#             print(real_path)
            frame_extraction(i,video,real_path)
#             return real_path
    
if __name__ == "__main__":
	global args
	args = parser.parse_args()
	data_path=args.data
	if data_path:
		# give your vedio file path to extect frame
		# data_path = "/home/saiful/Desktop/data_set/ucf101"
		data_folder(data_path)
	else:
		print("python file_name.py --data /to/the/dataset")

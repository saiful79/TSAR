import shutil
import os
# shutil.move(dir_to_move, dest)
rp = "/path/to/ucf/data"
dest =  '/path/to/destination/'
# os.mkdir(dest+'ApplyEyeMakeup')
# os.mkdir(dest+'ApplyEyeMakeup'+ '/separated_images/')
action_list ='path/to/UCF_list/classInd_old.txt'

#creating 101 action folder and 'separated_images'  folder within each of those action folder
for line in open(action_list, 'r'):
    line = line.strip('\n')
    name = line.split(' ')[1]
    action_dir = dest + name
    os.mkdir(action_dir)
    os.mkdir(action_dir+'/separated_images')
    

for folder in os.listdir(rp):
    f_n = folder.split('_')[1]
    print('folder name ',f_n)
    fol_to_mov = rp + folder
    destination = dest + f_n + '/separated_images/' + folder
    os.mkdir(destination)
#     print('destination ', destination)
# #     print('fol ',fol_to_mov)
    if os.path.isdir(fol_to_mov):
        for fil in os.listdir(fol_to_mov):
            file_to_mov = fol_to_mov + '/'+ fil
            print(fil)
            print('is dir ', fol_to_mov)
            shutil.move(file_to_mov, destination)

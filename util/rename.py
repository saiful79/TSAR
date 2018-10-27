import os
img_root = '/path/to/ucf101/'
for folder in os.listdir(img_root):
    sub_fol_path = img_root+folder+ '/separated_images'
    print('folder ', sub_fol_path)
    for sub_fol in os.listdir(sub_fol_path):
#         print(sub_fol)
        images_path = sub_fol_path + '/' + sub_fol
        print('sub fol ',images_path)
        # images_path ='/home/semanticslab3/development/python/two-stream-action/data/ucf101/v_ApplyEyeMakeup_g01_c01/'
        image_list = os.listdir(images_path)
        for i, image in enumerate(image_list):
            ext = os.path.splitext(image)[1] #getting the extension
            if ext == '.jpg':
                print(ext)
                src = images_path + '/' + image
                dst = images_path + '/' + str(i) + '.jpg'
                os.rename(src, dst)

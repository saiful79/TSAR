## Two-Stream-Action-Recognition

We use a spatial and motion stream cnn with ResNet101 for modeling video information in UCF101 dataset.

## Dependency
* python 2.7.15
* pytorch 0.4.1
* torchvision 0.2.1
* tqdm 4.26.0
* pillow 5.3.0
* opencv-python 3.4.3.18
* numpy 1.15.2
* scikit-image  0.14.1
* scikit-learn  0.20.0
* cuda 9.0
* cudnn 7.1

##File and Folder structure

The File and folder structure of two stream action recognition.
````
                        ├── average_fusion.py
                        ├── create_pickl
                        │   ├── create_pickel_optical_of.py
                        │   └── create_pickel_spatial.py
                        ├── dataloader
                        │   ├── dic
                        │   │   ├── frame_count101.pickle
                        │   │   └── frame_count_opf_101.pickle
                        │   ├── __init__.py
                        │   ├── motion_dataloader.py
                        │   ├── spatial_dataloader.py
                        │   └── split_train_test_video.py
                        ├── GUI_DEMO
                        │   ├── models
                        │   │   ├── classInd.txt
                        │   │   ├── motion
                        │   │   └── spatial
                        │   ├── network.py
                        │   ├── Readme.md
                        │   ├── two_stream_action.ui
                        │   └── two_stream_ui.py
                        ├── motion_cnn.py
                        ├── multiclass_svm_fusion.py
                        ├── network.py
                        ├── Readme.md
                        ├── Readme.pdf
                        ├── record
                        │   ├── motion
                        │   │   ├── checkpoint.pth.tar
                        │   │   ├── model_best.pth.tar
                        │   │   ├── motion_video_preds.pickle
                        │   │   ├── opf_test.csv
                        │   │   └── opf_train.csv
                        │   └── spatial
                        │       ├── checkpoint.pth.tar
                        │       ├── model_best.pth.tar
                        │       ├── rgb_test.csv
                        │       ├── rgb_train.csv
                        │       └── spatial_video_preds.pickle
                        ├── spatial_cnn_from_scratch.py
                        ├── spatial_cnn.py
                        ├── UCF_list
                        │   ├── classInd.txt
                        │   ├── testlist01.txt
                        │   ├── train_and_test.txt
                        │   └── trainlist01.txt
                        ├── util
                        │   ├── move_file.py
                        │   └── rename.py
                        └── utils.py


````
 
## Spatial input data -> rgb frames
Download the preprocessed RGB images data directly from,
  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003

Marge files
  * cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
  * unzip ucf101_jpegs_256.zip

  ####ucf101 data processing method
 The ucf101 data must be need to prepocess befor traning.the util folder contain the prepocess script.
 #### Dataset move
 Please modify the script `move_file.py`,
 ````
rp = "/path/to/the/ucf101"
dest =  '/path/to/the/destination/'
````
* Run the script `move_file.py` and data will move to the destination folder.

	####Rename the Destination dataset

	Use your destination directory `rename.py` script and run the script.

## Motion input data -> stacked optical flow images
In motion stream,
Download the preprocessed `tvl1` optical flow dataset directly from

  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001
  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002
  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003

Marge files
  * cat ucf101_tvl1_flow.zip* > ucf101_tvl1_flow.zip
  * unzip ucf101_tvl1_flow.zip


## Generate pickl file
Generate pickle for corresponding `UCF-101` video frame. To do that

For `Spatial CNN` run `create_pickel_spatial.py` script. In the script line no `3, 4` chagne the path ,
```sh
root = "path/to/the/ucf101/"
txt_file = open('path/to/the/UCF_list/test_train.txt')
```
where `test_train.txt` is the combination of `trainlist01.txt` and `testlist01.txt`. The pickle file named `frame_count_spatial_101.pickle` file is generated in the `dataloader/dic/` directory.

For `Motion CNN` run `create_pickel_optical_of.py` script In the script line no `3, 4` change the path,
```
root = "path/to/the/tvl1_flow"
txt_file = open('path/to/the/UCF_list/test_train.txt')
```
The pickle file named `frame_count_motion_101.pickle` is generated in the `/dataloader/dic/` directory.


##Training and Evaluate models
### Spatial Stream
Please modify the script `spatial_cnn.py` in line number 60
```
# path = "path/to/the/ucf101/"
```
Please modify the script `spatial_dataloader.py` in line number 104 and 214
```
with open('path/to/the/dataloader/dic/frame_count_spatial_101.pickle', 'rb')
path="path/to/the/ucf101/" of spatial_dataloader.py
```
####Pretrain + Fineune
To modify script `spatial_cnn.py` in line number `45,46`

````
parser.add_argument('--fine-tune-flag', default = True, type = bool, help = 'set fine tune flag. default is True')

parser.add_argument('--last-layer-flag', default = False, type = bool, help = 'set last layer flag. default is False')
````

To Train model,

`python spatial_cnn.py`

To Train model use pretrain model and run,

`python spatial_cnn.py --resume PATH_TO_PRETRAINED_MODEL`

For Evaluate model run,

`python spatial_cnn.py --resume PATH_TO_PRETRAINED_MODEL --evaluate`


####Pretrain  + last layer
To modify script `spatial_cnn.py` in line number `45,46`

````
parser.add_argument('--fine-tune-flag', default = False, type = bool, help = 'set fine tune flag. default is True')

parser.add_argument('--last-layer-flag', default = True, type = bool, help = 'set last layer flag. default is False')
````


To Train model,
`python spatial_cnn.py`

To Train model use pretrain model and run,

`python spatial_cnn.py --resume PATH_TO_PRETRAINED_MODEL`

For Evaluate model run,

`python spatial_cnn.py --resume PATH_TO_PRETRAINED_MODEL --evaluate`



####From scratch the spatial model implementation script

Please modify the script `spatial_cnn_from_scratch.py` line number 112,
````
path='path/to/the/ucf101'

````
To Train model run,

`python spatial_cnn_from_scratch.py`


To Train use pretrain model and run,
`python spatial_cnn_from_scratch.py --resume PATH_TO_PRETRAINED_MODEL`

For Evaluate model run,

`python spatial_cnn_from_scratch.py --resume PATH_TO_PRETRAINED_MODEL --evaluate`


### Motion stream

Please modify the script `motion_cnn.py` in line number 42
```
path='path/to/the/tvl1_flow/'
```
Please modify the script `motion_dataloader.py` in line number 249 and 152
```
path='path/to/the/tvl1_flow/'
with open('path/to/the/dataloader/dic/frame_count_motion_101.pickle','rb')
```

###Unidirectional
To Train unidirectional model ,
`python motion_cnn.py`

To Train use pretrain model and run,

`python motion_cnn.py --resume PATH_TO_PRETRAINED_MODEL`

For Evaluate model run,

`python motion_cnn.py --resume PATH_TO_PRETRAINED_MODEL --evaluate`
###Bidirectional
To Train bidirectional model,
`python motion_cnn.py --bidirectional True`

To Train use pretrain model and run,

`python motion_cnn.py --resume PATH_TO_BIDIRECTIONAL_PRETRAINED_MODEL --bidirectional True`

For Evaluate model run,

`python motion_cnn.py --resume PATH_TO_PRETRAINED_MODEL --evaluate --bidirectional True`


##Fusion
The Fusion method avarage two model prediction.we avarage spatial and motion model prediction use of fusion method.
####Avarage fusion
Please modify the script `average_fusion.py` in line number 22
````
path='path/to/the/ucf101/'
````
Run the `average_fusion.py` script and it show the avarage fusion prediction.

####Multiclass svm
Please modify the script `multiclass_svm_fusion.py` in line number 24
````
path='path/to/the/ucf101/'
````
Run the `multiclass_svm_fusion.py` script and it show the multiclass svm prediction.

##GUI Demo
For GUI, run the script `two_stream_ui.py`.
if you change the pretrain model then you change model path.

Change the script `two_stream_ui.py` model path, line number 43
```
model_path = "models/spatial/model_best_spatial_101.pth.tar"
```
Change the script `two_stream_ui.py` pretrain model path, line number 164
```
model_path = "models/motion/model_best_101.pth.tar"
```



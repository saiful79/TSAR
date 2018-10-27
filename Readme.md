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


## Spatial input data -> rgb frames
Download the preprocessed RGB images data directly from,
  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003

Marge files
  * cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
  * unzip ucf101_jpegs_256.zip
## Motion input data -> stacked optical flow images
In motion stream,
Download the preprocessed `tvl1` optical flow dataset directly from

  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001
  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002
  * wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003

Marge files
  * cat ucf101_tvl1_flow.zip* > ucf101_tvl1_flow.zip
  * unzip ucf101_tvl1_flow.zip

##Training and Evaluate models
### Spatial Stream
To Train model,

`python spatial_cnn.py`

To Train model use pretrain model and run,

`python spatial_cnn.py --resume PATH_TO_PRETRAINED_MODEL`

For Evaluate model run,

`python spatial_cnn.py --resume PATH_TO_PRETRAINED_MODEL --evaluate`


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


To Train unidirectional model ,
`python motion_cnn.py`

To Train use pretrain model and run,

`python motion_cnn.py --resume PATH_TO_PRETRAINED_MODEL`

For Evaluate model run,

`python motion_cnn.py --resume PATH_TO_PRETRAINED_MODEL --evaluate`



##Fusion
The Fusion method avarage two model prediction.we avarage spatial and motion model prediction use of fusion method.
####Avarage fusion
Please modify the script `average_fusion.py` in line number 22
````
path='path/to/the/ucf101/'
````
Run the `average_fusion.py` script and it show the avarage fusion prediction.




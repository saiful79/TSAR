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




##Reference Paper
````Two-Stream Convolutional Networks for Action Recognition in Videos````
http://papers.nips.cc/paper/5353-two-stream-convolutional-networks-for-action-recognition-in-videos.pdf
 
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

To Train unidirectional model ,
`python motion_cnn.py`

To Train use pretrain model and run,

`python motion_cnn.py --resume PATH_TO_PRETRAINED_MODEL`

For Evaluate model run,

`python motion_cnn.py --resume PATH_TO_PRETRAINED_MODEL --evaluate`

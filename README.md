# A-bi-directional-message-passing-model-for-salient-object-detection

## Introduction
This package contains the source code for [A Bi-directional Mssage Passing Model for Salient Object Detection](https://drive.google.com/file/d/1VRGKXaAqxJDhqx5YoMO09gjtMNgqdHgA/view?usp=sharing), CVPR 2018. This code is tested on Tensorflow 1.2.1 Ubuntu14.04.
## Usage Instructions
Test
* Instill these requirements if necessary: Python 2.7, Tensorflow 1.2.1, Numpy, Opencv.
* Put your test images in the `./data` directory.
* Download the pretrained model from [here](https://pan.baidu.com/s/1ZSUW8YPvLR9mRjZ7_ISVnw), and put it under the `./model` directory.
* Run `TestingModel.py` to generate saliency map.

Train
* Built a filename.txt for your training data, and revise the data path in `TrainingModel.py`.
* Run `TrainModel.py` for training the saliency model.
## Saliency Map
Saliency map of this paper can be downloaded [BaiduYun](https://pan.baidu.com/s/16kdXjC8HC0gvnKpdqQJ9uA), [GoogleDrive](https://drive.google.com/open?id=1I283XrnYzgY6mk70b5fhYAHAy7oMVQYw).
# Citation
    @InProceedings{Zhang_2018_CVPR,
        author = {Zhang, Lu and Dai, Ju and Lu, Huchuan and He, You and Wang, Gang},
        title = {A Bi-Directional Message Passing Model for Salient Object Detection},
        booktitle = CVPR,
        year = {2018}}

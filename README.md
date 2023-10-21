# CamoTectV2 Overview:
This document provides instructions for setting up and running CamoTectV2. CamoTectV2 uses SINetV2 (Search and Identification Network Version 2) for semantic segmentation and SINetProc (Search and Identification Network Processor) for object detection and statistics gathering. The objective of this tool is to detect camouflaged heads against a camouflaged background.

The two main goals CamoTectV2 achieves is image classification (SINetProc) and image segmentation (SINetV2). SINetProc is designed to create bounding boxes around the head object, edge detection, and image classification which involves identifying the class of an image as being a positive or negative image. An image is classified as being a positive image if there is a head detected somewhere within it. If there is no head detected in an image, then it will be classified as being a negative image. SINetV2 is the deep neural network used to outline the image region of where the head object is.


Note that SINet-V2 is only tested on Ubuntu OS and Windows OS
It may work on other operating systems (i.e. MacOS) as well but it is not guaranteed that it will properly function.

The training and testing experiments are conducted using [Pytorch] (https://github.com/pytorch/pytorch) with Geforce 2080 SUPER 8GB Memory, 32GB ram, Windows 10 Pro


CamoTectV2 is designed to work with image data stored in the following directory structure:

    Dataset/
        Train/
            Imgs/
                image1.jpg
                image2.jpg
                ...
            GT/
                image1.jpg
                image2.jpg
                ...
        Valid/
            Imgs/
                image1.jpg
                image2.jpg
                ...
            GT/
                image1.jpg
                image2.jpg
                ...
        Test/
            Imgs/
                image1.jpg
                image2.jpg
                ...

1) Training: `./Dataset/Train/Imgs` should contain the original background images and `./Dataset/Train/GT` contains the ground truth images corresponding to each of the original images.
2) Validation: `./Dataset/Valid/Imgs` and  `./Dataset/Valid/GT` contain validation data for use during training. The validation images should be similar to the train images but not contain the same exact images.
3) Testing: Test images are stored under `./Dataset/Test/Imgs`. The provided dataset can be used for testing or a new test dataset can be generated from the image generator.



## Installation:
1) [CUDA ToolKit](https://developer.nvidia.com/cuda-10.2-download-archive)

2) Navigate to the CamoTectV2 project directory

3) Run the following command to install all required dependencies: `pip install -r requirements.txt`
   
4) Run via command line



# Usage
Place your image data in the appropriate directories. Ensure the images are placed as described above under `CamoTectV2 Overview`.

## How to Use:
1) See `python train.py -h` or `python train.py --help` for all options

2) Execute `python train.py` to train a new model 
-   a) Training output is saved under `./weights/SINet_V2/`

    b) `python train.py --dataname [dataname]` to train while saving a copy of the current training session (replace `[dataname]`).
        i) Copy is saved under, `./train_output/[dataname]`

    c) `python train.py --load [path to best weight file]` to load a pre-trained weight.
        i) Example command: `python train.py --load ./weights/SINet_V2/Net_epoch_best.pth`
   

3) Execute `tensorboard --logdir=[path to summary folder]` and paste the address from the command line for your local host in a web browser to view the training summary(replace `[path to summary folder]` with file path).
    a) Example command: `tensorboard --logdir=./weights/SINet_V2/summary/`

4) Execute `python test.py` to generate SINetV2 results. 
-   a) Results are saved under `./sinet_res/[current date/time]`

    b) A copy of the latest SINetV2 test results will be saved under `./sinet_output` for use in SINetProc. This folder is overwritten with every test run.

5) Execute `python sinetProc.py` to generate SINetProc detection results and statistics.
-   a) `python sinetProc.py —advStats` to get true positive, false positive, true negative, and false negative numbers and percentages (names for negative images in test dataset must contain keywords “no” or “neg” somewhere)
    b) `python sinetProc.py –waste [waste value]` to set the waste ratio cutoff for image classification(determining whether a detected object is a head or not). The default value is 0.7. A waste cutoff of 0 would mean no filtering, and 1.01 means no objects could be classified as heads(waste ratio is number of ‘signal’ pixels divided by number of ‘waste’ + ‘signal’ pixels per object)

    b) Results are saved under `./sinetProc_output`



### Producing Good Training Weights
After many training runs using SINetV2, we found that when running the first round of training, it was best to use a training and validation dataset consisting of only positive images. The weights from this training run would act as the pre-trained weights.

These pre-trained weights are later loaded using the following command: `python train.py –load [path to best weight file]`,  when introducing noise within the next dataset to make the model more robust. For the size of the datasets, 5000 images is the minimum number of images we used.

[Generating a Dataset Using Image Generator](./../ImageGen/README.md)


Below is an example of how SINetV2 training would take place training in batches of 5000 images:

#### Creating initial pre-trained weights:
Train Dataset: 5000 Positive Background/Ground Truth Images
Valid Dataset: 5000 Positive Background/Ground Truth Images (different from train)
`python train.py –dataname “run1”`

#### Introducing noise to the model:
Train Dataset: 2500 Positive & 2500 Negative Background/Ground Truth Images
Valid Dataset: 5000 Positive Background/Ground Truth Images (different from train)
`python train.py –load ./train_output/run1/Net_epoch_best.pth –dataname “run2”`

# Efficient Few-shot Learning based on Channel Selective Relation Network for Facial Expression Recognition (CSRN)

Chae-Lin Kim, Ji-Woo Kang and Byung-Gyu Kim
Intelligent Vision Processing Lab. (IVPL), Sookmyung Women's University, Seoul, Republic of Korea

<hr>

This repository is the official PyTorch implementation of the paper submitted in Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24).


<hr>

### Summary of paper

Abstract
> Few-shot learning-based facial expression recognition (FER) aims to achieve maximum efficiency from a few numbers of data. Therefore, it is significant to
> utilize the given training dataset efficiently. Furthermore, it is essential to derive useful information from the organic movement of facial parts such as
> the eyes, nose, and mouth in usual FER. However, the generalization performance of the previous few-shot FERs still has a large variation. To address this
> problem, we propose a Channel Selective Relation Network with channel selection module and spatial data construction to extract more suitable facial
> features. Our method helps the network to prevent irrelevant information and focus on essential information by comparing the original sample features with
> the averaged feature. Also, our network learns dominant facial expression features in local patches, such as the eyes and lips. We verify that the optimal
> feature selected for facial images and additional spatial information can improve the generalization performance. We set 4-way 4-shot for training and 3
> way 3-shot for inferencing unseen facial emotion classes. Compared to the existing method, the average performances on RAFDB, FER2013, SFEW, and AFEW
> datasets are increased by 3.5%, 3.68%, 5.58%, and 2.31% accuracy, respectively.


Network Architecture

![model_network](https://github.com/smu-ivpl/CSRN/assets/53431568/be7e786f-2a17-41e4-b8ee-c0158663d4e6)


Experimental Results

![실험결과](https://github.com/smu-ivpl/CSRN/assets/53431568/8baa79c3-c639-4a74-bd1d-f6e02f9d8414)

<hr>

## Getting Started

### Installation

~~~
git clone https://github.com/smu-ivpl/CSRN.git

pip install -r requirements.txt
~~~

### Environment
- Ubuntu 18.04
- Cuda 10.1
- torch 1.12.1+cu113
- torchvision 0.13.1+cu113


### Dataset preparation

We used RAF-DB, FER2013, AFEW, and SFEW as training and testing.

> 1. Download
> 
> please download dataset from the official website.
> 
> 2. Arrange data
> 
> place original data and cropped eye and lip images on the specific ordering:

```bash
ex) RAFDB
|--dataset
|    |--RAFDB
|         |--images
|         |--images_eye
|         |--images_lip
|         |--test
|         |--train
```             

### Trained model

trained model is available in ./weights


## Training

Run in ./CODE

~~~
python csrn_main.py
~~~


## Testing

~~~
python test_csrn.py
~~~



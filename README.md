# Open-Set-FER
Official implementation of the AAAI2024 paper: [Open-Set Facial Expression Recognition](https://arxiv.org/abs/2401.12507)




## Abstract
Facial expression recognition (FER) models are typically trained on datasets with a fixed number of seven basic classes. However, recent research works point out that there are far more expressions than the basic ones. Thus, when these models are deployed in the real world, they may encounter unknown classes, such as compound expressions that cannot be classified into existing basic classes. To address this issue, we propose the open-set FER task for the first time. Though there are many existing open-set recognition methods, we argue that they do not work well for open-set FER because FER data are all human faces with very small inter-class distances, which makes the open-set samples very similar to close-set samples. In this paper, we are the first to transform the disadvantage of small inter-class distance into an advantage by proposing a new way for open-set FER. Specifically, we find that small inter-class distance allows for sparsely distributed pseudo labels of open-set samples, which can be viewed as symmetric noisy labels. Based on this novel observation, we convert the open-set FER to a noisy label detection problem. We further propose a novel method that incorporates attention map consistency and cycle training to detect the open-set samples. Extensive experiments on various FER datasets demonstrate that our method clearly outperforms state-of-the-art open-set recognition methods by large margins.


## Train


**Dataset**

Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), put the raf-basic folder under the dataset folder:
```key
-dataset/
  raf-basic/
	   Image/aligned/
	     train_00001_aligned.jpg
	     test_0001_aligned.jpg
	     ...

```

**Pretrained backbone model**

The pretrained ResNet-18 model can be downloaded from [here](https://github.com/zyh-uaiaaaa/Relative-Uncertainty-Learning).

The pretrained ResNet-50 model can be downloaded from [here](https://drive.google.com/file/d/1yQRdhSnlocOsZA4uT_8VO0-ZeLXF4gKd/view?usp=sharing).

Put the downloaded model files under the model directory. 

**Closed model**

Train a FER model on the close-set, save the model as "0.pth". '0' represents the open class, can be changed to other expression classes. 

```key
python close_set_training.py
```
**Pseudo labels**

Run "predict_pseudo_labels.ipynb" to predict pseudo labels for all test samples and save pseudo labels in "predicted_labels_open0.txt" and indexes in "predicted_labels_open0_openset_index.txt". 

**Detection**

Run open_set_detection_test.ipynb to evaluate the open-set detection method.

## Results


**Accuracy**

The detection performance of state-of-the-art open-set recognition methods on open-set FER. We start with the one-class open-set FER and utilize two common metrics AUROC (higher the better) and FPR@TPR95 (lower the better) for evaluation. The expression class listed on the left is the open-set class (Sur.: Surprise, Fea.: Fear, Dis.: Disgust, Hap.: Happiness, Sad.: Sadness, Ang.: Anger, Neu.: Neutral). 'R' represents RAF-DB and 'F' represents FERPlus. 'PROS' is 'PROSER' for short. Our method outperforms the state-of-the-art open-set recognition methods on open-set FER tasks with very large margins.

![](https://github.com/zyh-uaiaaaa/Open-Set-FER/blob/main/figures/detection_performance.png)

**Visualization**

Confidence scores of different methods. AUROC of each method is marked below. The baseline method fails as FER data have small inter-class distances, making open-set data have the same range of confidence scores as close-set data. Close-set and open-set data are separated by DIAS and PROSER while they still overlap a lot. Our method transforms open-set FER to noisy label detection and effectively separates close-set and open-set samples.

![](https://github.com/zyh-uaiaaaa/Open-Set-FER/blob/main/figures/confidence_scores.png)

The learned features of baseline and our method. The features are shown with the latent truth. Open-set features are marked as red. The learned open-set features of the baseline method are mixed with close-set features, while our method does not overfit the wrong pseudo labels of open-set samples and separates open-set features from close-set features.

![](https://github.com/zyh-uaiaaaa/Open-Set-FER/blob/main/figures/tsne.png)




**Citation**

If you find our code useful, please consider citing our paper:

```shell
@article{zhang2024open,
  title={Open-Set Facial Expression Recognition},
  author={Zhang, Yuhang and Yao, Yue and Liu, Xuannan and Qin, Lixiong and Wang, Wenjing and Deng, Weihong},
  journal={arXiv preprint arXiv:2401.12507},
  year={2024}
}
```

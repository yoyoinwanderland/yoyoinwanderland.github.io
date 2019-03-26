---
title: How to Structure DL Project
date: 2017-11-13 10:42:04
tags: 
- Machine learning
- CNN
category: 
- 时习之
- Deep Learning
description: Learning note Andrew's course Structuring Machine Learning Projects
---

If you have time for only one course from Andrew NG, I will recommend this one -  [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects/home/welcome). It doesn't go into too much technical details, but it provides a very useful way to structure and debug deep learning problems. This note won't follow the same logistics as the original course, instead I will just note down the key steps to organize and then debug deep learning projects. 

Note: all the knowledge and examples come from Andrew's lesson, and the quoted part come as the original quiz questions. I won't take any credit for anything.



## Guideline

* **Set up dev/ test set & metric**
* **Build initial system quickly**
* **Use Bias/Variance analysis to prioritize next step **



## Design the System

### Metric of Success

One important thing is to use a single number evaluation metrics. 

* **Case One**


| Classifier | Precision | Recall |
| ---------- | --------- | ------ |
| A          | 95%       | 90%    |
| B          | 98%       | 85%    |

 This brings confusion which classifier to choose. You better use F1 if you care about both metrics.

* **Case Two**


| Classifier | Accuracy | Running Time |
| ---------- | -------- | ------------ |
| A          | 90%      | 80 ms        |
| B          | 92%      | 90 ms        |
| C          | 95%      | 1,500 ms     |

  We can make accuracy as optimizing metric and running time as satisficing metric.



### Structuring the Data

* **Small data set**: 60-20-20
* **\> 100,000 examples**: 98-1-1
* **Mismatched training and dev/test set:**: dev & test set should be images of interest

> The distribution of data you care about contains images from your car’s front-facing camera; which comes from a different distribution than the images you were able to find and download off the internet. How should you split thedataset into train/dev/test sets?
>
> * Mixall the 100,000 images with the 900,000 images you found online. Shuffleeverything. Split the 1,000,000 images dataset into 600,000 for the trainingset, 200,000 for the dev set and 200,000 for the test set. (No. This won't measure how well the system performs on the images of interest - which is the images from car's front facing cameras in this case.)
> * *<u>Chooset he training set to be the 900,000 images from the internet along with 80,000 images from your car’s front-facing camera. The 20,000 remaining images will besplit equally in dev and test sets.</u>*



### Network Architecture

#### Transfer Learning

* **Two types:**
  * small set - retrain the last layers
  * Enough data - initialize the last layer randomly and retrain all the network
* **When to use transfer learning**:
  * Task a & b have same  input (all images, voice etc.)
  * Have a lot more data for task A than task B
  * Low level features from A could be helpful for learning B



#### Multi-Task Learning

Used in anonymous driving (pedestrians, signals, stop signs, other vehicles etc.), and multi-object detections.

* **When to use Multi-task learning:**
  * Training on a set of tasks that could benefit from having shared lower-level features
  * Usually: amount of data you have for each task is quite similar
  * Can train a big enough neural network to do well on all the tasks



#### End-to-End Learning

Used in machine translation, audio transcription. But some tasks, like facial recognition (find face - compare face), handbone age estimation (estimate average length of handbone - look up age mapping), are better off using traditional two-pass methods.

* **Pros:**
  * Let the data speaks
  * Less hand designingof components needed
* **Cons:**
  * Need large amount of data (x,y)
  * Excludes potentially useful hand-designed components



## Bias/Variance Analysis

**<u>Benchmark: BEST Human Level Performance</u>** (ask people to label images if you don't know!)

* **Case One**


| Human Level Performance | 1%      |
| ----------------------- | ------- |
| **Training Error**      | **8%**  |
| **Dev Error**           | **10%** |

  Prioritize to reduce bias

* **Case Two**


| Human Level Performance | 7.5%    |
| ----------------------- | ------- |
| **Training Error**      | **8%**  |
| **Dev Error**           | **10%** |

  Prioritize to reduce variance

* **Case Three**

  In this case, we have mismatched training and dev/test set. 


| Human Level Performance                  | 7.5%    |
| ---------------------------------------- | ------- |
| **Training Error** on mixed user images and internet images | **8%**  |
| **Dev Error **on only user images        | **10%** |


  We will have no idea if it's variance or it's the mismatched training and dev/test set. Instead, we analyze:



| Human Level Performance                  | 7.5%     |
| ---------------------------------------- | -------- |
| **Training Error** on mixed user images and internet images | **8%**   |
| **Training-dev Error** on mixed user images and internet images | **9.6%** |
| **Dev Error **on only user images        | **10%**  |
​    If the result as above, then we know our model has high variance that we want to minimize.



| Human Level Performance                  | 7.5%     |
| ---------------------------------------- | -------- |
| **Training Error** on mixed user images and internet images | **8%**   |
| **Training-dev Error** on mixed user images and internet images | **8.1%** |
| **Dev Error **on only user images        | **10%**  |

  If the result is as above, then we know our model has a high data mismatch problem.



## Error Analysis

### How to Perform Error Analysis

Pick 100/ 500 mislabeled images, and then create a table as follows:	



| No.  | Big Cat | Dog  | Blurry | Incorrectly Labeled | Comments |
| ---- | ------- | ---- | ------ | ------------------- | -------- |
| 1    | Y       |      |        |                     |          |
| 2    |         | Y    |        |                     |          |
| 3    |         |      | Y      |                     |          |
| 4    |         |      | Y      |                     |          |
| 5    |         |      |        | Y                   |          |
| 6    |         |      |        | Y                   |          |

And then decide what to prioritize first.


### Incorrectly Labeled Images

#### Definition:

* **Incorrectly labeled images:** the labeler (people) got it wrong (input data to classifier have wrong labeles)
* **Mislabled images:** classifier got it wrong.

#### When to fix it:

- **In training set ** 

   DL algorithms are quite robust to random errors in the training set. But not robust tosystematic errors. So for the first case, we can safely leave it in the training set.

- **In dev/ test set**

  the purpose of dev set is to pick the better classifier. If, for example, the incorrectly labeled training is 0.6%; we have the following two classifiers; It makes a significant differences to the performance on the dev set and test set. And we need to fix it. 



| Classifier A    | Classifier B    |
| --------------- | --------------- |
| 1.9% error rate | 2.1% error rate |



## Improve Performance

At the end of the day we need to know what to do next. So below is the action plans:

* **High Avoidable Bias:**

  - Bigger model
  - Train longer
  - Use better optimization algo (Adds momentum or RMS prop, or use a better algorithm like Adam)
  - change NN architecture, hyperparam search

* **High Variance:**

  - Bigger training set

  - Regularization

    * L2

    - Drop out
    - Data augmentation

  - NN architecture/ hyperparam search

* **Data Mismatch Problem**

  For example, the training set come from both in-car audio and audio from internet. We can add car noises to the audio from Internet and creates artificially synthesized in-car audio.

  > You decide to use data augmentation to addressfoggy images. You find 1,000 pictures of fog off the internet, and “add” the mto clean images to synthesize foggy days.
  >
  > *<u>So long as the synthesized fog looks realistic to the human eye, you can be confident that the synthesized data is accurately capturing the distribution ofreal foggy images, since human vision is very accurate for the problem you’re solving.</u>*

* **Incorrectly Labeled Problem**

  * Apply the same process (for example, if you hire someone to re-label data for you) on both dev and test set, to make sure they come from the same distribution
  * Consider examples your algorithm got right as well as ones got wrong.
  * Train and dev/test sets may now come from slightly different distributions.

* **Other Problem from Error Analysis**

  * Train a classifier just for big cat/ white dog problem if necessary.
  * Improve performance on blurry images. I think I can do data augmentation - create more blurry images WITH labels.
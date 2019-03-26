---
title: Deep Learning Review Notes
date: 2017-03-19 20:05:22
tags: 
- TensorFlow
- CNN
- Machine Learning
category: 
- 时习之
- Deep Learning
description: 为了明天面试理清思路,复（预）习一下Deep Learning
---




## Key Layers in a CNN Network

> Convolutional neural networks make strong and mostly correct assumptions about the nature of images (namely, stationarity of statistics and locality of pixel dependencies). - AlexNet



### Convolutional Layer

A convolutional layer compromise multiple convolutional kernels (image kernels). Below we will introduce what is an image kernel, why there are multiple image kernels than one, some computational details, and why a smaller image kernel is preffered in most recent research and models.

#### What is an image kernel

![kernel](https://ujwlkarn.files.wordpress.com/2016/07/convolution_schematic.gif?w=268&h=196&zoom=2)

 ​									[image source](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

 So above picture demonstrates perfectly how a kernel of 3*3 with stride =1 works. Basically it slides through the picture one pixel a time (thus stride = 1), and performs element wise multiplication. Why do we need this? Here is a more straightforward summary of what a kernel can do:

 ![imagekernel](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-05-at-11-03-00-pm.png)

 ​								[image source](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

 Besides the above demonstrations, image kernels could also retain certain colors of the image, or bottom sobel etc. [Here](http://setosa.io/ev/image-kernels/) is a link you can play with and get more understanding towards image kernels.

#### Why there are multiple image kernels in a convolutional layer? 

Here is a good answer (by [Prasoon Goyal](https://www.quora.com/profile/Prasoon-Goyal)) for the first question:

> But clearly, why would you want only *one* of those filters? Why not all? Or even something that is a hybrid of these? So instead of having one filter of size 5×55×5 in the convolution layer, you have kk filters of the same size in the layer. Now, each of these is independent and hopefully, they'll converge to different fitlers after learning. Here, k is the number of output channels. kk could be taken anywhere between few tens to few thousands. 

#### How to compute?

So, the input of a convolutional neural network ususally has three dimensions: long, wide, and color dimension (if grey scale then two dimensions). [Here](http://cs231n.github.io/convolutional-networks/#conv) is a good demo of how things work. 

Explained in text, if the input source is 7\*7\*3, and we have 2 kernels of size 3\*3\*3 with stride 2:

- The input could be seen as 3 stacked 2d 7\*7 matrices; and each kernel could be seen as 3 stacked 3\*3 matrix;
- Each stacked layer of the kernel will slide through the corresponding stacked layer of input with step of 2 pixels each time, and the element sum will be furthur **added** together. 
- The output after each kernel sliding will then be a 3*3 2d matrix. Here is why: (input width - kernel width + 2 zero padding)/stride + 1 = (7-3+0)/2 + 1 = 3. As we have two kernels, so the output volume will be 3\*3\*2.

#### Smaller kernel size preferred

Another important thing to know is that, smaller kernels capture more details of pictures. This helps to understand the evolvement on CNNs. A demonstration is as below:

| Kernel size: 3\*3                        | Kernel size: 10\*10                      |
| ---------------------------------------- | ---------------------------------------- |
| ![kernel3x3](https://cdn-images-1.medium.com/max/1600/1*s3MMrbrwtxsmj5g6KBGtHQ.png) | ![10x10 kernel](https://cdn-images-1.medium.com/max/1600/1*OZqJFAj5f_ogyB8ap0arrQ.png)[image source](https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59) |

### Pooling Layer

Pooling layer is often used after convolutional layer for down sampling. It reduces the amount of parameters carried forward while retaining the most useful information; thus it also prevents overfitting. A demonstration for max pooling could be shown in following picture: 

![pooling](https://cdn-images-1.medium.com/max/2000/1*Feiexqhmvh9xMGVVJweXhg.gif)

​								[Image source](https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59)

Some visualizations could be found below:

| Feature Map                              | After Pooling                            |
| ---------------------------------------- | ---------------------------------------- |
| ![without pooling](https://cdn-images-1.medium.com/max/2000/1*d3pDD4GW-QMW3anEECJ4uQ.png) | ![with pooling](https://cdn-images-1.medium.com/max/1600/1*7GkHhws29t93C2Cij9eKww.png) [image source](https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59) |

Another example ([image source](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)):

![pooling](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-6-11-53-pm.png)



### Fully Connected Layers

For fully connected layers we have $$ output = activation(dot(input, kernel) + bias) $$ 

Some of the key activation functions are as follows:

* **Sigmoid**
  Sigmoid function pushes large positive numbers to 1 while large negative numbers to 0. However it has two fallbacks: 1) It will kill the gradient. If the value of a neuron is either 0 or 1, the gradient for the neuron will become so closed to zero that, it will "kill" the multiplication results for all gradients in back propagation computation. 2) The sigmoid output are all positive. It will cause the gradient on weights become all positive or all negative.

  ![sigmoid](http://cs231n.github.io/assets/nn1/sigmoid.jpeg) [source: 3]

* **Tanh**
  Tanh activation is a scaled version of sigmoid function: $$tanh(x)=2σ(2x)−1tanh⁡(x)=2σ(2x)−1$$ Therefore it is zero centered with range [-1,1]. It still have the problem of killing gradient, but generally it is preferred to sigmoid activation.

* **ReLU**
  Short for Rectified Linear Units. A popular choice. It threshold upon 0.  $$ max (0, x) $$  Comparing to the previous two activation methods, it's much quicker to converge and involves much less computation time due to linearity. And it doesn't have the issue of non-zero centered. However, it should be noted, if the learning rate is set to be high, part of the neurons will "die" - they will be not activated during the whole training phase. With the learning rate set to be smaller, it won't be much an issue.

  And below is a demonstration of how ReLU activation looks like:

  ![relu](https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-6-18-19-pm.png?w=1496)			

  ​								[image source](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

* **SoftMax**
  A very common choice for multi-class output activation.

### Batch Normalization Layer
> This type of layer turns out to be useful when using neurons with unbounded activations (e.g. rectified linear neurons), because it permits the detection of high-frequency features with a big neuron response, while damping responses that are uniformly large in a local neighborhood. It is a type of regularizer that encourages "competition" for big activities among nearby groups of neurons."  - AlexNet

Batch normalization is a common practice in deep learning. In machine learning tasks, scaling with zero mean and one standard deviation will make the performance better. However, in deep learning, even if we normalize the data at the very beginning, the data distribution will change a lot in deeper layers. Therefore with batch normalization layer, we could always do data preprocessing again. It is often used right after the fully connected layer or convolutional layer, before the non-linear layers. It makes a significant difference and becomes much more robust to bad initializations. 

 Note that, Fei-Fei Li claims the contributions of batch normalization is minimal. And the use of local response normalization could hardly be seen in recent year models.

### Drop Out Layer
Drop out layer is a common choice to prevent over fitting. It's fast and effective. It will keep some neurons activated or 0 according to probabilities. 




## Sample Architecture and Codes

Sample architecture for convolutional neural network is as follows:

![CNN](http://cs231n.github.io/assets/cnn/convnet.jpeg) [source: 3]



Sample codes for MNIST solution using keras deep learning as follows:

```python
'''Transfer learning toy example:
1- Train a simple convnet on the MNIST dataset the first 5 digits [0..4].
2- Freeze convolutional layers and fine-tune dense layers
   for the classification of digits [5..9].
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_transfer_cnn.py
Get to 99.8% test accuracy after 5 epochs
for the first five digits classifier
and 99.2% for the last five digits after transfer + fine-tuning.
'''

from __future__ import print_function

import datetime
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

now = datetime.datetime.now

batch_size = 128
num_classes = 5
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
filters = 32
# size of pooling area for max pooling
pool_size = 2
# convolution kernel size
kernel_size = 3

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


def train_model(model, train, test, num_classes):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    t = now()
    model.fit(x_train, y_train,
              batch_size=batch_size, epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    print('Training time: %s' % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# create two datasets one with digits below 5 and one with 5 and above
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5

# define two groups of layers: feature (convolutions) and classification (dense)
feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]

# create complete model
model = Sequential(feature_layers + classification_layers)

# train model for 5-digit classification [0..4]
train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False

# transfer: train dense layers for new classification task [5..9]
train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)
```

[code source: keras documentation](https://github.com/fchollet/keras/blob/master/examples/mnist_transfer_cnn.py)

## Parameter Tuning

### Losses

- **Regression**: Mean_squared_error, Mean_absolute_error, mean_absolute_percentage_error, mean_squared_logarithmic_error
- **Classification**: <br> 
  two most commonly used: squared_hinge, cross entropy for softmax output
  1) hinge: hinge, squared_hinge
  2) cross entropy: categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy

### Optimizers

- **Batch Size**: It means the number of training examples in one forward-backward training phase. If the batch size is small, it requires less memory and the network trains faster (the parameters will be updated once a batch). If the batch size is large, the training takes more time but will be more accurate. 
- **Learning Rate**: If the learning rate is small, it will takes so long to reach the optimal solution; if the learning rate is large, it will stuck at some points and fail to reach optimal. So the best practice is to use a time based learning rate - it will decrease after each epoch. How? Use parameter **decay** - a common choice is 1e-2. <br> $$ lr = self.lr * (1. / (1. + self.decay * self.iterations)) $$
- **Momentum** (used in SGD optimizer): It helps accelerating convergence and avoid local optimal. A typical value is 0.9 .

## Comparison between Deep Learning Frameworks

- **Theano:** a pretty old deep learning framework written in Java. Raw Theano might not be perfect but It has many easy-to-use APIs built on top of it, such as Keras and Lasagne.
  - (+) RNN fits nicely
  - (-) Long compile time for large models
  - (-) Single GPU support
  - (-) Many Bugs on AWS


- **TensorFlow:** a newly created machine learning framework to replace Theano - but TensorFlow and Theano share some amount of the same creators so they are pretty similar. 
  - (+) Supports more than deep learning tasks - can do reinforcement learning
  - (+) Faster model compile time than Theano
  - (+) Supports multiple GPU for data and model parallelism
  - (-) Computational graph is written in Python, thus pretty slow


- **Caffe:** mainly used for visual recognition tasks. 
  - (+) Large amount of existing models
  - (+) CNN fits nicely
  - (+) Good for image processing
  - (+) Easy to tune or train models
  - (-) needs to write extra codes for GPU models
  - (-) RNN doesn't fit well so not good for text or sound applications
- **Deeplearning4J:** a deep learning library written in Java. It includes distributed version for Hadoop and Spark.
  - (+) Supports distributed parallel computing
  - (+) CNN fits nicely
  - (-) Takes 4X computation time than the other three frameworks
- **Keras:** An easy-to-use Wrapper API for Theano, TensorFlow and Deeplearning4J. It supports all the functionality that TensorFlow supports!

## Pre-trained Models

### What are Pre-trained Models?

Pre-trained models are those models that people train on a very large datasets, such as ImageNet (it has 1.2 million images and 1000 categories). We could either use it as a start point for the deep learning tasks to raise accuracy, or use them as a feature extraction tool and feed the features generated with pre-train models into other machine learning models (e.g. SVM).

### Pre-trained Models: A Comparison

Some of the pre-trained models for image tasks include: ResNet, VGG, AlexNet, GoogLeNet. <br>We use top-1-error and top-5-error to represent the accuracy on ImageNet. Top-1-error is just 1- accuracy; top-5-error measures if the true label resides in the 5-most-probable labels predicted.

| Release | **Model Name** | Top-1 Error | Top-5 Error | Images per second |
| ------- | :------------: | :---------: | :---------: | :---------------: |
| 2015    |   ResNet 50    |    24.6     |     7.7     |       396.3       |
| 2015    |   ResNet 101   |    23.4     |     7.0     |       247.3       |
| 2015    |   ResNet 152   |    23.0     |     6.7     |       172.5       |
| 2014    |     VGG 19     |    28.7     |     9.9     |       166.2       |
| 2014    |     VGG 16     |    28.5     |     9.9     |       200.2       |
| 2014    |   GoogLeNet    |    34.2     |    12.9     |       770.6       |
| 2012    |    AlexNet     |    42.6     |    19.6     |      1379.8       |

Tensorflow has a recent package [Slim](https://github.com/tensorflow/models/tree/master/slim) that implements more advanced models including Inception V4 etc. Right now the lowest top-1-error is 19.6% by Inception-Resnet-V2.

## Reference

1. [Comparing Frameworks: Deeplearning4j, Torch, Theano, TensorFlow, Caffe, Paddle, MxNet, Keras & CNTK](https://deeplearning4j.org/compare-dl4j-torch7-pylearn#comparing-frameworks-deeplearning4j-torch-theano-tensorflow-caffe-paddle-mxnet-keras--cntkhttps://deeplearning4j.org/compare-dl4j-torch7-pylearn#tensorflow)

2. [Deep Learning with Theano, Torch, Caffe, TensorFlow, and Deeplearning4J: Which One Is the Best in Speed and Accuracy?](https://www.researchgate.net/publication/302955130_Deep_Learning_with_Theano_Torch_Caffe_TensorFlow_and_Deeplearning4J_Which_One_Is_the_Best_in_Speed_and_Accuracy)

3. [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/transfer-learning/)

4. [Pretrained Models](http://www.vlfeat.org/matconvnet/pretrained/)

5. [An overview of gradient descent optimization algorithms](http://sebastianruder.com/optimizing-gradient-descent/index.html#momentum)

6. [Keras: Deep Learning library for TensorFlow and Theano](https://github.com/fchollet/keras/)

7. [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

   ​
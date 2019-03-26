---
title: Improving Deep Neural Networks
date: 2017-12-09 15:09:24
tags: 
- Machine learning
- CNN
category: 
- 时习之
- Deep Learning
description: Notes from Andrew NG's second module. Some pratical codes are added.
---

## Tuning Tips

* **Importance of hyper-parameters:**

| Most Important | Learning Rate                            |
| -------------- | ---------------------------------------- |
| 2nd            | Momentum, Mini-batch size                |
| 3rd            | Hidden Units, Number of Layers, Learning rate decay |

* **Randomly choose hyper-parameter on a log scale. Don't do grid search!**

```
r = -4 * np.random.rand()  # r [-4, 0]
theta = 10^r
```



## Mini-batch

> Small training set (m <= 2000): Use batch gradient descent
>
> Large training set (m > 2000): typical mini-batch size: `2^n`, for example, 64, 128, 256, 512

* [**Tensorflow Example**](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/5_DataManagement/build_an_image_dataset.ipynb)

```
## Cheatsheet from above link

def (image_path_list, label_list):
	imagepaths = tf.convert_to_tensor(image_path_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    # slice tensors into single instances and queue them up using threads.
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X, Y
```

```
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Start the data queue
    tf.train.start_queue_runners()

    # Training cycle
    for step in range(1, num_steps+1):
            sess.run(train_op)
```

* **Keras Example**

```
model.fit(x_train, y_train,
          epochs=2000,
          batch_size=128)
```



## Adam Optimizer

Adam optimizer combines the advantages of both Momentum and RMSprop thus enabling quick convergence. 

> Parameters: Learning rate: needs to be tuned
>
> Beta1: 0.9 (dw)
>
> Beta2: 0.999 (dw^2)
>
> Epsilon: 10^-8

* **Tensorflow Example**

```
“”“
default setting:
__init__(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam'
)
”“”

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
```

* **Keras Example**

```
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
```



## Learning Rate Decay

learning rate changes with current epoch number. Some common algorithms are as follows:

```
lr = 0.95^epoch_num*alpha
lr = k / np.sqrt(epoch_num)*alpha
lr = alpha/(1+kt)
```



* **[TensorFlow Example](https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py#L243)**

```
# Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
batch = tf.Variable(0, dtype=data_type())
learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
# Use simple momentum for the optimization.
optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)
```

* **Keras Example**

```
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.95)
```



## Initialization

The reason we want to refine initialization is to avoid vanishing and exploding gradient in deep networks. For example, let's say g(z) = z, b = 0; then in a 15-layers NN we will have ` y = (w^15) * x`; thus we will have either very large or very small gradient. One way to avoid is to take into account number of features that will feed into the current layer. Some recommended initializations include:

`np.sqrt(2/ (num_features_last_layer))`

`np.sqrt( 2/ (number_features_last_layer + number_features_this_layer))`



* **Tensorflow Example**

```
W = tf.get_variable('W', shape=(512, 256), initializer=tf.contrib.layers.xavier_initializer()) 
```

```
tf.contrib.layers.fully_connected(
    inputs,
    num_outputs,
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None
)
```

* **Keras Example**

```
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
"""
Other choices include:
keras.initializers.Zeros()
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
lecun_uniform(seed=None)
glorot_normal(seed=None) # Xavier normal initializer.
...
"""
```



## Tensorflow Tutorials

- [Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/index.html)
- [Tensorflow Python API](https://www.tensorflow.org/versions/r0.12/api_docs/python/)
- [Tensorflow Examples by aymericdamien](https://github.com/aymericdamien/TensorFlow-Examples)
- [Tensorflow Tutorials by HvassLabs](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
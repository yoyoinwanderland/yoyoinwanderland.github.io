---
title: CV Traditional Algorithms
date: 2018-02-20 20:55:37
tags: Computer Vision
category: 
- 时习之
- Computer Vision
description: Computer Vision Basics Recap
---

## Blurring

* **Purpose: to reduce noises in the image.**
* **Idea:** Basically we can think of the noises are affected by its neighbors and neighbors affected by noises so they don't look much different.

### Gaussian Filter

* **Parameters:**

  * Size of the kernel
  * Sigma: bigger sigma, larger the blur

* **Implementation:** [Reference](https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python)

  ```
  def gaussian_filter(shape =(5,5), sigma=1):
      x, y = [edge /2 for edge in shape]
      grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in xrange(-x, x+1)] for j in xrange(-y, y+1)])
      g_filter = np.exp(-grid)
      g_filter /= np.sum(g_filter)
      return g_filter
  ```

  ```
  [[ 0.00296902  0.01330621  0.02193823  0.01330621  0.00296902]
   [ 0.01330621  0.0596343   0.09832033  0.0596343   0.01330621]
   [ 0.02193823  0.09832033  0.16210282  0.09832033  0.02193823]
   [ 0.01330621  0.0596343   0.09832033  0.0596343   0.01330621]
   [ 0.00296902  0.01330621  0.02193823  0.01330621  0.00296902]]
  ```

  ![gaussian](https://i.stack.imgur.com/0HnWF.png)

  ​

* **Pros & Cons:**

  * Quick computation (function of space alone)
  * Used in conjunction with edge detection to reduce noises while finding edges
  * Not best in noise removal
  * Will blur edges too

### Median Filter

![median filter](http://opencv-python-tutroals.readthedocs.io/en/latest/_images/median.jpg)

* **Pros and Cons:**
  * Reduce pepper and salt noises



### Bilateral Filter

* **Concepts:**

   the spatial kernel for smoothing differences in coordinates:

  ![formular](http://img.blog.csdn.net/20130515234049748)

   The range kernel for smoothing differences in intensities:

  ![formular](http://img.blog.csdn.net/20130515234053466)

 The above two filters multiplied to get the final bilateral filter.

Spatial kernel enables one center value to discriminate all other values around it by distance. Range kernel enables one center value to discriminate pixels that has a big value differences. Therefore, noises will be discriminated. If center is (left) part of the edge, its value also get retained rather than averaged by pixels of the other part of the edge.


* **Pros and Cons:**
  * Highly effective at noise removal
  * Edge preserving
  * Slower compared to other filters.



## Morphological Transformations

* **Purpose: Shape manipulation/ noise reduction for binary images**

### Erosion

> Erodes away the boundaries of foreground object. A pixel in the original image (either 1 or 0) will be considered 1 only if ***all the pixels*** under the kernel is 1, otherwise it is eroded (made to zero).

![original](https://docs.opencv.org/3.0-beta/_images/j.png) ![Erosion](https://docs.opencv.org/3.0-beta/_images/erosion.png)



### Dilation

> A pixel element is ‘1’ if atleast one pixel under the kernel is ‘1’. So it increases the white region in the image or size of foreground object increases.

![original](https://docs.opencv.org/3.0-beta/_images/j.png)![](https://docs.opencv.org/3.0-beta/_images/dilation.png)



### Applications

#### Opening

> in cases like noise removal, erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won’t come back, but our object area increases.

![open](https://docs.opencv.org/3.0-beta/_images/opening.png)

#### Closing

> Closing is reverse of Opening, **Dilation followed by Erosion**. It is useful in closing small holes inside the foreground objects, or small black points on the object.

![close](https://docs.opencv.org/3.0-beta/_images/closing.png)

#### Morphological Gradient

>  It is the difference between dilation and erosion of an image. The result will look like the outline of the object.

![gradient](https://docs.opencv.org/3.0-beta/_images/gradient.png)



## Edge Detection

* **Purpose: to detect sudden changes in an image**

* **A good detection**:

  * good localization: find true positives
  * single response: minimize false positives

* **Concepts behind edge detection:** 

  >  let’s assume we have a 1D-image. An edge is shown by the “jump” in intensity in the plot below. The edge “jump” can be seen more easily if we take the first derivative (actually, here appears as a maximum). So, from the explanation above, we can deduce that a method to detect edges in an image can be performed by locating pixel locations where the gradient is higher than its neighbors (or to generalize, higher than a threshold). - [OpenCV Documentation](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html#sobel-derivatives)

  ![jump](https://docs.opencv.org/2.4/_images/Sobel_Derivatives_Tutorial_Theory_Intensity_Function.jpg) ![direvative](https://docs.opencv.org/2.4/_images/Sobel_Derivatives_Tutorial_Theory_dIntensity_Function.jpg)

* **Challenge:**

![noise](http://slideplayer.com/5324858/17/images/13/Effects+of+noise+Consider+a+single+row+or+column+of+the+image.jpg)

### Simple Edge Detection

Simple filter: `-0.5, 0, 0.5`, mean of left derivative `0,-1,1` and right derivative `-1,1,0`.



### Sobel Edge Detection

The idea is gaussian smoothing together with discrete direvative to get less noisy edges.

![sobel](https://wikimedia.org/api/rest_v1/media/math/render/svg/19a75a4374e6d56d45ca0e61f25d9134aafd8b58)

### Canny Edge Detection

* **Algorithms:**

  1. Filter out noises with Gaussian.

  2. Find magnitude and orientation of gradient

     ![gradient](https://docs.opencv.org/2.4/_images/math/4c2af1833fd9f9af4ec5506ff8a83e217ebbe6db.png)

  3. Non maximum suppression applied

     Purpose is to "thin" the edges. The approach is like this: 

     * the edge strength `G` will be compared and only the largest value remains. 
     * The comparison is against neighboring pixels *in the same directions* (for example, for y direction gradient: up and down will be compared). 

  4. Linking and thresholding

     * Gradient > Threshold_high -> edges
     * Gradient < Threshold_low   -> suppress
     * Threshold_low < Gradient < Threshold_high -> remains only attached to strong edge pixels

* **Pros and Cons:**

  * **Low error rate:** Meaning a good detection of only existent edges.
  * **Good localization:** The distance between edge pixels detected and real edge pixels have to be minimized.
  * **Minimal response:** Only one detector response per edge.



## Hough Transformation

* **Motivation:** 

  * edge detection might miss / break/  distort a line because of noises
  * Extra edge points will confuse line formation

* **Concepts:**

  1. Match **edge** points to Hough space
  2. Find the theta, d bin that has the most votes

  ```
  ## Pseudo code

  H = np.zeros((d_candidate_length, 180))
  for x,y in edge_points:
  	for theta in xrange(0, 181):
  		d = x*cos(theta) - y*sin(theta)
  		H[d, theta] += 1
  max_line = np.argmax(H)
  max_d, max_theta = H[max_line]
  ```



* **Pros and Cons**
  * Handles missing and occluded data well; multiple matches per image
  * Easy implementation
  * Computationally complex for objects with many parameters k**n (n dimensions, k bins in total)
  * looks for only one type of object



## Feature Descriptors

>  "A feature descriptor is a representation of an image or an image patch that simplifies the image by extracting useful information and throwing away extraneous information." [Learn OpenCV](https://www.learnopencv.com/histogram-of-oriented-gradients/)



### HOG - Histogram of Gradients

* **Algorithms:**

  1. Calculate gradients using sobel filter, calculate magnitude and orientation

     ```
     gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
     gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
     mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
     ```

     ![](https://www.learnopencv.com/wp-content/uploads/2016/12/hog-cell-gradients.png)

     [				Copyright belongs to Learn OpenCV](https://www.learnopencv.com/wp-content/uploads/2016/12/hog-cell-gradients.png)

  2. Calculate histogram of gradient for each window

     ![](https://www.learnopencv.com/wp-content/uploads/2016/12/hog-histogram-1.png)

     ​	![](https://www.learnopencv.com/wp-content/uploads/2016/12/histogram-8x8-cell.png)			

     ​					[Copyright belongs to Learn OpenCV](https://www.learnopencv.com/histogram-of-oriented-gradients/)

     3. Normalize and concatenate

        Normalization because of lighting conditions will affect the overall pixel values and then gradients.

        ![](https://www.learnopencv.com/wp-content/uploads/2016/12/hog-visualization.png)

        ​				[Copyright belongs to Learn OpenCV](https://www.learnopencv.com/histogram-of-oriented-gradients/)

  ​

* **Pros and Cons**

  - Edges and corners pack in a lot more information about object shape than flat regions; magnitude of gradients is large around edges and corners ( regions of abrupt intensity changes) 



### SIFT - Scale Invariant Feature Transformation

* **Pros and Cons** 

  - Detector invariant to scale and rotation
  - Robust to variations corresponding to typical viewing conditions

* **Algorithm:**

  1. Scale space

     Basically it iteratively does Gaussian blur and resizing to half. In total there will be 5 sigma and 5 blur levels.

     ![](http://aishack.in/static/img/tut/sift-octaves.jpg)

     ​				[Copyright belongs to AIShack](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-scale-space/)

  2. LoG Approximations

     Calculate Laplacian of Gaussian to retain only edges and corners.

     ![laplacian](https://docs.opencv.org/2.4/_images/Laplace_Operator_Tutorial_Result.jpg)

     So why approximation and how? Problem is it's too slow. A get around is to calculate difference of Gaussians of two consecutive scales.

     ![](http://aishack.in/static/img/tut/sift-dog-idea.jpg)

     ​				[Copyright belongs to AIShack](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-scale-space/)

     And this brings us scale invariant features (how?).

  3. Finding Keypoints 

     * Finding local maximal and minimal

       First find the maximal/ minimal inside one image; compare these local maximal/ minimals to its 26 neighbors in order to ensure it's key point.

       ![](http://aishack.in/static/img/tut/sift-maxima-idea.jpg)

       ​				[Copyright belongs to AIShack](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-scale-space/)

     * Find subpixels by math

       ![](http://aishack.in/static/img/tut/sift-maxima-subpixel.jpg)

       ​				[Copyright belongs to AIShack](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-scale-space/)

  4. Getting rid of low contrasting key points

     Low contrasting features (low gradient magnitude) are removed. Edges are removed, only corners are retained. Ideas / maths is from Harris Corner detectors.

  5. Calculate gradient of key points

     > The idea is to collect gradient directions and magnitudes around each keypoint. Then we figure out the most prominent orientation(s) in that region. And we assign this orientation(s) to the keypoint.
     >
     > Any later calculations are done **<u>RELATIVE TO this orientation</u>**. This ensures ***rotation invariance.***

     ![](http://aishack.in/static/img/tut/sift-a-keypoint.jpg)

     ​				[Copyright belongs to AIShack](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-scale-space/)

## Object Detection

### Harr Cascades

* **Algorithms**

  > Haar features shown in below image are used. They are just like our convolutional kernel. Each feature is a single value obtained by subtracting sum of pixels under white rectangle from sum of pixels under black rectangle.

  ![](https://docs.opencv.org/3.3.0/haar_features.jpg)

  > The first feature selected seems to focus on the property that the region of the eyes is often darker than the region of the nose and cheeks. The second feature selected relies on the property that the eyes are darker than the bridge of the nose. But the same windows applying on cheeks or any other place is irrelevant. 

  ![](https://docs.opencv.org/3.3.0/haar.png)

  And then apply ada boosing to train features to object labelling (positive/ negative).



## References

***Note: All the images come from opencv tutorials documentation unless specified.***

* [OpenCV Tutorial's Documentation](http://opencv-python-tutroals.readthedocs.io/en/latest/)
* [Learn OpenCV](https://www.learnopencv.com/histogram-of-oriented-gradients/)
* [Udacity - Introduction to Computer Vision](https://classroom.udacity.com/courses/ud810)
* [SIFT: Theory and Practices](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/)


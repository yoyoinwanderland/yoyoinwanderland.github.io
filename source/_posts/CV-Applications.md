---
title: Image Processing Cheatsheet from PyImageSearch 
date: 2017-10-19 10:51:50
tags: Computer Vision
category: 
- 时习之
- Computer Vision
description: Learn image processing techniques from OpenCV & Adrian's blog posts.
---

This blog summarizes image processing methods from [pyimagesearch](pyimagesearch.com). All the source codes and pictures come from the blog and I won't take any credit for anything.

## Image Processing

### References

[Blur detection with OpenCV](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2015/09/detecting_blur_header.jpg)

[OpenCV Gamma Correction](https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2015/09/gamma_correction_example.jpg)

### Codes

* **Blur Detection**

```
cv2.Laplacian(image, cv2.CV_64F).var()
```

* **Gamma Correction**

```
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
```



## Object Detection

### References

[Detecting Circles in Images using OpenCV and Hough Circles](https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/)

[Detecting Barcodes in Images with Python and OpenCV](https://www.pyimagesearch.com/2014/11/24/detecting-barcodes-images-python-opencv/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2014/11/barcode_gradient_and_detection.jpg)

[Target acquired: Finding targets in drone and quadcopter video streams using Python and OpenCV](https://www.pyimagesearch.com/2015/05/04/target-acquired-finding-targets-in-drone-and-quadcopter-video-streams-using-python-and-opencv/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2015/05/drone_acquired_02.jpg)

[Recognizing digits with OpenCV and Python](https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2017/02/digit_reco_complete.jpg)

[Detecting machine-readable zones in passport images](https://www.pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2015/11/mrz_output_04.jpg)

[Bubble sheet multiple choice scanner and test grader using OMR, Python and OpenCV](https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2016/10/omr_result_05.jpg)

### Codes

* **Detecting Circles**

```
# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 100)
 
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
```

* **Detect squares in a video**

```
# load the video
camera = cv2.VideoCapture(args["video"])
 
# keep looping
while True:
	# grab the current frame and initialize the status text
	(grabbed, frame) = camera.read()
	status = "No Targets"
 
	# check to see if we have reached the end of the
	# video
	if not grabbed:
		break
 
	# convert the frame to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 0)
	edged = cv2.Canny(blurred, 50, 150)
 
	# find contours in the edge map
	(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	
     for c in cnts:
      # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # ensure that the approximated contour is "roughly" rectangular
        if len(approx) >= 4 and len(approx) <= 6:
        # compute the bounding box of the approximated contour and
        # use the bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        aspectRatio = w / float(h)

        # compute the solidity of the original contour
        area = cv2.contourArea(c)
        hullArea = cv2.contourArea(cv2.convexHull(c))
        solidity = area / float(hullArea)

        # compute whether or not the width and height, solidity, and
        # aspect ratio of the contour falls within appropriate bounds
        keepDims = w > 25 and h > 25
        keepSolidity = solidity > 0.9
        keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

        # ensure that the contour passes all our tests
        if keepDims and keepSolidity and keepAspectRatio:
        # draw an outline around the target and update the status
        # text
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
        status = "Target(s) Acquired"
        # draw the status text on the frame
        cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 255), 2)
 
        # show the frame and record if a key is pressed
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
```

- **Detectin Texture** (Barcode in this case)

```
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth = cv2.cv.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.cv.CV_32F, dx = 0, dy = 1, ksize = -1)
 
# subtract the y-gradient from the x-gradient
# to find regions that have high horizontal and low vertical gradients.
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
# smooth out high frequency noise in the gradient 
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# construct a closing kernel and apply it to the thresholded image
# this kernel has a width that is larger than the height
# thus close the gaps between vertical stripes of the barcode
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# perform a series of erosions and dilations
# erode the white pixels in the image, thus removing the small blobs
# dilate the remaining white pixels and grow the white regions back out.
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.cv.BoxPoints(rect))

# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
```

* **Detect Digits Areas**

```
# extract the thermostat display, apply a perspective transform to it
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []
 
# loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
 
	# if the contour is sufficiently large, it must be a digit
	if w >= 15 and (h >= 30 and h <= 40):
		digitCnts.append(c)
		
# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
```

* **Detect Machine Readable Zones**

```
# initialize a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

image = cv2.imread(imagePath)
image = imutils.resize(image, height=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# smooth the image using a 3x3 Gaussian, then apply the blackhat
# morphological operator to find dark regions on a light background
gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
# extremely helpful in reducing false-positive MRZ detections
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

# apply a closing operation using the rectangular kernel to close
# gaps in between letters -- then apply Otsu's thresholding method
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# perform another closing operation, this time using the square
# kernel to close gaps between lines of the MRZ, then perform a
# series of erosions to break apart connected components
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
thresh = cv2.erode(thresh, None, iterations=4)

# during thresholding, it's possible that border pixels were
# included in the thresholding, so let's set 5% of the left and
# right borders to zero
p = int(image.shape[1] * 0.05)
thresh[:, 0:p] = 0
thresh[:, image.shape[1] - p:] = 0

# find contours in the thresholded image and sort them by their
# size
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# loop over the contours
for c in cnts:
  # compute the bounding box of the contour and use the contour to
  # compute the aspect ratio and coverage ratio of the bounding box
  # width to the width of the image
  (x, y, w, h) = cv2.boundingRect(c)
  ar = w / float(h)
  crWidth = w / float(gray.shape[1])

  # check to see if the aspect ratio and coverage width are within
  # acceptable criteria
  if ar > 5 and crWidth > 0.75:
      # pad the bounding box since we applied erosions and now need
      # to re-grow it
      pX = int((x + w) * 0.03)
      pY = int((y + h) * 0.03)
      (x, y) = (x - pX, y - pY)
      (w, h) = (w + (pX * 2), h + (pY * 2))

      # extract the ROI from the image and draw a bounding box
      # surrounding the MRZ
      roi = image[y:y + h, x:x + w].copy()
      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
      break
```



## Object Transformation

### References

[4 Point OpenCV getPerspective Transform Example](https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2014/08/getperspective_transform_01.jpg)

[How to Build a Kick-Ass Mobile Document Scanner in Just 5 Minutes](https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2014/08/receipt-scanned.jpg)

[Text skew correction with OpenCV and Python](https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/)![sample](https://www.pyimagesearch.com/wp-content/uploads/2017/02/text_skew_pos41_results.png)

[Seam carving with OpenCV, Python, and scikit-image](https://www.pyimagesearch.com/2017/01/23/seam-carving-with-opencv-python-and-scikit-image/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2017/01/seam_carving_vertical.jpg)

### Codes

* **Four Point Transformation**

```
### Find Four Points and Call Function
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
 
# show the contour (outline) of the piece of paper
print "STEP 2: Find contours of paper"
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
```

```
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped
```

* **Text Skew Correction** 

```
# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
 
# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
 
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = -(90 + angle)
 
# otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = -angle

# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
```



## Template Matching

### References

[Multi-scale Template Matching using Python and OpenCV](https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/)

[Image Difference with OpenCV and Python](https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2017/05/image_difference_output_02.jpg)

### Codes

* **Robust Template Matching**

```
"""
1. Loop over the input image at multiple scales (i.e. make the input image progressively smaller and smaller).
2. Apply template matching using cv2.matchTemplate and keep track of the match with the largest correlation coefficient (along with the x, y-coordinates of the region with the largest correlation coefficient).
3. After looping over all scales, take the region with the largest correlation coefficient and use that as your “matched” region.

While we can handle variations in translation and scaling, our approach will not be robust to changes in rotation or non-affine transformations.

If we are concerned about rotation on non-affine transformations we are better off taking the time to detect keypoints, extract local invariant descriptors, and apply keypoint matching.
"""
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]

image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found = None
 
	# loop over the scales of the image
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])
 
		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break
			
		# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the image
		edged = cv2.Canny(resized, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
 
		# check to see if the iteration should be visualized
		if args.get("visualize", False):
			# draw a bounding box around the detected region
			clone = np.dstack([edged, edged, edged])
			cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
				(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
			cv2.imshow("Visualize", clone)
			cv2.waitKey(0)
 
		# if we have found a new maximum correlation value, then ipdate
		# the bookkeeping variable
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)
 
	# unpack the bookkeeping varaible and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
 
	# draw a bounding box around the detected result and display the image
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
	cv2.imshow("Image", image)
```

* **Image Difference**

```
# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
```





## Color Manipulation


### References

[Finding the Brightest Spot in an Image using Python and OpenCV](https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2014/08/bright-area-retina-noise.jpg)

[OpenCV and Python K-Means Color Clustering](https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2014/05/jurassic-park-colors.jpg)

[Color Quantization with OpenCV using K-Means Clustering](https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/)

![sample](https://www.pyimagesearch.com/wp-content/uploads/2014/07/quant_kmeans_worldcup.jpg)

### Codes

* **Brightest color:**

```
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (args["radius"], args["radius"]), 0)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
image = orig.copy()
cv2.circle(image, maxLoc, args["radius"], (255, 0, 0), 2)
```

* **Color Quantization:** 

  Color quantization limits the number of colors remained in one picture. For example if there is sky blue and dark blue, they might be combined into some color in the middle of their RGB value. It removes redundant color information thus saves storage spaces. It's useful in image search problems.

```
# convert the image from the RGB color space to the L*a*b*
# color space -- since we will be clustering using k-means
# which is based on the euclidean distance, we'll use the
# L*a*b* color space where the euclidean distance implies
# perceptual meaning
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
 
# reshape the image into a feature vector so that k-means
# can be applied
image = image.reshape((image.shape[0] * image.shape[1], 3))
 
# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
clt = MiniBatchKMeans(n_clusters = args["clusters"])
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]
 
# reshape the feature vectors to images
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))
 
# convert from L*a*b* to RGB
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
```
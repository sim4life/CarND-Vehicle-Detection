##Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image1a]: ./examples/car_hog.jpg
[image1b]: ./examples/noncar_hog.jpg
[image2]: ./examples/HOG_example.jpg
[image2a]: ./examples/car_features.jpg
[image2b]: ./examples/noncar_features.jpg
[image3]: ./examples/hot_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/heat.jpg
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video_proc.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `get_hog_features()` lines 65 through 82 of the file called `helper_functions.py`.  This gets called by the function `single_img_features()` lines 123 through 156, which in turn gets called by the function `extract_features()` lines 160 through 179 for each image in the list of images array argument.

I started by reading in all the `vehicle` and `non-vehicle` images in the function `train_classifier()` of file `search_classify.py` lines 94 through 107.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=5`, `pixels_per_cell=(8, 8)` and `cells_per_block=(4, 4)` with `spatial_size (16, 16)` and `histogram_bins (24)`:


![alt text][image1a]
![alt text][image1b]
![alt text][image2a]
![alt text][image2b]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and input normalisation techniques. I found the above parameters to train with almost 98% test accuracy and showing good results in the video output.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the function `train_classifier()` of file `search_classify.py` lines 110 through 142 using Standard Scalar function for inputs features normalization and used randomised data for training and test sets.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In file `search_classify.py`, I performed sliding window search on all the window positions at the scales of 

	x_start = 40% of image X size
	x_end  = end of image X size
	y_start = 50% of image Y size
	y_end  = 95% of image Y size

I did this in the function `process_search_boxes()` lines 183 through 196. I assumed the car to be on the fastest lane with left-side driving to chose a region of interest for hot windows selection. I also chose a 50% overlap for sliding windows.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried, LUV, YUV and YCrCb color spaces but found that using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, provided optimal results.  Here is an example of an image with hot windows:

![alt text][image3]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result][video1]


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I applied a technique of maintaining a history of heatmaps of the past X frames and then running a threshold over the combination of their heatmaps. I performed this in the file `heatmap.py` function `process_heatmap_history()` lines 81 through 103.
This techniques helped combining vehicle bounding boxes and eliminate some false positives.

### Here is a frames and its corresponding heatmaps:

![alt text][image5]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Majority of problems were due to lower resolution of training images and low frame rate video. The high test accuracy, even after shuffling, of some of the colorspaces and parameters were also misleading as they produced a lot of false positives. My pipeline will fail in case the car drives in any lane except the left most lane. If my assumption of car driving in the left-most lane is disregarded then I can make it more robust by tweaking some of the parameters a bit more. I can also apply some image intensity normalisation or other techniques to make detections more robust.


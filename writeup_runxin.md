## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3_0]: ./output_images/sliding_windows0.png
[image3_1]: ./output_images/sliding_windows2.png
[image3_2]: ./output_images/sliding_windows3.png
[image4]: ./output_images/sliding_window.png
[image5_1]: ./output_images/bboxes_and_heat1.png
[image5_2]: ./output_images/bboxes_and_heat2.png
[image5_3]: ./output_images/bboxes_and_heat3.png
[image5_4]: ./output_images/bboxes_and_heat4.png
[image5_5]: ./output_images/bboxes_and_heat5.png
[image5_6]: ./output_images/bboxes_and_heat6.png
[image6_1]: ./output_images/labels_map.png
[image6_2]: ./output_images/labels_map2.png
[image6_3]: ./output_images/labels_map3.png
[image6_4]: ./output_images/labels_map4.png
[image6_5]: ./output_images/labels_map5.png
[image6_6]: ./output_images/labels_map6.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

The code and pipeline are implemented in IPython notebook, pipeline.ipynb.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The codes for this step is contained in the first 1-8 code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of some of the `vehicle` and `non-vehicle` classes:

![vehicles_or_non-vehicles][image1]

Where label 'car' is vehicle and 'nope' is 'non-vehicle'.  

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hog_example][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, they are reflected in the pipeline.ipynb. After serverl tests, the current parameters have the best performance.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the parameters in the cell 7, "prepare training set, extract HoG feature, label, split" and cell 16, "Train a Classifier". 
The parameters are listed in cell 7, "prepare training set, extract HoG feature, label, split".

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;).

No, I did not search by random window, it is not efficient, actually, for this project, I followed the RoI trick in Project, Advance Lane Lines, where we focused on a manually selected RoI.

Here are some examples of selected regions and windows:

[slide_window_0][image3_0]

[slide_window_1][image3_1]

[slide_window_2][image3_2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are one example image:

![slide_window][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![bboxes_and_heat1][image5_1]

![bboxes_and_heat2][image5_2]

![bboxes_and_heat3][image5_3]

![bboxes_and_heat4][image5_4]

![bboxes_and_heat5][image5_5]

![bboxes_and_heat6][image5_6]


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

![labels_map1][image6_1]

![labels_map2][image6_2]

![labels_map3][image6_3]

![labels_map4][image6_4]

![labels_map5][image6_5]

![labels_map6][image6_6]


### Here the resulting bounding boxes are drawn onto the last frame in the series:

![output_bboxes][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most issue is how to select reasonable sliding windows to balance accuracy and efficiency. Finally I used the RoI trick, where the parameters are special to these project video. 
In this case, when we extend my pipeline to other video or road case, it needs re-calibration most of sliding window parameters.
A reliable way to adjust sliding window parameters is appreciated in the future for more robust.

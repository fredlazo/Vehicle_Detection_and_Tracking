
## **Vehicle Detection Project**

### Project Files
* Extract **vehicles.zip** and **non_vehicles.zip** prior to running **detect.py**

* **detect.py**:  workflow of reading training data, extracting features and processing video

* **functions.py**:  helper functions for extracting features, creating heatmaps, drawing bounding boxes, etc.

* **project_output.mp4**:  final output video with cars detected





**The goals / steps of this project are the following:**

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the **test_video.mp4** and later implement on full **project_video.mp4**) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/1.png
[notcar]: ./output_images/image7.png
[hog_output]: ./output_images/hog_output.png
[windows96]: ./output_images/windows96.png
[cars64]: ./output_images/cars64.png
[cars160]: ./output_images/cars160.png
[heatmaps]: ./output_images/heatmaps.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I extracted HOG features using `get_hog_features()`, defined on **line 8** of **functions.py**.
HOG features are extracted during the training step within `extract_features()`, defined
at **line 48** of **functions.py** and called at **lines 78 and 86** of **detect.py**.  In the pipeline, HOG features 
are extracted in `single_img_features()`, which is defined at **line 173** of **functions.py** and called by the `search_windows()` function,
which in turn is called at **line 173** of **detect.py**.

Also included are optional color features (a coarsened and flattened representation of the image) as well as histograms of
the intensities of each channel in the image.  The function `bin_spatial()` defined at **line 29** of **functions.py**, computes the color features,
and `color_hist()`, at **line 36** of **functions.py**, computes color intensity histograms.  Both `bin_spatial()` and `color_hist()` are
called from `extract_features()` and `single_img_features()`.

The training set consists of several 64x64 images of cars of various models and colors, taken 
from different angles.  Example image below:

![car from training set][car]

It also contains several 64x64 "non-car" images, which are pictures of "non-car"
features such as lane lines, empty roadway, trees, signs, etc. that are typically encountered while driving.  
Here is an example of a "non-car" image:

![not car from training set][notcar]

The feature extraction functions allow you to try different color spaces and `skimage.hog()` parameters such as `orientations`, `pixels_per_cell`, and `cells_per_block`.  This helps in constructing an image processing pipeline.

Below is a visualization of a HOG for the above car image,
 using color channel G in RGB color space and the following HOG parameters: `orientations=8`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`:

![hog output][hog_output]

#### 2. Explain how you settled on your final choice of HOG parameters.

I experimented with different combinations of HOG parameters when using the HOG in the quizzes, based on how accurately an SVM
trained using those HOG parameters performed on a test set of data.  I transferred those parameters over and they performed
well.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using color features and HOG features extracted from the images in the "vehicles" and "non-vehicles" directories.
Training data originally consisted of 8000 vehicle images and 8000 non-vehicle images but 20% of the training was used as a test set, so the SVM only trained on 6400 vehicle images and 6400 non-vehicle images. Also, images were randomly shuffled prior to training. 

Various combinations of color features, color-histogram features, and HOG features were used during training.  I experimented
with different color spaces and tweaked the internal parameters associated with each set of features 
(number of spatial bins for color features, number of histogram bins
for color histograms, HOG parameters).  I first assessed the accuracy on the test set.  Promising configurations
were used to process individual images from the test_images directory, 
and if they worked well, further assessed on the short video **test.mp4**.

Accuracy on the test set was highest (>99%) when I used all three feature types.
Using color histograms however, seemed to make it more difficult for the pipeline to detect the black car in **test.mp4**.
Therefore, for the final video, I used only color features and HOG features.  
Accuracy of the SVC on the 
test decreased to 98% after removing the color histograms, but the SVC detected the black and white cars equally well.  

The final parameters I used can be found at
**line 53** of **detect.py**. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I searched using four different image scales (64x64, 96x96, 128x128, and 160x160 pixels), which seemed reasonable based on the 
sizes nearby, mid-distance, and faraway cars present in the test images and videos.  I scanned the images across 
a chosen y-region of interest with an overlap fraction of 0.5, which proved sufficient to enable reliable detections.
y-regions of interest are defined at **line 151** of **detect.py**. The function to create the set of windows to process for each scale
is `slide_window()`, defined at **line 109** of **functions.py**.  Within my processing pipeline, `slide_window()` is called at **line 165** of **detect.py**.


Here are windows for a window scale of 96x96:
![Windows used for the 96x96 scale][windows96]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I optimized my classifier as described in ***"3. Describe how (and identify where in your code) you trained a classifier..."*** above.

As an example of the performance of different window sizes, 
here is a test image showing car regions identified by the 64x64 sliding windows:

![Car regions identified by 64x64 windows][cars64]

Here is that same image showing the car regions identified by 160x160 sliding windows:

![Car identified by 160x160 windows][cars160]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
The final output video is [project_output.mp4](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For each frame, I created a heatmap by adding 1s to the regions of a blank grayscale image corresponding to window (of all 4 scales) 
identified as containing cars.  I added the heatmap for the current frame to a queue of the most recent several frames, 
and created a total heatmap consisting of the sum of the heatmaps for over the queue.
In the summed heatmap, intensity values below a certain threshold were zeroed.
This process of summing over a brief history and then thresholding helped remove transient false positives and strengthen
persistent detections. 
The number of frames to keep in the history queue (`n_history`) and the threshold value were tunable parameters; I ended up using a 
history of 5 frames and a threshold value of 7.

The history queue and the running total heatmap were tracked by an instance of a RunningHeatmap class,
defined at **line 295** of **functions.py**.

I then applied `scipy.ndimage.measurements.label()` to identify individual island regions of detection 
in the history-summed heatmap.  I assumed each blob corresponded to a vehicle, and constructed bounding 
boxes to cover the area of each island region.  These bounding boxes were then drawn onto the output image in
my processing pipeline (**line 208** of **detect.py**).


I experimented with a small range of values for the number of history frames to keep, and the threshold below which the history-summed
heatmap was set to zero.  I ended up using a history of 5 frames and a threshold of 7, which proved reliable for both **test.mp4**
and **project_video.mp4**.

**Top left:**  example of a heatmap from an individual frame of **test_video.mp4**.  
**Top right:**  the summed heatmap from that frame and the four previous frames.  
**Bottom left:**  output of `scipy.ndimage.measurements.label()` on the history-summed and thresholded heatmap.  
**Bottom right:**  bounding boxes of the labeled regions drawn onto the current frame.  
![Heatmaps and resulting detections][heatmaps]  
A pair of false detections can be seen in the instantaneous heatmap (top left) but these have a much smaller
relative intensity in the history-summed heatmap (top right) and are eliminated by thresholding.  Therefore,
the false-detection regions are not labeled as car regions or drawn on the output frame.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most of the parameters I used had been chosen in earlier experimentation in the quizzes, with the goal of optimizing the performance of 
the linear SVC on the test set of my data.  Trying different parameters to get good performance on the test set was a relatively 
quick process, so achieving high accuracy on the test set was not difficult.  The main difficulty arose when my SVC had trouble 
recognizing the black car in the test video.  I eventually discovered that omitting the color histogram from the features
allowed the SVC to recognize the black car more consistently, even though it reduced the SVC's accuracy on the test set.

My pipeline seems robust overall, at least on the videos provided.  In the final output, all cars are tracked, and false
detections are minimal.  However, the pipeline does briefly lose track 
of the white car at a certain distance range, then find it again (see **0:24-0:26** of **project_output.mp4**).  This could be because there
is a "sour spot" in the scales of sliding windows I chose to use, 
where the car may be too small to be identified by the 
96x96 windows but too large to be identified by the 64x64 windows.  I could probably improve this by searching a finer range 
of window scales with a greater overlap fraction.

The biggest weakness of my pipeline is its relatively high computational cost.  On my PC, it took about 40 minutes
to process the project video. For simplicity, I initially wrote the pipeline 
such that after selecting each search window, it samples to 64x64, then creates a new HOG.   A more efficient alternative
would be to compute
the HOG once for an entire image, then for each search window, take the preexisting HOG and interpolate it to a size corresponding 
to a 64x64 image, as described in the lessons.  If I had to do futher experimentation with the final project video, I would implement
this method.  However, after testing on **test_video.mp4**, my pipeline created a satisfactory output for **project_video.mp4** on the first
try.  Rewriting to use a single HOG with interpolation seems like something I could easily get wrong at first, and involve a lot
of breaking and fixing things.  I don't want to spend additional hours tinkering with my pipeline when I already have something 
that works.



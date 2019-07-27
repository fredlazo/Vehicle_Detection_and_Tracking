# For processing video files
from moviepy.editor import VideoFileClip

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.utils import shuffle
from scipy.ndimage.measurements import label
from functions import *

np.set_printoptions(threshold=np.nan)

# Read in cars and notcars datasets
images_vehicles = []
images_vehicles.append( glob.glob( 'vehicles/vehicles/GTI_Far/*.png' ) )
images_vehicles.append( glob.glob( 'vehicles/vehicles/GTI_Left/*.png' ) )
images_vehicles.append( glob.glob( 'vehicles/vehicles/GTI_MiddleClose/*.png' ) )
images_vehicles.append( glob.glob( 'vehicles/vehicles/GTI_Right/*.png' ) )
images_vehicles.append( glob.glob( 'vehicles/vehicles/KITTI_extracted/*.png' ) )

images_nonvehicles = []
images_nonvehicles.append( glob.glob( 'non-vehicles/non-vehicles/Extras/*.png' ) )
images_nonvehicles.append( glob.glob( 'non-vehicles/non-vehicles/GTI/*.png' ) )

notcars = []
cars = []
for imagelist in images_vehicles:
    for image in imagelist:
        cars.append(image)
for imagelist in images_nonvehicles:
    for image in imagelist:
        notcars.append(image)


# Shuffle the training data to reduce the possibility of choosing a 
# time-series of very similar images when selecting a subset of the data
shuffle( cars )
shuffle( notcars )
# Select a subset of the data to reduce overfitting.
# Check that the "car" and "not car" sets are the same size, which improves svm training
sample_size = 8000
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

# Code from course quiz
# Tunable hyperparameters to use when extracting features
color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # Number of orientation bins to use for histogram of oriented gradients (HOG)
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off

# To remove transient false positives from the heatmap of pixels identified as containing
# a car.
# Number of heatmaps from previous frames that are summed
# to create the current total heatmap
# e.g., if n_history = 3, the current frame's heatmap and 
# the heatmap from the 2 previous frames are summed,
# and the summed heatmap is then thresholded and used to label cars.
# Threshold below which the history-summed heatmap should 
# be zeroed to reduce false detections
n_history = 5
threshold = 7

# Extract feature vectors from training images labelled as cars.
# (see functions.py)
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
# Extract feature vectors from training data labelled as "not cars."
# (see functions.py)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column (per-feature) scaler that will
# ensure each feature has zero average and unit standard deviation
X_scaler = StandardScaler().fit(X)
# Apply the scaler to the features from the training data
scaled_X = X_scaler.transform(X)

# Create the labels vector from the training data
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear support vector classifier, as in the quizzes
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the accuracy using the test set
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single test image
t=time.time()

# Declare a running heatmap object (see functions.py)
# to store the heatmaps from the last n_history frames
# and sum them to create the total heatmap used to 
# identify car locations. This moving sum helps reduce 
# the influence of transient false positives.
#
# heat_running is global. It must be used persistently
# across invocations of the "pipeline" function below.
heat_running = RunningHeatmap( n_history )

# nframes tracks the total number of frames we have processed so far.
nframes = 0
# "pipeline" is the function that will be used to pass each  
def pipeline( input_image ):
    # nframes and heat_running are both persistent 
    # and modified between invocations of "pipeline."
    global nframes
    global heat_running
    
    print( "Processing frame {}".format( nframes ) )
    nframes += 1
    
    draw_image = np.copy(input_image)
   
    # Number of different window sizes to try during sliding window search 
    window_sizes = [64,96,128,160]

    # y-region to search for each choice of window size
    y_start_stops =[ [ np.int( input_image.shape[0]*6/11 ), input_image.shape[0]*9/11], 
    	   [ np.int( input_image.shape[0]*6/11 ), input_image.shape[0]     ], 
    	   [ np.int( input_image.shape[0]*6/11 ), input_image.shape[0]     ],
    	   [ np.int( input_image.shape[0]*6/11 ), input_image.shape[0]     ]]

    # heatmap for this frame, created as a sum of the heatmaps 
    # for each choice of window size
    heatmap = np.zeros_like( input_image[:,:,0] ).astype( np.uint8 )

    # Loop over window sizes
    for window_size,y_start_stop in zip( window_sizes, y_start_stops ):

        # Create the list of windows locations to be searched for this window size
        # (see functions.py)
        windows = slide_window(input_image, 
                               x_start_stop=[None, None], 
                               y_start_stop=y_start_stop, 
        	                   xy_window=(window_size, window_size), 
                               xy_overlap=(0.5, 0.5))

        # Find the list of windows where a car is detected        
        # (see functions.py)
        hot_windows = search_windows(input_image, windows, svc, X_scaler, 
                                     color_space=color_space, 
                                     spatial_size=spatial_size, 
                                     hist_bins=hist_bins, 
                                     orient=orient, 
                                     pix_per_cell=pix_per_cell, 
                                     cell_per_block=cell_per_block, 
                                     hog_channel=hog_channel, 
                                     spatial_feat=spatial_feat, 
                                     hist_feat=hist_feat, 
                                     hog_feat=hog_feat)                       
        
        # window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=3) 
        # hot_window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=3)                    

        # Add heatmap for this window size to the heatmap for this frame
        add_heat( heatmap, hot_windows )

    # Add the heatmap for this frame to the heat_running object's queue of
    # the last n_history frames.  This call also forces heat_running
    # to update its internal total heatmap (the sum of heatmaps for the
    # last n_history frames)
    heat_running.push( heatmap )

    # Get a copy of the current total heatmap (sum of last n_history heatmaps)
    thresholded_total_heatmap = np.copy( heat_running.get_running_heatmap() )
    # Threshold the total heatmap to reduce false detections
    thresholded_total_heatmap[thresholded_total_heatmap <= threshold] = 0
 
    # Label each unique island region of the thresholded heatmap
    # as a separate car
    labels = label( thresholded_total_heatmap ) 

    # Draw the bounding box of each labelled island region
    # on the final output image
    tracking_image = draw_labeled_bboxes( draw_image, labels )
   
    return tracking_image

# clip.iter_frames is a Python generator that loops through the frames.
# for image in clip.iter_frames():
#     single_image_pipeline( image )

# Open the input video
clip = VideoFileClip('project_video.mp4')
# Process the input video to create the output clip
output_clip = clip.fl_image( pipeline )
# Write the output clip
output_clip.write_videofile( 'project_output.mp4', audio=False)

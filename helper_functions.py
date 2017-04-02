import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
from skimage.feature import hog

def convert_color(img, cspace_conv='RGB2YCrCb'):
    if cspace_conv != 'RGB':
        if cspace_conv == 'BGR2RGB':
            converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif cspace_conv == 'RGB2HSV':
            converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace_conv == 'BGR2HSV':
            converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif cspace_conv == 'RGB2LUV':
            converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace_conv == 'BGR2LUV':
            converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif cspace_conv == 'RGB2HLS':
            converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace_conv == 'BGR2HLS':
            converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif cspace_conv == 'RGB2YUV':
            converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace_conv == 'BGR2YUV':
            converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif cspace_conv == 'RGB2YCrCb':
            converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif cspace_conv == 'BGR2YCrCb':
            converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: converted_img = np.copy(img)

    return converted_img


def save_bboxes(box_list, box_file='bbox_pickle.p'):
    bbox_pickle = box_list
    pickle.dump( bbox_pickle, open( box_file, "wb" ) )

def find_boxes_overlap(box_A, box_B):
    # box1 = (xmin1, xmax1)
    # box2 = (xmin2, xmax2)
    # overlapping1D(box1,box2) = xmax1 >= xmin2 and xmax2 >= xmin1
    #
    # box1 = (x:(xmin1,xmax1),y:(ymin1,ymax1))
    # box2 = (x:(xmin2,xmax2),y:(ymin2,ymax2))
    # overlapping2D(box1,box2) = overlapping1D(box1.x, box2.x) and
    #                        overlapping1D(box1.y, box2.y)

    print("box_A is:", box_A)
    print("box_B is:", box_B)
    box1 = {x:(box_A[0][0],box_A[1][0]),y:(box_A[1][0],box_A[1][1])}
    box2 = {x:(box_B[0][0],box_B[1][0]),y:(box_B[1][0],box_B[1][1])}
    print("box1 is:", box1)
    print("box2 is:", box2)

    # overlapping2D(box1,box2) = overlapping1D(box1.x, box2.x) and
    #                        overlapping1D(box1.y, box2.y)

# Perform normalisation on image
def normalise(image):
    return (image - image.mean()) / (image.max() - image.min())

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        trans_sqrt=True, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=trans_sqrt,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=trans_sqrt,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

def bin_spatial_chan(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def color_hist_def(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        t_sqrt=False, spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, color_space)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    trans_sqrt=t_sqrt, vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, trans_sqrt=t_sqrt, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        t_sqrt=False, spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        # Read in each one by one
        # image = mpimg.imread(file)
        # image = cv2.imread(img)
        image = mpimg.imread(img)
        # Normalise input
        image = normalise(image)

        img_features = single_img_features(image, color_space, spatial_size, hist_bins, orient, pix_per_cell, \
                                cell_per_block, hog_channel, t_sqrt, spatial_feat, hist_feat, hog_feat)
        features.append(img_features)
    # Return list of feature vectors
    return features

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

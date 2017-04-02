import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import fnmatch
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from helper_functions import *
from sklearn.model_selection import train_test_split

class Trainer():
    # color_space = 'RGB2LUV' #'RGB2YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    # orient = 9  # HOG orientations
    # pix_per_cell = 8 # HOG pixels per cell
    # cell_per_block = 2 # HOG cells per block
    # hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    # spatial_size = (32, 32) # Spatial binning dimensions
    # hist_bins = 32 #16    # Number of histogram bins
    # trans_sqrt = False
    # spatial_feat = True # Spatial features on or off
    # # spatial_feat = False
    # hist_feat = True # Histogram features on or off
    # # hist_feat = False
    # hog_feat = True # HOG features on or off


    #def __init__(self, color_space='RGB2LUV', orient=8, pix_per_cell=8, cell_per_block=2, spatial_size=16, hist_bins=32, hog_channel='ALL', trans_sqrt=False, spatial_feat=True, hist_feat=True, hog_feat=True):
    # def __init__(self, color_space, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hog_channel, trans_sqrt, spatial_feat, hist_feat, hog_feat):
    #     self.color_space = color_space #'RGB2YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #     self.orient = orient  # HOG orientations
    #     self.pix_per_cell = pix_per_cell # HOG pixels per cell
    #     self.cell_per_block = cell_per_block # HOG cells per block
    #     self.hog_channel = hog_channel # Can be 0, 1, 2, or "ALL"
    #     self.spatial_size = spatial_size # Spatial binning dimensions
    #     self.hist_bins = hist_bins #16    # Number of histogram bins
    #     self.trans_sqrt = trans_sqrt #False    # Transform square root
    #     self.spatial_feat = spatial_feat # Spatial features on or off
    #     self.hist_feat = hist_feat # Histogram features on or off
    #     self.hog_feat = hog_feat # HOG features on or off

    def __init__(self):
        color_space = 'RGB2LUV' #'RGB2YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 8  # HOG orientations
        pix_per_cell = 8 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
        spatial_size = (16, 16) # Spatial binning dimensions
        hist_bins = 32 #16    # Number of histogram bins
        trans_sqrt = False
        spatial_feat = True # Spatial features on or off
        # spatial_feat = False
        hist_feat = True # Histogram features on or off
        # hist_feat = False
        hog_feat = True # HOG features on or off

    def printVals(self):
        print("colorspace:{}, orient:{}, pix_per_cell:{}, cell_per_block:{},\n hog_channel:{}, spatial_size:{}, hist_bins:{}, trans_sqrt:{}\nspatial_feat:{}, hist_feat:{}, hog_feat:{}".
            format(self.color_space, self.orient, self.pix_per_cell, self.cell_per_block, self.hog_channel, self.spatial_size, self.hist_bins, self.trans_sqrt, self.spatial_feat, self.hist_feat, self.hog_feat))
    # def train_classifier(self, car_images, notcar_images, param_file='svc_pickle.p'):
    #
    #     car_features = extract_features(car_images, color_space=self.color_space,
    #                             spatial_size=self.spatial_size, hist_bins=self.hist_bins,
    #                             orient=self.orient, pix_per_cell=self.pix_per_cell,
    #                             cell_per_block=self.cell_per_block,
    #                             hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
    #                             hist_feat=self.hist_feat, hog_feat=self.hog_feat)
    #     notcar_features = extract_features(notcar_images, color_space=self.color_space,
    #                             spatial_size=self.spatial_size, hist_bins=self.hist_bins,
    #                             orient=self.orient, pix_per_cell=self.pix_per_cell,
    #                             cell_per_block=self.cell_per_block,
    #                             hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
    #                             hist_feat=self.hist_feat, hog_feat=self.hog_feat)
    #
    #     X = np.vstack((car_features, notcar_features)).astype(np.float64)
    #     # Fit a per-column scaler
    #     X_scaler = StandardScaler().fit(X)
    #     # Apply the scaler to X
    #     scaled_X = X_scaler.transform(X)
    #
    #     # Define the labels vector
    #     y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    #
    #
    #     # Split up data into randomized training and test sets
    #     rand_state = np.random.randint(0, 100)
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         scaled_X, y, test_size=0.2, random_state=rand_state)
    #
    #     print('Using:',self.orient,'orientations',self.pix_per_cell,
    #         'pixels per cell and', self.cell_per_block,'cells per block')
    #     print('Feature vector length:', len(X_train[0]))
    #     # Use a linear SVC
    #     svc = LinearSVC()
    #     # Check the training time for the SVC
    #     t=time.time()
    #     svc.fit(X_train, y_train)
    #     t2 = time.time()
    #     print(round(t2-t, 2), 'Seconds to train SVC...')
    #     # Check the score of the SVC
    #     print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    #     # Check the prediction time for a single sample
    #     t=time.time()
    #
    #     svc_pickle = {}
    #     svc_pickle["svc"]               = svc
    #     svc_pickle["scaler"]            = X_scaler
    #     svc_pickle["color_space"]       = self.color_space
    #     svc_pickle["orient"]            = self.orient
    #     svc_pickle["pix_per_cell"]      = self.pix_per_cell
    #     svc_pickle["cell_per_block"]    = self.cell_per_block
    #     svc_pickle["spatial_size"]      = self.spatial_size
    #     svc_pickle["hist_bins"]         = self.hist_bins
    #     svc_pickle["hog_channel"]       = self.hog_channel
    #     svc_pickle["spatial_feat"]      = self.spatial_feat
    #     svc_pickle["hist_feat"]         = self.hist_feat
    #     svc_pickle["hog_feat"]          = self.hog_feat
    #     pickle.dump( svc_pickle, open( param_file, "wb" ) )
    #
    #     return svc, X_scaler

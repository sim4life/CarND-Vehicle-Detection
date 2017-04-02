import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import fnmatch
import sys, getopt
import time
import pickle
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from helper_functions import *
from sklearn.model_selection import train_test_split
from Trainer import Trainer
from hog_subsample import find_subsample
from heatmap import draw_heatmap, process_heatmap
# from sklearn.cross_validation import train_test_split

svc = 0
X_scaler = 0
hog_trainer = Trainer()
param_file = 'output_rsc/svc_pickle.p'

def save_params(svc, X_scaler, hog_trainer, param_file='output_rsc/svc_pickle.p'):
    svc_pickle = {}
    svc_pickle["svc"]               = svc
    svc_pickle["scaler"]            = X_scaler
    svc_pickle["color_space"]       = hog_trainer.color_space
    svc_pickle["orient"]            = hog_trainer.orient
    svc_pickle["pix_per_cell"]      = hog_trainer.pix_per_cell
    svc_pickle["cell_per_block"]    = hog_trainer.cell_per_block
    svc_pickle["spatial_size"]      = hog_trainer.spatial_size
    svc_pickle["hist_bins"]         = hog_trainer.hist_bins
    svc_pickle["hog_channel"]       = hog_trainer.hog_channel
    svc_pickle["trans_sqrt"]        = hog_trainer.trans_sqrt
    svc_pickle["spatial_feat"]      = hog_trainer.spatial_feat
    svc_pickle["hist_feat"]         = hog_trainer.hist_feat
    svc_pickle["hog_feat"]          = hog_trainer.hog_feat
    pickle.dump( svc_pickle, open( param_file, "wb" ) )

def load_params(hog_trainer, param_file='output_rsc/svc_pickle.p'):
    print("\n\nload_params called\n\n")

    svc_pickle = pickle.load( open(param_file, "rb" ) )
    svc = svc_pickle["svc"]
    X_scaler = svc_pickle["scaler"]
    # hog_trainer = Trainer(
    #     color_space = svc_pickle["color_space"],
    #     orient = svc_pickle["orient"],
    #     pix_per_cell = svc_pickle["pix_per_cell"],
    #     cell_per_block = svc_pickle["cell_per_block"],
    #     spatial_size = svc_pickle["spatial_size"],
    #     hist_bins = svc_pickle["hist_bins"],
    #     hog_channel = svc_pickle["hog_channel"],
    #     trans_sqrt = svc_pickle["trans_sqrt"],
    #     spatial_feat = svc_pickle["spatial_feat"],
    #     hist_feat = svc_pickle["hist_feat"],
    #     hog_feat = svc_pickle["hog_feat"])

    # print("before load_params hog_trainer: ",hog_trainer.printVals())

    hog_trainer.color_space = svc_pickle["color_space"]
    hog_trainer.orient = svc_pickle["orient"]
    hog_trainer.pix_per_cell = svc_pickle["pix_per_cell"]
    hog_trainer.cell_per_block = svc_pickle["cell_per_block"]
    hog_trainer.spatial_size = svc_pickle["spatial_size"]
    hog_trainer.hist_bins = svc_pickle["hist_bins"]
    hog_trainer.hog_channel = svc_pickle["hog_channel"]
    hog_trainer.trans_sqrt = svc_pickle["trans_sqrt"]
    hog_trainer.spatial_feat = svc_pickle["spatial_feat"]
    hog_trainer.hist_feat = svc_pickle["hist_feat"]
    hog_trainer.hog_feat = svc_pickle["hog_feat"]

    # print("after load_params hog_trainer: ",hog_trainer.printVals())

    return svc, X_scaler, hog_trainer


def load_images(path='dataset'):
    # Read in cars and notcars
    cars = []
    notcars = []
    # images = []
    # for dirpath, dirs, files in os.walk(path):
    #     for filename in fnmatch.filter(files, '*.jpeg'):
    #         images.append(os.path.join(dirpath, filename))

    print("Reading images from :", path)

    # print("number of images read:", len(images))
    # print("images type:", type(images))
    # print("images[0] type:", type(images[0]))
    # print("images[0] dtype:", type(images[0].dtype))
    # print("images[0] shape:", type(images[0].shape))
    # for image in images:
    #     if 'image' in image or 'extra' in image:
    #         notcars.append(image)
    #     else:
    #         cars.append(image)

    cars = glob.glob(path+'/vehicles/*/*.png')
    notcars = glob.glob(path+'/non-vehicles/*/*.png')


    print("number of car images read:", len(cars))
    print("number of noncar images read:", len(notcars))
    min_size = min(len(cars), len(notcars))
    cars = cars[0:min_size]
    notcars = notcars[0:min_size]
    print("number of car images read:", len(cars))
    print("number of noncar images read:", len(notcars))

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    # sample_size = 500
    # cars = cars[0:sample_size]
    # notcars = notcars[0:sample_size]

    return cars, notcars


def train_classifier(hog_trainer, images_path='dataset', param_file='output_rsc/svc_pickle.p'):
    # # Read in cars and notcars
    cars, notcars = load_images(images_path)

    car_features = extract_features(cars, color_space=hog_trainer.color_space,
                            spatial_size=hog_trainer.spatial_size, hist_bins=hog_trainer.hist_bins,
                            orient=hog_trainer.orient, pix_per_cell=hog_trainer.pix_per_cell,
                            cell_per_block=hog_trainer.cell_per_block, hog_channel=hog_trainer.hog_channel,
                            t_sqrt=hog_trainer.trans_sqrt, spatial_feat=hog_trainer.spatial_feat,
                            hist_feat=hog_trainer.hist_feat, hog_feat=hog_trainer.hog_feat)
    notcar_features = extract_features(notcars, color_space=hog_trainer.color_space,
                            spatial_size=hog_trainer.spatial_size, hist_bins=hog_trainer.hist_bins,
                            orient=hog_trainer.orient, pix_per_cell=hog_trainer.pix_per_cell,
                            cell_per_block=hog_trainer.cell_per_block, hog_channel=hog_trainer.hog_channel,
                            t_sqrt=hog_trainer.trans_sqrt, spatial_feat=hog_trainer.spatial_feat,
                            hist_feat=hog_trainer.hist_feat, hog_feat=hog_trainer.hog_feat)


    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',hog_trainer.orient,'orientations',hog_trainer.pix_per_cell,
        'pixels per cell and', hog_trainer.cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

    save_params(svc, X_scaler, hog_trainer, param_file)

    return svc, X_scaler


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, hog_trainer):
    # print("search_windows hog_trainer: ",hog_trainer.printVals())
    # print("hog_trainer colorspace: ",hog_trainer.color_space)
    # print("svc classifier is: ",clf)
    # print("X_scaler is:", scaler)
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=hog_trainer.color_space,
                            spatial_size=hog_trainer.spatial_size, hist_bins=hog_trainer.hist_bins,
                            orient=hog_trainer.orient, pix_per_cell=hog_trainer.pix_per_cell,
                            cell_per_block=hog_trainer.cell_per_block, hog_channel=hog_trainer.hog_channel,
                            t_sqrt=hog_trainer.trans_sqrt, spatial_feat=hog_trainer.spatial_feat,
                            hist_feat=hog_trainer.hist_feat, hog_feat=hog_trainer.hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def search_draw_boxes(image, svc, X_scaler, hog_trainer, box_file='output_rsc/bbox_pickle.p'):
        # # Uncomment the following line if you extracted training
        # # data from .png images (scaled 0 to 1 by mpimg) and the
        # # image you are searching is a .jpg (scaled 0 to 255)
        # #image = image.astype(np.float32)/255
        # draw_image = np.copy(image)
        #
        # x_start_stop=[None, None]
        # y_start_stop = [None, None] # Min and max in y to search in slide_window()
        # y_start_stop[0] = int(image.shape[0] * 0.5)
        # print("x_start_stop is:", x_start_stop)
        # print("y_start_stop is:", y_start_stop)
        # windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
        #                     xy_window=(96, 96), xy_overlap=(0.5, 0.5))
        #
        # hot_windows = search_windows(image, windows, svc, X_scaler, hog_trainer)
        #
        # save_bboxes(hot_windows, box_file=box_file)
        #
        # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        hot_windows, window_img = process_search_boxes(image, svc, X_scaler, hog_trainer)

        save_bboxes(hot_windows, box_file=box_file)

        # plt.imshow(cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB))
        plt.imshow(window_img)
        plt.show()

        return hot_windows, window_img

def process_search_boxes(image, svc, X_scaler, hog_trainer):
    draw_image = np.copy(image)

    x_start_stop=[None, None]
    y_start_stop = [int(image.shape[0] * 0.5), int(image.shape[0] * 0.95)] # Min and max in y to search in slide_window()
    # y_start_stop[0] = int(image.shape[0] * 0.5)
    # y_start_stop[1] = int(image.shape[0] * 0.9)
    # print("x_start_stop is:", x_start_stop)
    # print("y_start_stop is:", y_start_stop)
    windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    # print("windows are:", windows)

    hot_windows = search_windows(image, windows, svc, X_scaler, hog_trainer)

    # print("hot_windows are:", hot_windows)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    return hot_windows, window_img


def perform_training(hog_trainer, images_path='dataset', param_file='output_rsc/svc_pickle.p'):
    ### TODO: Tweak these parameters and see how the results change.

    # cars, notcars = load_images(images_path)
    svc, X_scaler = train_classifier(hog_trainer, images_path=images_path, param_file=param_file)
    # svc, X_scaler = hog_trainer.train_classifier(cars, notcars, param_file)

    return svc, X_scaler

def run_image_pipeline(image):
    # global hog_trainer
    draw_img = np.copy(image)
    image = image.astype(np.float32)/255
    # image = normalise(image)
    # print("run_image_pipeline hog_trainer: ",hog_trainer.printVals())
    # print("type hog_trainer:", type(hog_trainer))
    # print("hog_trainer color_space:", hog_trainer.color_space)
    # print("after hog_trainer print")
    # print("svc is:", svc)
    hot_boxes, window_img = process_search_boxes(image, svc, X_scaler, hog_trainer)
    draw_img, heatmap = process_heatmap(draw_img, hot_boxes)
    # return draw_img
    return draw_img


def process_video(image_processor, param_file, in_vid, out_vid):
    global svc
    global X_scaler
    global hog_trainer
    print("\nbefore process_video hog_trainer: ",hog_trainer.printVals())
    print("before svc is:", svc)
    # Read in the saved svc, X_sclare and hog_trainer values
    svc, X_scaler, hog_trainer = load_params(hog_trainer, param_file=param_file)
    print("after svc is:", svc)
    print("after process_video hog_trainer: ",hog_trainer.printVals())
    clip = VideoFileClip(in_vid)
    video_clip = clip.fl_image(image_processor) # function expects color image
    video_clip.write_videofile(out_vid, audio=False)

def run_pipeline(pipeline, test_img, images_path='dataset', video_file='test_video.mp4', param_file='output_rsc/svc_pickle.p', box_file='output_rsc/bbox_pickle.p'):
    global hog_trainer
    box_list = []
    # image = cv2.imread(test_img)
    image = mpimg.imread(test_img)
    image = image.astype(np.float32)/255
    # image = normalise(image)
    # svc, X_scaler = 0, 0
    if 'train' in pipeline:
        svc, X_scaler = perform_training(hog_trainer, images_path=images_path, param_file=param_file)
        # image = mpimg.imread('bbox-example-image.jpg')
        # image = mpimg.imread('test_image.jpg')
    if 'search' in pipeline:
        if 'train' not in pipeline:
            svc, X_scaler, hog_trainer = load_params(hog_trainer, param_file=param_file)

        box_list, window_img = search_draw_boxes(image, svc, X_scaler, hog_trainer, box_file=box_file)
        # if 'vid' not in pipeline:
            # plt.imshow(cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB))
            # plt.imshow(window_img)
            # plt.show()

    if 'sub' in pipeline:
        if 'train' not in pipeline:
            svc, X_scaler, hog_trainer = load_params(hog_trainer, param_file=param_file)

        box_list = find_subsample(image, svc, X_scaler, hog_trainer, box_file=box_file)
    if 'heat' in pipeline:
        if 'train' not in pipeline and 'sub' not in pipeline:
            box_list = pickle.load( open( box_file, "rb" ))

        draw_img, heatmap = draw_heatmap(image, box_list)
    if 'vid' in pipeline:
        input_video_file = video_file
        output_video_file = input_video_file.split(".")[0] + '_proc.' + input_video_file.split(".")[1]
        print("output_video_file =", output_video_file)
        # test_img = cv2.imread(test_image)

        process_video(image_processor=run_image_pipeline, param_file=param_file, in_vid=input_video_file, out_vid=output_video_file)





def main(argv):

    global hog_trainer
    pickle_file     = 'output_rsc/svc_pickle.p'
    wind_box_file   = 'output_rsc/bbox_pickle.p'
    images_path     = 'dataset'
    video_file      = 'test_video.mp4'
    test_img        = 'test_images/test1.jpg'
    pipeline        = 'search'

    colorspace = 'RGB2LUV' #'BGR2YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 8
    hog_channel = 'ALL'
    trans_sqrt = False
    spat_size = 16
    hist_bins = 32

    # hog_trainer = Trainer(
    #             color_space = 'RGB2LUV', #'RGB2YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #             orient = 9,  # HOG orientations
    #             pix_per_cell = 8, # HOG pixels per cell
    #             cell_per_block = 2, # HOG cells per block
    #             hog_channel = 'ALL', # Can be 0, 1, 2, or "ALL"
    #             spatial_size = (32, 32), # Spatial binning dimensions
    #             hist_bins = 32, #16    # Number of histogram bins
    #             spatial_feat = True, # Spatial features on or off
    #             hist_feat = True, # Histogram features on or off
    #             hog_feat = True) # HOG features on or off


    try:
        opts, args = getopt.getopt(argv,"hp:d:v:i:c:o:n:t:s:b:",["pipe", "dataset", "vid_file", "test_img", "colorsp=", "orient=", "num_chan=", "t_sqrt", "spat_size=", "hist_bins="])
    except getopt.GetoptError:
        print ('search_classify.py -p <pipe> -d <dataset> -v <vid_file> -i <test_img> -c <color_type> -o <orientation> -n <number of hog channels> -t <transform_sqrt> -s <spatial size> -b <histogram bins>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('search_classify.py -p <pipe> -d <dataset> -v <vid_file> -i <test_img> -c <color_type> -o <orientation> -n <number of hog channels> -t <transform_sqrt> -s <spatial size> -b <histogram bins>')
            sys.exit()
        elif opt in ("-p", "--pipe"):
            pipeline = arg
        elif opt in ("-d", "--dataset"):
            images_path = arg
        elif opt in ("-v", "--vid_file"):
            video_file = arg
        elif opt in ("-i", "--test_img"):
            test_img = arg
        elif opt in ("-c", "--colorsp"):
            colorspace = arg
        elif opt in ("-o", "--orient"):
            orient = int(arg)
        elif opt in ("-n", "--num_chan"):
            hog_channel = arg
        elif opt in ("-t", "--trans_sqrt"):
            trans_sqrt = arg
        elif opt in ("-s", "--spat_size"):
            spat_size = int(arg)
        elif opt in ("-b", "--hist_bins"):
            hist_bins = int(arg)

    if hog_channel != 'ALL':
        hog_channel = int(hog_channel)
    if trans_sqrt in ['Y', 'y']:
        trans_sqrt = True
    else:
        trans_sqrt = False

    print ('colorspace: {},\norientation: {},\nnumber of hog channels: {},\ntransform sqrt: {}\nspatial size: {},\nhistogram bins: {}-'\
     .format(colorspace, orient, hog_channel, trans_sqrt, (spat_size, spat_size), hist_bins))

    # hog_trainer = Trainer(
    #         color_space = colorspace, #'RGB2YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #         orient = orient,  # HOG orientations
    #         pix_per_cell = 8, # HOG pixels per cell
    #         cell_per_block = 2, # HOG cells per block
    #         hog_channel = hog_channel, # Can be 0, 1, 2, or "ALL"
    #         trans_sqrt = trans_sqrt, # transform square root, True, False
    #         spatial_size = (spat_size, spat_size), # Spatial binning dimensions
    #         hist_bins = hist_bins, #16    # Number of histogram bins
    #         spatial_feat = True, # Spatial features on or off
    #         hist_feat = True, # Histogram features on or off
    #         hog_feat = True) # HOG features on or off


    hog_trainer.color_space = colorspace #'RGB2YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    hog_trainer.orient = orient  # HOG orientations
    hog_trainer.pix_per_cell = 8 # HOG pixels per cell
    hog_trainer.cell_per_block = 2 # HOG cells per block
    hog_trainer.hog_channel = hog_channel # Can be 0, 1, 2, or "ALL"
    hog_trainer.trans_sqrt = trans_sqrt # transform square root, True, False
    hog_trainer.spatial_size = (spat_size, spat_size) # Spatial binning dimensions
    hog_trainer.hist_bins = hist_bins #16    # Number of histogram bins
    hog_trainer.spatial_feat = True # Spatial features on or off
    hog_trainer.hist_feat = True # Histogram features on or off
    hog_trainer.hog_feat = True # HOG features on or off

    run_pipeline(pipeline, test_img, images_path=images_path, video_file=video_file, param_file=pickle_file, box_file=wind_box_file)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])

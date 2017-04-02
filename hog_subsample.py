import sys
import sys, getopt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from helper_functions import *
# from search_classify import load_params
from Trainer import Trainer



# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, hog_trainer):

    on_windows = []
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, cspace_conv=hog_trainer.color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // hog_trainer.pix_per_cell)-1
    nyblocks = (ch1.shape[0] // hog_trainer.pix_per_cell)-1
    nfeat_per_block = hog_trainer.orient*hog_trainer.cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // hog_trainer.pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, hog_trainer.orient, hog_trainer.pix_per_cell, hog_trainer.cell_per_block, trans_sqrt=False, feature_vec=False)
    hog2 = get_hog_features(ch2, hog_trainer.orient, hog_trainer.pix_per_cell, hog_trainer.cell_per_block, trans_sqrt=False, feature_vec=False)
    hog3 = get_hog_features(ch3, hog_trainer.orient, hog_trainer.pix_per_cell, hog_trainer.cell_per_block, trans_sqrt=False, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*hog_trainer.pix_per_cell
            ytop = ypos*hog_trainer.pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial_chan(subimg, size=hog_trainer.spatial_size)
            hist_features = color_hist_def(subimg, nbins=hog_trainer.hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)

                startx = xbox_left
                endx = xbox_left+win_draw
                starty = ytop_draw+ystart
                endy = ytop_draw+win_draw+ystart

                on_windows.append(((startx, starty), (endx, endy)))
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                cv2.rectangle(draw_img,(startx, starty),(endx,endy),(0,0,255),6)

    return draw_img, on_windows


def find_subsample(img, svc, X_scaler, hog_trainer, box_file='output_rsc/bbox_pickle.p'):

    # img = mpimg.imread('test_image.jpg')
    # img = mpimg.imread('test_image.png')
    # img = cv2.imread(test_img)
    # img = cv2.imread('test_image.png')
    print("img size", img.size)
    print("img type", type(img))
    print("img shape", img.shape)
    print("img dtype", img.dtype)
    # print(img)
    # img = mpimg.imread('bbox-example-image.jpg')

    ystart = 400
    ystop = 656
    scale = 1.5

    out_img, sub_windows = find_cars(img, ystart, ystop, scale, svc, X_scaler, hog_trainer)

    save_bboxes(sub_windows, box_file=box_file)

    plt.imshow(out_img)
    plt.show()
    return sub_windows

def main(argv):
    # image = mpimg.imread('cutouts/cutout1.jpg')
    pickle_file     = 'output_rsc/svc_pickle.p'
    wind_box_file   = 'output_rsc/bbox_pickle.p'
    test_img        = 'test_image.jpg'
# hog_trainer = Trainer(
    # # color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #     color_space = 'RGB2YCrCb', #'RGB2LUV'
    #     orient = 9,  # HOG orientations
    #     pix_per_cell = 8, # HOG pixels per cell
    #     cell_per_block = 2, # HOG cells per block
    #     hog_channel = 'ALL', # Can be 0, 1, 2, or "ALL"
    #     spatial_size = (16, 16), # Spatial binning dimensions
    #     hist_bins = 32, #16    # Number of histogram bins
    #     spatial_feat = True, # Spatial features on or off
    #     hist_feat = True, # Histogram features on or off
    #     hog_feat = True) # HOG features on or off


    # train_classifier(param_file='svc_pickle.p')
    # svc, X_scaler = train_classifier(color_space=color_space,
    #                         spatial_size=spatial_size, hist_bins=hist_bins,
    #                         orient=orient, pix_per_cell=pix_per_cell,
    #                         cell_per_block=cell_per_block,
    #                         hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                         hist_feat=hist_feat, hog_feat=hog_feat, param_file=pickle_file)

    try:
        opts, args = getopt.getopt(argv,"hi:",["test_img"])
    except getopt.GetoptError:
        print ('hog_subsample.py -i <test_img> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('hog_subsample.py -i <test_img>')
            sys.exit()
        elif opt in ("-i", "--test_img"):
            test_img = arg

    image = cv2.imread(test_img)
    svc, X_scaler, hog_trainer = load_params(param_file=pickle_file)

    find_subsample(image, svc, X_scaler, hog_trainer, box_file=wind_box_file)
    # svc, X_scaler, hog_trainer = load_pickle(param_file=pickle_file)
    #
    # # img = mpimg.imread('test_image.jpg')
    # # img = mpimg.imread('test_image.png')
    # img = cv2.imread(test_img)
    # # img = cv2.imread('test_image.png')
    # print("img size", img.size)
    # print("img type", type(img))
    # print("img shape", img.shape)
    # print("img dtype", img.dtype)
    # # print(img)
    # # img = mpimg.imread('bbox-example-image.jpg')
    #
    # ystart = 400
    # ystop = 656
    # scale = 1.5
    #
    # out_img, sub_windows = find_cars(img, ystart, ystop, scale, svc, X_scaler, hog_trainer)
    #
    # save_bboxes(sub_windows, box_file=wind_box_file)
    #
    # plt.imshow(out_img)
    # plt.show()

if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])

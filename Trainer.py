
class Trainer():
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

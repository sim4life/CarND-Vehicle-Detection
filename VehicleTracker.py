import numpy as np

class VehicleTracker():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_retain_boxes = []
        # x values of the last n fits of the line
        self.retain_boxes = []
        #array of polynomial coefficients of the last n iterations
        self.recent_nboxes = [] #np.array([0,0,0], dtype='float')
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = []
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

    def isVehicleDetected(self, boxes):
        retained_boxes = len(self.retain_boxes)
        if retained_boxes == 0:
            self.retain_boxes = boxes
            self.recent_nboxes = boxes
            self.recent_retain_boxes = boxes
        # else:

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys, getopt
import cv2
from scipy.ndimage.measurements import label

# box_list = [][]
# heatmap = np.array([0], dtype=np.float64)

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    boxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        boxes.append(bbox)
    # Return the image
    return img, boxes

def draw_heatmap(img, box_list):
    # global box_list
    # box_list = box_list_in
    # heat = np.zeros_like(img[:,:,0]).astype(np.float)
    #
    # # Add heat to each box in box list
    # heat = add_heat(heat,box_list)
    #
    # # Apply threshold to help remove false positives
    # heat = apply_threshold(heat,1)
    #
    # # Visualize the heatmap when displaying
    # heatmap = np.clip(heat, 0, 255)
    #
    # # Find final boxes from heatmap using label function
    # labels = label(heatmap)
    # draw_img = draw_labeled_bboxes(np.copy(img), labels)

    draw_img, heatmap, boxes = process_heatmap(img, box_list)
    fig = plt.figure()
    plt.subplot(121)
    # plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()

    return draw_img, heatmap, boxes

def process_heatmap(img, box_list):
    # global box_list
    # global heatmap

    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, boxes = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img, heatmap, boxes

def process_heatmap_history(img, all_bbox_list, recent_frames_used=20, threshold=5):
    # global box_list
    # global heatmap

    # Finding out valid recent_frames_used
    if len(all_bbox_list) < recent_frames_used + 1:
        recent_frames_used = len(all_bbox_list) - 1

    frame_heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Construct heatmap of history
    for bbox_list in all_bbox_list[-recent_frames_used:]:
        frame_heat = add_heat(frame_heat, bbox_list)


    # Apply threshold to help remove false positives
    frame_heat = apply_threshold(frame_heat,threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(frame_heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img, bboxes = draw_labeled_bboxes(np.copy(img), labels)

    return draw_img, heatmap, bboxes

def main(argv):
    # image = mpimg.imread('cutouts/cutout1.jpg')
    box_file='output_rsc/bbox_pickle.p'
    test_img = 'test_image.jpg'

    try:
        opts, args = getopt.getopt(argv,"hi:",["test_img"])
    except getopt.GetoptError:
        print ('heatmap.py -i <test_img> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ('heatmap.py -i <test_img>')
            sys.exit()
        elif opt in ("-i", "--test_img"):
            test_img = arg

    # Read in a pickle file with bboxes saved
    # Each item in the "all_bboxes" list will contain a
    # list of boxes for one of the images shown above
    box_list = pickle.load( open( box_file, "rb" ))

    # Read in image similar to one shown above
    # image = cv2.imread(test_img)
    image = mpimg.imread(test_img)
    image = image.astype(np.float32)/255
    # image = mpimg.imread('bbox-example-image.jpg')

    draw_heatmap(image, box_list)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv[1:])

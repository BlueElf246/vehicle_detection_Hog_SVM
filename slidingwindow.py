import cv2
import numpy as np
import pickle
from detect_vehicle import *
from draw import *
def load_classifier(pickle_file="lp_model.p"):
    dist_pickle = pickle.load( open(pickle_file, "rb" ) )
    return dist_pickle
def process_image_hog_pipeline(image, frame_ind, useHeatmap=True, thresh=4, avgBoxes=None, verbose=False,params=None, win_size=None):

	frame_ind += 1
	draw_image = np.copy(image)
	image = image.astype(np.float32) / 255

	if verbose == False:
		bboxes = detect_vehicles(image, frame_ind, thresh=thresh, useHeatmap=useHeatmap,
								 avgBoxes=avgBoxes, params=params, win_size=win_size)
		result = draw_boxes(draw_image, bboxes, thick=2)
	else:
		bboxes, hot_windows, heatmap, labels = detect_vehicles(image, frame_ind, thresh=thresh,
															   useHeatmap=useHeatmap, avgBoxes=avgBoxes, verbose=True, params=params, win_size=win_size)
		result = draw_boxes(draw_image, bboxes, thick=2)
		result = draw_debug_board(result, frame_ind, bboxes, hot_windows, heatmap, labels[0])

		# add frame_index text at the bottom of board
		xmax = 900
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(result, 'frame {:d}'.format(frame_ind), (xmax + 20, 50), font, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

	return result

def load_img(img_path):
	img= cv2.imread(img_path)
	return img
def show_img(img):
	cv2.imshow('img',img)
	cv2.waitKey(0)


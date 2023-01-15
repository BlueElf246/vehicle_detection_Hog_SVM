from find_car import *
import numpy as np
def find_cars_multiscale(img, params, scales=(0, 1, 2, 3), verbose=False, win_size= None):
	svc = params['svc']
	X_scaler = params['scaler']
	ystart_0, ystop_0, scale_0= win_size['ystart_0'], win_size['ystop_0'], win_size['scale_0']
	ystart_1, ystop_1, scale_1= win_size['ystart_1'], win_size['ystop_1'], win_size['scale_1']
	ystart_2, ystop_2, scale_2= win_size['ystart_2'], win_size['ystop_2'], win_size['scale_2']
	ystart_3, ystop_3, scale_3= win_size['ystart_3'], win_size['ystop_3'], win_size['scale_3']
	if verbose:
		print(params)
	bboxes = []
	# Scale 0
	if 0 in scales:
		boxes = find_cars(img, ystart_0, ystop_0, scale_0, svc, X_scaler, params)
		if len(boxes):
			bboxes.extend(boxes)

	# Scale 1
	if 1 in scales:
		boxes = find_cars(img, ystart_1, ystop_1, scale_1, svc, X_scaler, params)
		if len(boxes):
			bboxes.extend(boxes)

	# Scale 2
	if 2 in scales:
		boxes = find_cars(img, ystart_2, ystop_2, scale_2, svc, X_scaler, params)
		if len(boxes):
			bboxes.extend(boxes)

	#Scale 3
	if 3 in scales:
		boxes = find_cars(img, ystart_3, ystop_3, scale_3, svc, X_scaler, params)
		if len(boxes):
			bboxes.extend(boxes)
	return bboxes

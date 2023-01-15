import numpy as np
# Returns heatmap for list of bounding boxes.
def heatmap_from_detections(img, bbox_list):
	h, w, _ = img.shape
	heatmap = np.zeros((h, w)).astype(np.float32)

	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		x1, y1 = box[0]
		x2, y2 = box[1]
		heatmap[y1:y2, x1:x2] += 1
	# Return updated heatmap
	return heatmap  # Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
	# create a copy to exclude modification of input heatmap
	heatmap = np.copy(heatmap)
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	heatmap = np.clip(heatmap, 0, 255)
	# Return thresholded map
	return heatmap

def get_labeled_bboxes(labels):
    bboxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)
    # Return list of bounding boxes
    return bboxes
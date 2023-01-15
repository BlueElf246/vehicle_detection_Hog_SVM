# Draw debug board with Binarization-View, Lane-Detesction-View
import cv2
import numpy as np
def draw_debug_board(img, frame_ind, bboxes, hot_windows, heatmap, labels):
	# prepare RGB heatmap image from float32 heatmap channel
	img_heatmap = (np.copy(heatmap) / np.max(heatmap) * 255.).astype(np.uint8)
	img_heatmap = cv2.applyColorMap(img_heatmap, colormap=cv2.COLORMAP_HOT)
	img_heatmap = cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2RGB)

	# prepare RGB labels image from float32 labels channel
	img_labels = (np.copy(labels) / np.max(labels) * 255.).astype(np.uint8)
	img_labels = cv2.applyColorMap(img_labels, colormap=cv2.COLORMAP_HOT)
	img_labels = cv2.cvtColor(img_labels, cv2.COLOR_BGR2RGB)

	# draw hot_windows in the frame
	img_hot_windows = np.copy(img)
	img_hot_windows = draw_boxes(img_hot_windows, hot_windows, thick=2)

	ymax = 0

	board_x = 5
	board_y = 5
	board_ratio = (img.shape[0] - 3 * board_x) // 3 / img.shape[0]  # 0.25
	board_h = int(img.shape[0] * board_ratio)
	board_w = int(img.shape[1] * board_ratio)

	ymin = board_y
	ymax = board_h + board_y
	xmin = board_x
	xmax = board_x + board_w

	offset_x = board_x + board_w

	# draw hot_windows in the frame
	img_hot_windows = cv2.resize(img_hot_windows, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
	img[ymin:ymax, xmin:xmax, :] = img_hot_windows

	# draw heatmap in the frame
	xmin += offset_x
	xmax += offset_x
	img_heatmap = cv2.resize(img_heatmap, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
	img[ymin:ymax, xmin:xmax, :] = img_heatmap

	# draw heatmap in the frame
	xmin += offset_x
	xmax += offset_x
	img_labels = cv2.resize(img_labels, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
	img[ymin:ymax, xmin:xmax, :] = img_labels

	return img

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
from find_car_multi_scale import *
from Avg_box import *
from heat_map_filtering import *
from scipy.ndimage.measurements import label
def detect_vehicles(image, frame_idx, avgBoxes=None, thresh=1, useHeatmap=True, verbose=False,
                    params=None,win_size=None):
    scales= win_size['use_scale']
    hot_windows = find_cars_multiscale(image, params=params, scales=scales, win_size= win_size)
    if avgBoxes:
        avgBoxes.add(hot_windows)
        hot_windows = avgBoxes.all_boxes

    heatmap = []
    labels = []

    if useHeatmap:
        heatmap = heatmap_from_detections(image, hot_windows)
        heatmap_thresh = apply_threshold(heatmap, thresh)

        labels = label(heatmap_thresh)
        bboxes = get_labeled_bboxes(labels)
    else:
        bboxes = hot_windows

    if verbose == False:
        return bboxes

    return bboxes, hot_windows, heatmap, labels

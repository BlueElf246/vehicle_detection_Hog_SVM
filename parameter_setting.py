params = {}
params['color_space'] = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
params['orient'] = 9  # HOG orientations
params['pix_per_cell'] = 8  # HOG pixels per cell
params['cell_per_block'] = 2  # HOG cells per block
params['hog_channel'] = 'ALL'  # Can be 0, 1, 2, or "ALL"
params['spatial_size'] = (16, 16)  # Spatial binning dimensions
params['hist_bins'] = 16  # Number of histogram bins
params['spatial_feat'] = True  # Spatial features on or off
params['hist_feat'] = True  # Histogram features on or off
params['hog_feat'] = True  # HOG features on or off

win_size={}
win_size['ystart_0'], win_size['ystop_0'], win_size['scale_0']=350,600,0.8
win_size['ystart_1'], win_size['ystop_1'], win_size['scale_1']=350,600,1.0
win_size['ystart_2'], win_size['ystop_2'], win_size['scale_2']=350,600,1.2
win_size['ystart_3'], win_size['ystop_3'], win_size['scale_3']=350,600,1.5
win_size['use_scale']=(0,1,2,3)
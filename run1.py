from slidingwindow import *
from parameter_setting import win_size
import os
os.chdir("/Users/datle/Desktop/phantich")
params= load_classifier('lp_model.p')
image= load_img('image/test1.jpg')
result= process_image_hog_pipeline(image, 0, useHeatmap=True, thresh=4, avgBoxes=None, verbose=True,params=params, win_size=win_size)
show_img(result)
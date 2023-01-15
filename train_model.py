import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob
import os
import pickle
os.chdir("/Users/datle/Desktop/phantich/dataset1")
from hog_ultils import *
from  parameter_setting import params
def load_dataset():
    cars = glob.glob('./vehicles1/GTI_Far/*.png')
    cars += glob.glob('./vehicles1/GTI_MiddleClose/*.png')
    cars += glob.glob('./vehicles1/GTI_Left/*.png')
    cars += glob.glob('./vehicles1/GTI_Right/*.png')
    notcars = glob.glob('./non-vehicles1/GTI_Far/*.png')
    notcars += glob.glob('./non-vehicles1/GTI_MiddleClose/*.png')
    notcars += glob.glob('./non-vehicles1/GTI_Left/*.png')
    notcars += glob.glob('./non-vehicles1/GTI_Right/*.png')
    return cars, notcars
def get_feature(dataset,params):
    color_space = params['color_space']
    spatial_size = params['spatial_size']
    hist_bins = params['hist_bins']
    orient = params['orient']
    pix_per_cell = params['pix_per_cell']
    cell_per_block = params['cell_per_block']
    hog_channel = params['hog_channel']
    spatial_feat = params['spatial_feat']
    hist_feat = params['hist_feat']
    hog_feat = params['hog_feat']
    feature=[]
    for x in dataset:
        img= cv2.imread(x)
        img_feature= get_feture_of_image(img, color_space=color_space, spatial_size=spatial_size,
                                           hist_bins=hist_bins, orient=orient,
                                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                           hog_channel=hog_channel,
                                           spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        feature.append(img_feature)
    return feature
def combine_feature(lp_feature, non_lp_feature):
    X= np.vstack((lp_feature, non_lp_feature)).astype(np.float32)
    y= np.hstack((np.ones(len(lp_feature)), np.zeros(len(non_lp_feature))))
    return X,y
def Normalize_ds(X):
    sc= StandardScaler()
    return sc ,sc.fit_transform(X)
def split(X,y):
    rand_state= np.random.randint(0,100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rand_state, test_size=0.2)
    return X_train, X_test, y_train, y_test
def fit(X,y):
    svc= LinearSVC()
    svc.fit(X,y)
    return svc
def evaluate_model(svc,X_test, y_test):
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
def save_model(pickcle_file, svc, sc, params):
    print('Saving to pickle_file...')
    try:
        with open(pickcle_file, 'wb') as pfile:
            pickle.dump(
                {'svc': svc,
                 'scaler': sc,

                 'color_space': params['color_space'],
                 'orient': params['orient'],
                 'pix_per_cell': params['pix_per_cell'],
                 'cell_per_block': params['cell_per_block'],
                 'hog_channel': params['hog_channel'],
                 'spatial_size': params['spatial_size'],
                 'hist_bins': params['hist_bins'],
                 'spatial_feat': params['spatial_feat'],
                 'hist_feat': params['hist_feat'],
                 'hog_feat': params['hog_feat'],
                 'y_start_0': params['ystart_0'],
                 'ystop_0': params['ystop_0'],
                 'scale_0': params['scale_0'],
                 'y_start_1':params['ystart_1'],
                 'ystop_1':params['ystop_1'],
                 'scale_1':params['scale_1'],
                 'y_start_2': params['ystart_2'],
                 'ystop_2': params['ystop_2'],
                 'scale_2': params['scale_2'],
                 'y_start_3': params['ystart_3'],
                 'ystop_3': params['ystop_3'],
                 'scale_3': params['scale_3']
                 },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save classifier to', pickle_file, ':', e)
        raise
    print('Classifier saved in pickle file.')





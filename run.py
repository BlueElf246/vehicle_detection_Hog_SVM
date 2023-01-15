import numpy as np
from train_model import *
import os
if __name__ == '__main__':
    #load dataset
    lp, non_lp= load_dataset()
    #get feature of dataset
    lp_feature= get_feature(lp,params) # 3425, 6108
    non_lp_feature= get_feature(non_lp, params)
    # combine into dataset
    X,y =combine_feature(lp_feature,non_lp_feature) # (7325,6108)    (7325,)
    # scale dataset
    sc, X_scaled = Normalize_ds(X)
    # train_test_split
    X_train, X_test, y_train, y_test= split(X_scaled,y)
    # train_SVM
    svc= fit(X_train,y_train)
    #evualuate model
    evaluate_model(svc,X_test, y_test)
    # save SVM
    os.chdir("/Users/datle/Desktop/phantich")
    save_model('lp_model.p', svc, sc, params)

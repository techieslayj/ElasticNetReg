import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import time

def fit_EN(Train_X, Train_Y, Test_X, Test_Y, Predefined_Split):

    #  1. Fitting
    # In article,
    # - l1_ratio: 0.5
    # - alpha: log: 1e-6 -> 1e0, base 10
    Fit_EN = GridSearchCV(\
        ElasticNet(l1_ratio = 0.5), cv = Predefined_Split, \
        param_grid = {"alpha": np.logspace(-6,0,7)})
    Time0 = time.time()
    Fit_EN.fit(Train_X, Train_Y)
    Time1 = time.time()
    Time_Train = Time1 - Time0

    #  2. Prediction and Error
    Time0 = time.time()
    Predict_Y = Fit_EN.predict(Test_X)
    # debug
    #print "Predict_Y = ", Predict_Y
    Err_MAD  = mean_absolute_error(Test_Y, Predict_Y)
    Err_RMSD = mean_squared_error (Test_Y, Predict_Y)
    Time1 = time.time()
    Time_Test = Time1 - Time0

    #  3. Final return
    return (Err_MAD, Err_RMSD, Time_Train, Time_Test)

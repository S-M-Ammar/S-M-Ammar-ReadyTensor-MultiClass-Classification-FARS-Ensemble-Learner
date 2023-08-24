import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer
import joblib
from skopt import gp_minimize
from config import paths
from logger import get_logger
import os
from skopt.utils import use_named_args
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from config.paths import OPT_HPT_DIR_PATH
from sklearn.svm import SVC

np.int = np.int_


HPT_RESULTS_FILE_NAME = "HPT_results.csv"

logger = get_logger(task_name="tune")


"""
    The following fucntion used sickit-optimization module for hyper parameter tuning of models.
    We need to define the start and end range of integer or float (Real) hyper-parameters. This defined space 
    is then used in cross fold validation for each model hyper parameter tuning.
"""


def run_hyperparameter_tuning(train_X , train_Y):


    rf_01_space  = [
              Integer(3, 150, name='n_estimators'),
              Integer(1, 8, name='max_depth'),
              Integer(2,20, name = 'min_samples_split'),
              Integer(2,10, name = 'min_samples_leaf')
             ]

    

    
    rf_01 = RandomForestClassifier(max_features="log2")
    
   

    

    @use_named_args(rf_01_space)
    def objective_rf_01(**params):
        rf_01.set_params(**params)
        return -np.mean(cross_val_score(rf_01, train_X, train_Y, cv=5, n_jobs=-1,scoring="f1_macro"))

    

    res_01_gp = gp_minimize(objective_rf_01, rf_01_space, n_calls=30, random_state=42)

    

   
    best_hyperparameters = {
           
        "rf_01_n_estimators": res_01_gp.x[0],
        "rf_01_max_depth": res_01_gp.x[1],
        "rf_01_min_samples_split":res_01_gp.x[2] , 
        "rf_01_min_samples_leaf": res_01_gp.x[3] , 
        "rf_01_max_features": "log2",
    }
    
    # Making data hyper paramters directory
    if not os.path.exists(paths.OPT_HPT_DIR_PATH):
        os.makedirs(paths.OPT_HPT_DIR_PATH)
    
    joblib.dump(best_hyperparameters,OPT_HPT_DIR_PATH+"/optimized_hyper_parameters.joblib")
    return best_hyperparameters

    
    





import pandas as pd
import math
import numpy as np
from feature_engine.imputation import CategoricalImputer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from scipy.stats import pointbiserialr
from collections import OrderedDict
import scipy.stats as stats
import os
from config import paths
from preprocessing_data.preprocessing_utils import *


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns, is_training=False , missing_value_threshold_percent=30):
        self.categorical_columns =  categorical_columns
        self.categorical_imputator = None
        self.total_records = None
        self.is_training = is_training
        self.missing_value_threshold_percent = missing_value_threshold_percent
        self.categorical_columns_to_considered = []
        
    def fit(self, X, y=None):
        return self

    def transform(self, data):

        
        if(len(self.categorical_columns)==0):
            return {"data":data , "categorical_data":pd.DataFrame()}
        
        if(self.is_training):

            self.total_records = len(data)
            categorical_data = data[self.categorical_columns]
            categorical_data = cast_to_object_categorical_columns(categorical_data)
            self.categorical_columns_to_considered = get_categorical_columns_with_missing_value_threshold(categorical_data,self.categorical_columns,self.total_records,self.missing_value_threshold_percent)
            
            if(len(self.categorical_columns_to_considered) >= 1):
            
                categorical_data = categorical_data[self.categorical_columns_to_considered]
                categorical_imputator = CategoricalImputer(imputation_method='frequent')
                categorical_imputator , transformed_categorical_data = perform_categorical_imputation(categorical_imputator , categorical_data)
                save_categorical_imputer(categorical_imputator)
                transformed_categorical_data = cast_to_object_categorical_columns(transformed_categorical_data)
                
                one_hot_encoder = OneHotEncoder(drop_last=True , variables=self.categorical_columns_to_considered)
                one_hot_encoder , transformed_categorical_data = perform_one_hot_encoder(one_hot_encoder , transformed_categorical_data)
                save_one_hot_encoder(one_hot_encoder)
                save_categorical_columns_to_be_considered(self.categorical_columns_to_considered)

                return {"data":data , "categorical_data":transformed_categorical_data}
            
            return {"data":data , "categorical_data":categorical_data}
        
        
        else:
            # check if there are any categorical columns to be considered.
            self.categorical_columns_to_considered = load_categorical_columns_to_be_considered()

            if(len(self.categorical_columns_to_considered)>=1):
                categorical_data = data[self.categorical_columns_to_considered]
                categorical_data = cast_to_object_categorical_columns(categorical_data)
                categorical_imputator = load_categorical_imputor()
                categorical_data = categorical_imputator.transform(categorical_data)
                categorical_data = cast_to_object_categorical_columns(categorical_data)
                one_hot_encoder = load_one_hot_encoder()
                categorical_data = one_hot_encoder.transform(categorical_data)

                return {"data":data , "categorical_data":categorical_data}
            
            return {"data":data , "categorical_data":pd.DataFrame()}



class NumericTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_columns , is_training=False , missing_value_threshold_percent=30):
        self.numeric_columns =  numeric_columns
        self.numeric_imputator = None
        self.total_records = None
        self.is_training = is_training
        self.missing_value_threshold_percent = missing_value_threshold_percent
        self.numeric_columns_to_considered = []
        self.data = None

    def fit(self, X, y=None):
        return self

    def transform(self, data_json):  # received original data and transformed categorical data
        data = data_json['data']
        if(len(self.numeric_columns)==0):
            data_json['numeric_data'] = pd.DataFrame()
            return data_json
        
        if(self.is_training):
            self.total_records = len(data)
            numeric_data = data[self.numeric_columns]
            numeric_data = cast_to_object_numeric_columns(numeric_data)
            self.numeric_columns_to_considered = get_numeric_columns_with_missing_value_threshold(numeric_data,self.numeric_columns,self.total_records,self.missing_value_threshold_percent)

            if(len(self.numeric_columns_to_considered) >= 1):
                numeric_data = numeric_data[self.numeric_columns_to_considered]
                numeric_imputator = MeanMedianImputer(imputation_method='median')
                numeric_imputator , transformed_numeric_data = perform_numeric_imputation(numeric_imputator , numeric_data)
                save_numeric_imputor(numeric_imputator)

                min_max_scaler = MinMaxScaler()
                min_max_scaler , transformed_numeric_data = perfrom_min_max_scaling(min_max_scaler , transformed_numeric_data)
                sav_min_max_scaler(min_max_scaler)
                save_numeric_columns_to_be_considered(self.numeric_columns_to_considered)
                data_json['numeric_data'] = transformed_numeric_data
                return data_json

            data_json['numeric_data'] = pd.DataFrame()
            return data_json
    
        else:
            # check if there are any numerical columns to be considered.
            self.numeric_columns_to_considered = load_numeric_columns_to_be_considered()

            if(len(self.numeric_columns_to_considered) >= 1):
                numeric_data = data[self.numeric_columns_to_considered]
                numeric_data = cast_to_object_numeric_columns(numeric_data)
                numeric_imputator = load_numeric_imputor()
                numeric_data = numeric_imputator.transform(numeric_data)
                min_max_scaler = load_min_max_scaler()
                numeric_data = pd.DataFrame(min_max_scaler.transform(numeric_data), columns=numeric_data.columns)
                data_json['numeric_data'] = numeric_data
                return data_json
            
            data_json['numeric_data'] = pd.DataFrame()
            return data_json
            

class Merger(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.data = None

    def fit(self, X, y=None):
        return self
        

    def transform(self, data_json):  # received original data , transformed numeric data and transformed categorical data
        
        numeric_data = data_json["numeric_data"]
        categorical_data = data_json['categorical_data']

        if(len(numeric_data)==0 and len(categorical_data)==0):
            raise f"No data found for training. Try dataset with less missing values and more columns"
        
        elif(len(numeric_data)>=1 and len(categorical_data)>=1):
            categorical_data.reset_index(drop=True,inplace=True)
            numeric_data.reset_index(drop=True,inplace=True)
            columns = list(categorical_data.columns) + list(numeric_data.columns)
            processed_data = pd.concat([categorical_data,numeric_data],axis=1,ignore_index=True)
            processed_data.columns = columns
            data_json["processed_data"] = processed_data
            return data_json
        
        elif(len(numeric_data)==0):
            categorical_data.reset_index(drop=True,inplace=True)
            data_json["processed_data"] = categorical_data
            return data_json
        else:
            numeric_data.reset_index(drop=True,inplace=True)
            data_json["processed_data"] = numeric_data
            return data_json
        


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,target_field,target_classes):
        self.target_field = target_field
        self.classes_ = list(
            OrderedDict((str(c).strip(), None) for c in target_classes)
        )

        if len(self.classes_) < 2:
            raise ValueError("At least two unique classes must be provided.")

        self.class_encoding = {c: i for i, c in enumerate(self.classes_)}

    def fit(self, X, y=None):
        return self
    

    def transform(self, data):  
        if self.target_field in data.columns:
            targets = data[self.target_field].astype(str).str.strip()
            observed_classes = set(targets)

            if not observed_classes.issubset(self.classes_):
                extra_classes = observed_classes - set(self.classes_)
                raise ValueError(
                    f"Observed classes {list(extra_classes)} in the data are not among"
                    f" the allowed target classes {self.classes_}"
                )

            if len(observed_classes.intersection(self.classes_)) != len(self.classes_):
                missing_classes = set(self.classes_) - observed_classes
            transformed_targets = targets.str.strip().map(self.class_encoding)
        else:
            transformed_targets = None
        return transformed_targets



class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self):
       self.significant_columns = []

        
    def fit(self, X, y=None):
        return self
    
    def transform(self, data):  
        X_train = data['X_train']
        Y_train = data['Y_train']

        numeric_columns_to_be_considerd = load_numeric_columns_to_be_considered()
        for column in X_train.columns:
            if(column in numeric_columns_to_be_considerd):
                # perform point biserial correlation
                p = pointbiserialr(Y_train.values, X_train[column].values)
                if(p[1] <=0.05):
                    self.significant_columns.append(column) 
            else: 
                # perform chi sqaure correlation
                compare = pd.crosstab(Y_train,X_train[column])
                chi2, p, dof, ex = stats.chi2_contingency(compare)
                if(p <= 0.05):
                    self.significant_columns.append(column) 

        if(len(self.significant_columns) >=5):
            save_correlated_features(self.significant_columns)

        return self.significant_columns
       
    
class DataBalancer(BaseEstimator, TransformerMixin):
    def __init__(self):
       pass
        
    def fit(self, X, y=None):
        return self
    

    def transform(self, data):  
        X_train = data['X_train']
        Y_train = data['Y_train']
        targets = Y_train.unique()

        # compare difference
        percentage_1 = round((Y_train.tolist().count(targets[0]) / len(Y_train))*100 , 2)
        percentage_2 = round((Y_train.tolist().count(targets[1]) / len(Y_train))*100 , 2)

        difference = np.abs(percentage_1-percentage_2)

        if(difference > 25):
            # perform SMOTE
            print("Balancing dataset .....")
            oversample = SMOTE()
            X_train, Y_train = oversample.fit_resample(X_train, Y_train)

            return X_train ,Y_train
        
        else:
            return X_train , Y_train



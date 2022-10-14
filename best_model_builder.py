####################################################################################################################################################
# Author: Kim Ju Hyeong, Gachon Univ, AI-SW dept. Year: 2022                                                                                       #        
# Best model builder for autonomous combining of various encoding, scaling, modeling methods and parameters as extendsion of Scikit-learn libraies #
####################################################################################################################################################

#requirements
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
import itertools
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

#list of default split size, scaler, encoder methods
split_size_list_default = [0.7, 0.8, 0.9]
scaler_list_default = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), Normalizer()]
encoder_list_default = [LabelEncoder(), OneHotEncoder(), OrdinalEncoder()]

#dictionary of default model parameters
#key: name of each model
model_params_default = {
    "DecisionTreeClassifier(Entropy)" : {
        "criterion" : ["entropy"],
        "min_samples_split" : [3,4,5],
        "max_depth" : [4,5],
        "random_state" : [0]
    },

    "DecisionTreeClassifier(Gini)" : {
        "criterion" : ["gini"],
        "min_samples_split" : [3,4,5],
        "max_depth" : [4,5],
        "random_state" : [0]
    },

    "LogisticRegression" : {
        "C" : [0.01, 0.02, 0.03],
        "dual" : [False],
        "random_state" : [0]
    },

    "SupportVectorMachine" : {
        "C" : [0.01, 0.02, 0.03],
        "random_state" : [0]
    }
}

models = {
        "DecisionTreeClassifier(Entropy)" : DecisionTreeClassifier,
        "DecisionTreeClassifier(Gini)" : DecisionTreeClassifier,
        "LogisticRegression" : LogisticRegression, 
        "SupportVectorMachine" : SVC
    }

#returns X_train, X_test, y_train, y_test after scaling, encoding
def scale_encode_split(X, y, split_size, encoder, scaler):
    scaler.fit_transform(X)
    
    #get categorical data
    """encode categorical data with encoder"""

    y = y.map({2:0, 4:1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_size, random_state=0)

    return X_train, X_test, y_train, y_test

def build_model(K, model_key, X_train, y_train, X_test, y_test, model_params, split_size, method, scaler, encoder):
    
    model = models[model_key]
    grid = GridSearchCV(model(), param_grid=model_params , cv=K, scoring=method)
    grid.fit(X_train, y_train)

    return model_key, K, grid.best_params_, split_size, grid.best_score_, scaler, encoder


def best_combination(N, X, y, **kwargs):
    #N: number of best N combinations to get returned
    #following variables are parameters
    #arguments of testing can be customed by user
    get_distinct_values=kwargs.get("get_distinct_values",False) #whether user gets distinct results, default=False
    K_range=kwargs.get('K_range', 5) #inteager in range(2, K_range) as K parameter of K-Fold validation, default: length of X
    split_size_list=kwargs.get('split_size_list', split_size_list_default)
    scaler_list=kwargs.get('scaler_list', scaler_list_default)
    encoder_list=kwargs.get('encoder_list', encoder_list_default)
    model_params=kwargs.get('model_param',model_params_default) #model paramters dictionary, default: model_params_default 
    method=kwargs.get('method', 'accuracy') #model evaluation method: [accuracy, recall, precision, f1], default: accuracy
    
    #dataframe to store results
    df_result = pd.DataFrame([], columns=["model", "K", "model_params", "split_size", method, "scaler", "encoder"])

    for K in (2, K_range): #K value
        for split_size in split_size_list: #split size
            for scaler in scaler_list: #scalers
                for encoder in encoder_list: #encoders
                    X_train, X_test, y_train, y_test = scale_encode_split(X=X, y=y, scaler=scaler, encoder=encoder, split_size=split_size)
                    #tries each models
                    for model_name in models.keys(): 
                        #row_df => row to concat in final result dataframe
                        row_df = pd.DataFrame(build_model(K=K, model_key=model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, model_params=model_params[model_name], split_size=split_size, method=method, scaler=scaler, encoder=encoder)).T
                        row_df.columns = df_result.columns
                        df_result = pd.concat([df_result, row_df],axis=0, ignore_index=True)
    
    #sort score to get best N combinations
    df_result = df_result.sort_values(by=[method], ascending=False)

    if (get_distinct_values):
        #consider only one combination for one score
        df_result = df_result.drop_duplicates(subset=[method])

    return df_result[:N]
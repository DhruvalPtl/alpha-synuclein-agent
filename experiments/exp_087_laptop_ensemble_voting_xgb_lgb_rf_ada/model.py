
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import ADASYN

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Oversample minority classes
    ada = ADASYN(random_state=42, n_neighbors=3)
    X_res, y_res = ada.fit_resample(X_train, y_train)
    
    # Define models
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
    lgb = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, n_jobs=-1, verbose=-1)
    rf = RandomForestClassifier(n_estimators=500, max_depth=10, n_jobs=-1)
    
    # Voting ensemble
    clf = VotingClassifier(estimators=[('xgb', xgb), ('lgb', lgb), ('rf', rf)], voting='soft')
    clf.fit(X_res, y_res)
    return clf

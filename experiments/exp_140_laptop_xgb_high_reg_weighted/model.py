
from xgboost import XGBClassifier
import numpy as np

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Compute class weights manually based on inverse frequency
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights_dict = {k: v for k, v in zip(unique, max(counts)/counts)}
    sample_weights = np.array([class_weights_dict[y] for y in y_train])
    
    # Highly regularized XGBoost
    clf = XGBClassifier(
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.5,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=1.0,
        objective='multi:softprob',
        random_state=42
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    return clf

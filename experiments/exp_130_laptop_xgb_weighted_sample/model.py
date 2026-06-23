
import numpy as np
from xgboost import XGBClassifier

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Map the class_weights dictionary to an array for sample_weight
    sample_weights = np.array([class_weights[y] for y in y_train])
    
    clf = XGBClassifier(
        n_estimators=800,
        learning_rate=0.015,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    return clf

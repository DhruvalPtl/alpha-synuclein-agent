
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
import numpy as np

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Calculate scale_pos_weight based on class weights
    # Assuming class_weights is {0: w0, 1: w1, ...}
    # For multiclass, XGBoost doesn't use scale_pos_weight easily,
    # but we can use sample weights during fit.
    
    sample_weights = np.array([class_weights[y] for y in y_train])
    
    pipeline = Pipeline([
        ('sampling', ADASYN(random_state=42)),
        ('clf', XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train, clf__sample_weight=sample_weights)
    return pipeline

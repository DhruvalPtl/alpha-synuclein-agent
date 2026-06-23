
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    import numpy as np

    # Calculate scale_pos_weight for XGBoost
    # Using class_weights provided by the harness
    # Assuming class labels 0, 1, 2, 3
    # For multiclass, we can use the weights to calculate an effective multiplier
    # or use sample weights in fit.
    
    # SMOTE is effective for this level of imbalance.
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

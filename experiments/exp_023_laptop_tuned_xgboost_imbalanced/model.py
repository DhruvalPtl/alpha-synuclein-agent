
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from xgboost import XGBClassifier
    import numpy as np
    
    # Calculate sample weights for XGBoost
    weights = np.array([class_weights[y] for y in y_train])
    
    clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42
    )
    clf.fit(X_train, y_train, sample_weight=weights)
    return clf

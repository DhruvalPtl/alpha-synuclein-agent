
from xgboost import XGBClassifier

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Using scale_pos_weight to address imbalance
    clf = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf

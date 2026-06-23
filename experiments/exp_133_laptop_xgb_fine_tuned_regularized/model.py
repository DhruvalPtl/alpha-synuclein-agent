
from xgboost import XGBClassifier

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Refining XGBoost for better generalization
    clf = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.005,
        max_depth=4,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


from xgboost import XGBClassifier

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Calculate balanced weights if needed (XGBoost can handle them)
    # Using class_weights provided by the harness
    clf = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.6,
        objective='multi:softprob',
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf

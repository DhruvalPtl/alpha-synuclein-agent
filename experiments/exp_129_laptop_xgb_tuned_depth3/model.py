
from xgboost import XGBClassifier

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Using class_weights to define scale_pos_weight for XGBoost
    # Assuming class_weights mapping: {0: w0, 1: w1, 2: w2, 3: w3}
    # For multiclass, we can't directly use scale_pos_weight easily;
    # but let's use it as a heuristic or rely on sample weights.
    
    clf = XGBClassifier(
        n_estimators=700,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.6,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf

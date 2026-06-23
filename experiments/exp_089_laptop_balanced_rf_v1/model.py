
from imblearn.ensemble import BalancedRandomForestClassifier

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    clf = BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        sampling_strategy='all',
        replacement=True,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.pipeline import Pipeline

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # BalancedRandomForestClassifier performs undersampling internally,
    # which is efficient and handles the class imbalance effectively.
    clf = BalancedRandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        min_samples_split=2,
        sampling_strategy='all',
        replacement=True,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

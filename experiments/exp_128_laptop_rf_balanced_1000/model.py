
from sklearn.ensemble import RandomForestClassifier

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    clf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

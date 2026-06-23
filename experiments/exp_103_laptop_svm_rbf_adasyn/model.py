
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('sampling', ADASYN(random_state=42)),
        ('clf', SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

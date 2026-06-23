
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    pipeline = Pipeline([
        ('resample', ADASYN(random_state=42)),
        ('clf', LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

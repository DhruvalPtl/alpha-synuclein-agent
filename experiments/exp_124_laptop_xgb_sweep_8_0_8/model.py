
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    pipeline = Pipeline([
        ('sampling', ADASYN(random_state=42)),
        ('clf', XGBClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

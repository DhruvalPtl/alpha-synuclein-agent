
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    pipeline = Pipeline([
        ('sampling', SMOTE(random_state=42)),
        ('clf', XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

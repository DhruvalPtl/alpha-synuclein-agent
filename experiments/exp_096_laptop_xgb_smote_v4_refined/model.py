
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    pipeline = Pipeline([
        ('sampling', SMOTE(random_state=42)),
        ('clf', XGBClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.6,
            gamma=0.2,
            reg_lambda=1.0,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

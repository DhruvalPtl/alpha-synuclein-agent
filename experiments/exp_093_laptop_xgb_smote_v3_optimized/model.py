
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    pipeline = Pipeline([
        ('sampling', SMOTE(sampling_strategy='auto', random_state=42)),
        ('clf', XGBClassifier(
            n_estimators=1000, 
            learning_rate=0.015, 
            max_depth=7, 
            subsample=0.7, 
            colsample_bytree=0.7, 
            reg_alpha=0.1,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

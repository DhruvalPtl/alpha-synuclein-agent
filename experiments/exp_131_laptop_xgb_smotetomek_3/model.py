
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    # Using SMOTETomek to oversample and clean noise
    pipeline = Pipeline([
        ('resample', SMOTETomek(random_state=42)),
        ('clf', XGBClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.6,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

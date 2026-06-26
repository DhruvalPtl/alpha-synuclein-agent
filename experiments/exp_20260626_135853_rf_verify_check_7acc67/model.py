
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class SklearnWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, df):
        X = df[["concentration"]].values
        return self.model.predict(X)

def build_and_train(df_train, df_val, class_weights):
    X_train = df_train[["concentration"]].values
    y_train = df_train["label_int"].values
    
    # Simple model
    clf = RandomForestClassifier(
        n_estimators=10,
        class_weight=class_weights,
        random_state=42
    )
    clf.fit(X_train, y_train)
    return SklearnWrapper(clf)

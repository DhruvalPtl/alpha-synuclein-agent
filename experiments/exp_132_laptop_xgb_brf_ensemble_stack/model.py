
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import VotingClassifier

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42
    )
    brf = BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        random_state=42,
        sampling_strategy='all'
    )
    
    # Voting ensemble to balance variance
    clf = VotingClassifier(
        estimators=[('xgb', xgb), ('brf', brf)],
        voting='soft'
    )
    
    clf.fit(X_train, y_train)
    return clf

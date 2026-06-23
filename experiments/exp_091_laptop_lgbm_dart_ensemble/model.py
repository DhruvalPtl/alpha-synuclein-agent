
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    lgb1 = LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, boosting_type='dart', class_weight='balanced', n_jobs=-1, verbose=-1)
    lgb2 = LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=15, boosting_type='dart', class_weight='balanced', n_jobs=-1, verbose=-1)
    
    clf = VotingClassifier(estimators=[('lgb1', lgb1), ('lgb2', lgb2)], voting='soft')
    clf.fit(X_train, y_train)
    return clf

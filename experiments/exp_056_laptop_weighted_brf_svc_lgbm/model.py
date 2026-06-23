
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import VotingClassifier
    from imblearn.ensemble import BalancedRandomForestClassifier
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    
    # Use class_weights directly where supported, 
    # and BalancedRandomForest for handling imbalance at bagging level
    model = VotingClassifier(
        estimators=[
            ('brf', BalancedRandomForestClassifier(n_estimators=500, random_state=42)),
            ('svc', SVC(kernel='rbf', probability=True, C=1.0, class_weight='balanced', random_state=42)),
            ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight='balanced', random_state=42))
        ],
        voting='soft'
    )
    model.fit(X_train, y_train)
    return model

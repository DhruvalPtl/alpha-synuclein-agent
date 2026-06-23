
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    from sklearn.ensemble import VotingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Simple ensemble of MLP and RF
    clf1 = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ensemble', VotingClassifier(estimators=[('mlp', clf1), ('rf', clf2)], voting='soft'))
    ])
    
    model.fit(X_train, y_train)
    return model


def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.utils.class_weight import compute_sample_weight
    
    class AttentionMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super().__init__()
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.attn = nn.Linear(hidden_dim, 1)
            self.fc = nn.Linear(hidden_dim, num_classes)
            
        def forward(self, x):
            h = torch.relu(self.hidden(x))
            weights = torch.softmax(self.attn(h), dim=0)
            context = torch.sum(h * weights, dim=0)
            return self.fc(context)

    # Simple training harness
    X_train_t = torch.tensor(X_train.astype('float32'))
    y_train_t = torch.tensor(y_train.astype('int64'))
    
    model = AttentionMLP(X_train.shape[1], 64, 4)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(50):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
    class ModelWrapper:
        def __init__(self, model): self.model = model
        def predict(self, X):
            with torch.no_grad():
                return self.model(torch.tensor(X.astype('float32'))).argmax(dim=1).numpy()
                
    return ModelWrapper(model)

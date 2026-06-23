
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_classes)
            )
        def forward(self, x):
            return self.net(x)

    X_train_t = torch.tensor(X_train.astype('float32'))
    y_train_t = torch.tensor(y_train.astype('int64'))
    
    model = SimpleMLP(X_train.shape[1], 128, 4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Simple mini-batch training
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for _ in range(100):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        
    class ModelWrapper:
        def __init__(self, model): self.model = model
        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                probs = self.model(torch.tensor(X.astype('float32')))
                return probs.argmax(dim=1).numpy()
                
    return ModelWrapper(model)

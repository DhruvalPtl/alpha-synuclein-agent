
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    
    # Simple MLP with feature embedding simulation
    class ModernTabular(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        def forward(self, x): return self.net(x)
    
    # Quick training setup
    X_train_t = torch.tensor(X_train.astype('float32'))
    y_train_t = torch.tensor(y_train.astype('long'))
    model = ModernTabular(X_train.shape[1], 4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(50):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
    
    class Predictor:
        def __init__(self, model): self.model = model
        def predict(self, X):
            with torch.no_grad():
                return self.model(torch.tensor(X.astype('float32'))).argmax(dim=1).numpy()
    return Predictor(model)

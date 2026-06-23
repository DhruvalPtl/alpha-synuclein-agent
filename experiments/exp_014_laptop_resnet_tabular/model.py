
def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    class ResidualBlock(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(size, size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(size, size),
                nn.ReLU()
            )
        def forward(self, x):
            return x + self.net(x)

    class ResNetTabular(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.res1 = ResidualBlock(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.res1(x)
            return self.fc2(x)

    X_train_t = torch.tensor(X_train.astype('float32'))
    y_train_t = torch.tensor(y_train.astype('int64'))
    model = ResNetTabular(X_train.shape[1], 64, 4)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    for _ in range(150):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
    class ModelWrapper:
        def __init__(self, model): self.model = model
        def predict(self, X):
            self.model.eval()
            with torch.no_grad():
                return self.model(torch.tensor(X.astype('float32'))).argmax(dim=1).numpy()
    return ModelWrapper(model)


def build_and_train(X_train, y_train, X_val, y_val, class_weights):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    
    # Simple MLP
    class ModernTabular(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        def forward(self, x): return self.net(x)
    
    X_train_t = torch.tensor(X_train.astype('float32'))
    y_train_t = torch.tensor(y_train.astype('int64'))
    
    weights = torch.tensor([class_weights[i] for i in range(4)], dtype=torch.float32)
    
    model = ModernTabular(X_train.shape[1], 4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    for _ in range(100):
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

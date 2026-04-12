import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

logits = torch.randn(10, 3)

targets = torch.randint(low=0, high=3, size=(10,))

loss = criterion(logits, targets)

print("loss item:", loss.item())
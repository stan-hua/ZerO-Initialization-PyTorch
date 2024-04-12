# ZerO-Initialization-PyTorch
A PyTorch implementation of ZerO initialization (for Linear and Conv2d layers)

Original Paper: [https://arxiv.org/abs/2110.12661](https://arxiv.org/abs/2110.12661)


## Usage
```
# Import
from zero_init import ZerO_Init

# Define your model
model = ...

# Apply ZerO Initialization
# NOTE: Only initializes torch.nn.Linear and torch.nn.Conv2d layers
model.apply(ZerO_Init)
```
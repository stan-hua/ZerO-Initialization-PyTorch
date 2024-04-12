# ZerO-Initialization-PyTorch
A PyTorch implementation of ZerO initialization (for Linear and Conv2d layers)

Original Paper: [https://arxiv.org/abs/2110.12661](https://arxiv.org/abs/2110.12661)


## Installation
1. Install through `pip`
```
# Command-line
pip install zero_init
```

2. Clone the git repository
```
git clone https://github.com/stan-hua/ZerO-Initialization-PyTorch.git

# Temporarily add package to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}/ZerO-Initialization-PyTorch"
```

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

# GRAPH-SIM-LEARNING

This is an advanced PyTorch implementation exploring attention mechanism for graph similarity learning. We call our unified graph similarity learning framework, **N**ode-wise **A**ttention guided **G**raph **S**imilarity **L**earning, or **GRAPH-SIM-LEARNING**, which includes the following components:

i) a hybrid of graph convolution and graph self-attention for node embedding learning,

ii) a cross-attention (GCA) module for graph interaction modeling,

iii) similarity self-attention (SSA) module for graph similarity matrix fusion and alignment, and

iv) graph similarity structure learning for predicting the similarity score.

## Requirements

* python==3.8
* pytorch==1.10.2
* torch_geometric==1.10
* tqdm
* scipy
* texttable

## Run the Project

To run the project, follow the instructions below:
```
cd src
python main.py --dataset=LINUX
```
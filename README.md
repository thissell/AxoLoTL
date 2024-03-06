<p align="center">

<img src="brand/axolotl.png" height="128px" width="128px"/>
</p>

# AxoLoTL: Axiomatic Language-Tree Learning

Authors:
- Jackson Thissell (jackson@thissell.me) via **AIIRLab**
- Behrooz Mansouri (behrooz.mansouri@maine.edu) via  **AIIRLab**

---

## Introduction

AxoLoTL is an attempt at encoding representations of learned, generalizable, logic through examples of entailment 
between syntax trees. 

Other works which try to train logical understanding into LLMs often use Bidirectional Transformer models on 
sequential data, with entailment properties of the logic provided beforehand. This has seen highly limited results. 
In all circumstances, the LLM can not generalize to problem sizes above that which was seen in the training data.

AxoLoTL attempts to solve this by leveraging the graph topology and spectral features of
the problem's syntax tree. By using graph convolution and cross-talk attention between the shape (encoded as the graph's 
Laplacian Eigendecomposition) and node features of the syntax tree, AxoLoTL lets the shape and information of the problem 
not only speak to eachother, and also be malleable across encoding layers.

---

## Getting Started
### Installation

```sh
pip install requirements.txt
```
**NOTE**: AxoLoTL was developed on a machine running CUDA 11.8 and Python 3.11. The requirements file reflects this, looking for 
DGL (Deep Graph Library) and PyTorch releases based upon that version. If an error occurs during requirement installation, 
please use DGL and PyTorch's installation guide to download the versions for your computer.

### Running AxoLoTL
```shell
python run_axolotl.py
```
This script within the base directory currently does a single run-through of the AxoLoTL 
pretraining model on dummy data.

### Generating and Viewing Data

The AxoLogic dataset consists of 11k syntax trees with a classification and entailment node as starting elements. These
nodes have analogous properties to the classificiation and separation tokens in a BERT model.
In each dataset, a node has been replaced in specifications of the BERT model as well.

Within `/dataset/generated/` lives binary files for the DGL graphs and label data within the corresponding `.csv` 
files, including entailment labeled and (masked index, masked token) pairs.

To generate your own AxoLogic dataset, use the `/dataset/data_generator.py` script. To view the data in the generated
folder, use the `/dataset/data_explorer.py` script.

---

## Architecture

AxoLoTL is built upon layers of a modified Specformer model, introducing cross-talk attention heads between the Laplacian
eigen-decomposition and node features of the problem syntax tree.

Listening to a drum tells you information about its shape. The same is true for a graph and its associated Laplacian 
eigen-decomposition (LED). The generic Specformer model performs attention on the LED to find a convolutional basis, 
which is then used for graph convolution on the node features.

A layer of AxoLoTL modifies this by concatenating the LED and node features together before performing
attention. Two linear layers separate the concatenated attention back into the two original domains, and then the two
pieces are fed through dual feed-forward residual layers like a common Transformer. This is a process we are calling **cross-talk**. 
Graph convolution is then performed as usual with residual connections.

---

## Methodology

There exist four datasets within AxoLogic: Training, Testing, Validation and Generalization.

With pretrained Bidirectional Encoder models for logic, the model will get high marks for training, 
testing and validation if they all come from the same sample space. That is, if the size of the problem and the number
of free variables within the problem are equivariant between the different data sets. 

The models then fail miserably when tasked with any problem slightly larger than what was seen in the training test. 
Our goal is to create a model which can generalize learned logic to larger problems, solving this issue.
Therefore, we include a generalization set with the following parameters, with problem size given in terms of syntax tree height:

|   Dataset Name | Problem Size Range | Free Variable Count Range |
|---------------:|:------------------:|:-------------------------:|
|        Testing |     1 to 8 STH     |        0 to 5 FVC         |
|       Training |     1 to 8 STH     |        0 to 5 FVC         |
|     Validation |     1 to 8 STH     |        0 to 5 FVC         |
| Generalization |    1 to 11 STH     |        0 to 7 FVC         |

Our baseline model will be a BERT model with the same layer size, embedding dimension, and fine-tuned hyperparameters, 
pre-trained using a linearized AxoLogic dataset, which has parentheses added for syntactical accuracy.

If we find significantly better performance with AxoLoTL on the generalization set than BERT, and also find 
significantly similar F1 scores between validation and generalization, we may say that AxoLoTL is
able to generalize learned logical rules, and we have succeeded at our goal. :robot:

---

## TODO:
- ### AxoLoTL Model
  - [ ] Pre-training pipeline
  - [ ] Dataloader and collate functions
  - [ ] Hyperparameter tuning
  - [ ] General refactoring and function comments
- ### AxoLogic Dataset
  - [ ] Linearization set for baseline model
  - [ ] Arguments for data explorer / generator
  - [ ] Refactor user-facing interface code
- ### Baseline Model
  - [ ] Pretrain BERT-mini model on linearized dataset
- ### Miscellaneous
  - [ ] Cite necessary papers in README
  - [ ] Add diagrams to Architecture section
  - [ ] Add more information about datasets in the Methodology section
  - [x] Enjoy yourself :smile:
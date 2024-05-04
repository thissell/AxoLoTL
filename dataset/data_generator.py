import time

import pandas as pd
import random
import numpy as np
from dgl.data.utils import save_graphs

from lib.context_free_grammar import TERM_NODE_SET, VAR_NODE_SET, SAFE_NODE_SET, BinaryNode, UnaryNode, EntailsNode, BeginNode
from lib.graph_utils import to_graph

'''
    Data generator for AxoLoTL AxoLogic dataset
'''

def create_valid_ast(max_len: int, vars: list):
    if max_len == 1:
        node = random.choice(list(vars) + TERM_NODE_SET)
        return node()
    else:
        if len(vars) > 0:
            se = SAFE_NODE_SET + ['var']
        else:
            se = SAFE_NODE_SET
        
        node = random.choice(se)
        if node == 'var':
            node = random.choice(list(vars))
            return node()
        if issubclass(node, BinaryNode):
            left = create_valid_ast(max_len - 1, vars)
            right = create_valid_ast(max_len - 1, vars)
            return node(left, right)
        elif issubclass(node, UnaryNode):
            child = create_valid_ast(max_len - 1, vars)
            return node(child)
        else:
            return node()

def generate_entailment_ast(max_len: int, var_num: int):
    vars = np.random.choice(VAR_NODE_SET, var_num, replace=False)
    x = create_valid_ast(max_len, vars)
    y = create_valid_ast(max_len, vars)

    en1 = EntailsNode(x, y)
    return BeginNode(en1)


def generate_dataset(max_len: int, var_num: int, amt: int, repl: int):
    data = []
    graphs = []
    for _ in range(amt):
        b = generate_entailment_ast(max_len, var_num)
        g, nl = to_graph(b)

        mask = random.sample(nl[2:], repl)
        before = g.ndata['feat'][[m[0] for m in mask]]

        rand = random.randint(1,10)
        if rand < 7:
            g.ndata['feat'][[m[0] for m in mask]] = 2
        elif rand >= 7 and rand < 10:
            g.ndata['feat'][[m[0] for m in mask]] = random.randint(3, 15)

        graphs.append(g)
        data.append({
            'masked_idx': mask[0][0], 
            'masked_correct': before.item(), 
            'entailment': b.eval(None)
        })

    return graphs, data

def split_into(max_len, var_num, amt, repl, split):
    global train_graphs, train_data, test_graphs, test_data, valid_graphs, valid_data

    g, d = generate_dataset(max_len, var_num, amt, repl)

    tr_idx = int(split[0] * amt)
    te_idx = int((split[0] + split[1]) * amt)

    train_graphs += g[:tr_idx]
    train_data += d[:tr_idx]

    test_graphs += g[tr_idx:te_idx]
    test_data += d[tr_idx:te_idx]

    valid_graphs += g[te_idx:]
    valid_data += d[te_idx:]

def save_data(title, data):
    df = pd.DataFrame.from_dict(data)
    df.to_csv(title)

if __name__ == "__main__":
    start = time.time()

    train_graphs = []
    train_data = []

    test_graphs = []
    test_data = []

    valid_graphs = []
    valid_data = []

    split_into(max_len=4, var_num=0, amt=100,  repl=1, split=(0.8, 0.1))
    split_into(max_len=4, var_num=1, amt=200,  repl=1, split=(0.8, 0.1))
    split_into(max_len=6, var_num=2, amt=500,  repl=1, split=(0.8, 0.1))
    split_into(max_len=6, var_num=3, amt=1200, repl=1, split=(0.8, 0.1))
    split_into(max_len=8, var_num=2, amt=1500, repl=1, split=(0.8, 0.1))
    split_into(max_len=8, var_num=4, amt=2000, repl=1, split=(0.8, 0.1))
    split_into(max_len=8, var_num=5, amt=5000, repl=1, split=(0.8, 0.1))
    split_into(max_len=8, var_num=0, amt=1000, repl=1, split=(0.8, 0.1))

    # this is only for validation to see if it can generalize
    gener_graphs, gener_data = generate_dataset(max_len = 11, var_num = 7, amt = 500, repl = 1)

    end = time.time()
    print(end - start)

    save_graphs('./generated/train_graphs.bin', train_graphs)
    save_data('./generated/train_data.csv', train_data)

    save_graphs('./generated/test_graphs.bin', test_graphs)
    save_data('./generated/test_data.csv', test_data)

    save_graphs('./generated/valid_graphs.bin', valid_graphs)
    save_data('./generated/valid_data.csv', valid_data)

    save_graphs('./generated/gener_graphs.bin', gener_graphs)
    save_data('./generated/gener_data.csv', gener_data)
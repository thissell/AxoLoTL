from dgl.data.utils import load_graphs
import pandas as pd

from lib.graph_utils import draw_tree
import random


'''
    Data Explorer for AxoLoTL AxoLogic Dataset
    [Very Unfinished.]
    
    TODO:
        - Arguments to chose between datasets
        - Use formatted strings instead of concatenation
        - More functionality [TBD]
'''

def get_node_statistics(graphs):
    node_count_total = 0
    node_count_max = 0
    for g in graphs:
        node_count = g.num_nodes()
        node_count_total += node_count
        if node_count > node_count_max:
            node_count_max = node_count

    print("node count max in dataset: " + str(node_count_max))
    print("node count total in dataset: " + str(node_count_total))
    print("node count avg in dataset: " + str(node_count_total / len(graphs)))

def get_entailment_statistics(data, graphs):
    cnt = 0
    for v in data['entailment']:
        if v:
            cnt += 1


    print("percent of graphs which are labeled with entailment: " + str(cnt / len(graphs) * 100) + "%.")

def draw_graph_sample(N, split='gener'):
    train_graphs, label_dict = load_graphs("./generated/" + split + "_graphs.bin")
    train_data = pd.read_csv("./generated/" + split + "_data.csv")

    # print statistics
    get_node_statistics(train_graphs)
    get_entailment_statistics(train_data, train_graphs)

    for _ in range(N):
        # draw random sample graph
        chosen_g = random.choice(train_graphs)

        print(chosen_g.ndata['feat'])
        draw_tree(chosen_g)

draw_graph_sample(5, "test")
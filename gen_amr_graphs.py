import argparse

import amrlib
import penman
import dgl
import torch
import pandas as pd
import os
import re
import sys

from dgl.data.utils import save_graphs
from tqdm import tqdm

DEBUG = True

MODEL_DIR = None # Default

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device", device)

# This a dictionary which auto-populates on access.
class IterMap(dict):
    def __init__(self, *args, **kw):
        super(IterMap, self).__init__(*args, **kw)
        self.count = 0

    def __getitem__(self, item):
        if item not in super(IterMap, self).keys():
            super(IterMap, self).__setitem__(item, self.count)
            self.count += 1

        return super(IterMap, self).__getitem__(item)

    def get_count(self):
        return self.count


GLOBAL_FEATURE_MAP = IterMap()


def append_penman(g_rep, g, node_map, is_conclusion: False):
    g_penman = penman.decode(g_rep)

    current = node_map.get_count()

    # Create nodes for instances.
    for t in g_penman.instances():
        g.add_nodes(1, {
            'x': torch.tensor([GLOBAL_FEATURE_MAP[t[2]]])
        })

        _ = node_map[t[0]]

        if DEBUG:
            print(str(node_map[t[0]]) + ": " + str(t[0]) + " (" + str(t[2]) + ")")

    # Then, of course, edges for edges.
    for t in g_penman.edges():
        g.add_edges(
            torch.tensor([node_map[t[0]]]),
            torch.tensor([node_map[t[2]]]),
            {
                'h': torch.tensor([GLOBAL_FEATURE_MAP[t[1]]])
            }
        )

        if DEBUG:
            print(node_map[t[0]], "->", node_map[t[2]])

    # And finally create both nodes and edges for attributes.
    for t in g_penman.attributes():
        if t[2] not in node_map:
            g.add_nodes(1, {
                'x': torch.tensor([GLOBAL_FEATURE_MAP[t[2]]])
            })

        if DEBUG:
            print(node_map[t[0]], "->", node_map[t[2]])

        g.add_edges(
            torch.tensor([node_map[t[0]]]),
            torch.tensor([node_map[t[2]]]),
            {
                'h': torch.tensor([GLOBAL_FEATURE_MAP[t[1]]])
            }
        )

    g.add_edges(
        torch.tensor([node_map["[ENTAILMENT]"]]),
        torch.tensor([current + 1]),
        {
            'h': torch.tensor([GLOBAL_FEATURE_MAP["[CONCLUSION]" if is_conclusion else "[PREMISES]"]])
        }
    )


def generate_graphs(sentences: list[str], conclusions: list[str]):
    global GLOBAL_FEATURE_MAP

    graphs: list[dgl.DGLGraph] = []
    g_reps = []
    c_reps = []

    # Load sentence-to-graph model from AMRLib
    stog = amrlib.load_stog_model(model_dir=MODEL_DIR)
    for sent, con in tqdm(zip(sentences, conclusions), total=len(sentences)):
        print(sent, con)
        sent_1 = stog.parse_sents([sent, con])
        g_reps.append(sent_1[0])
        c_reps.append(sent_1[1])

    for g_rep, c_rep in zip(g_reps, c_reps):
        node_map = IterMap()
        g = dgl.graph([])

        g.add_nodes(1, {
            'x': torch.tensor([GLOBAL_FEATURE_MAP["[ENTAILMENT]"]])
        })

        _ = node_map["[ENTAILMENT]"]
        append_penman(g_rep, g, node_map, is_conclusion=False)
        append_penman(c_rep, g, node_map, is_conclusion=True)

        graphs.append(g)

    return graphs


def load_dataset(path, head: int|None=None):
    full_path = os.path.join(".", path)
    df = pd.read_csv(full_path, sep='\t')

    if head is None:
        prem_unsafe = df['premises']
        conc_unsafe = df['conclusion']
        labl_unsafe = df['label']
    else:
        prem_unsafe = df['premises'].head(head)
        conc_unsafe = df['conclusion'].head(head)
        labl_unsafe = df['label'].head(head)

    premises = [' '.join(re.findall(r'\'(.*?)\'', p)) for p in prem_unsafe]

    return premises, conc_unsafe.to_list(), labl_unsafe.to_list()


label_map = {
    'True': 1,
    'False': 0,
    'Unknown': 2,
    'Uncertain': 2
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='AMR DGL Graph Generator',
        description='Takes sentence premise-conclusion pairs and creates AMR DeepGraphLibrary graphs.',
        epilog='For Project AxoLoTL. Thanks for using it! - jackson')

    parser.add_argument('-i', '--input', dest='input', nargs='+', type=str, required=True,
                        help="the name of the tsv dataset to load from.")  # input file name

    parser.add_argument('-o', '--output', dest='output', nargs='+', type=str, required=True,
                        help="the name of the bin dataset to save to.")  # output file name

    parser.add_argument('-l', '--head', dest='head', nargs='+', type=int, required=False,
                        help="the number of the dataset to load.")  # output file name

    args = parser.parse_args()

    if len(args.input) != len(args.output):
        print("ERROR! Please make sure the input and output filenames line up and have the same count.")
        exit(-1)

    for i, (input_path, output_path) in enumerate(zip(args.input, args.output)):
        print(input_path)

        to_head = None
        if args.head is not None and len(args.head) > i:
            to_head = args.head[i]

        prem, conc, labels = load_dataset(input_path, head=to_head)
        p_graphs = generate_graphs(prem, conc)

        save_graphs(output_path, p_graphs, {
            "truth": torch.tensor([label_map[l] for l in labels])
        })

        print(f"INFO: Saved graph dataset '{input_path}' -> '{output_path}'.")

    print("fin.")
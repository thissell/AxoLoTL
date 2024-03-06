from model.language_model import LanguageAxoLoTL
from utils.graph_generation import create_nonsense_batch

''' AxoLoTL: Axiomatic Language-Tree Learning'''
''' By Jackson Thissell '''


'''
TODO:
    [ ] DataLoader and collate functions
    [ ] Attention masks for node-feature padding
    [ ] Tune hyperparameters
    [ ] Let 'er rip (pre-training time!)
'''

# Constants
device = "cuda"


# Main
if __name__ == "__main__":
    config = {
        "num_atn_heads": 4,
        "top_k_vec": 8,
        "vocab_size": 16,
        "embed_dim": 48,
        "dropout": 0.1,
        "num_layers": 2
    }

    axolotl = LanguageAxoLoTL(config).to(device)
    '''
        g   DGL Graph object
        e   Eigenvalues from Laplacian Matrix
        u   Eigenvectors from Laplacian Matrix
        nf  Node Features (Integers)
        pe  Laplacian Positional Encodings
        l   Size of Graph in Nodes.
    '''
    g, e, u, nf, pe, l = create_nonsense_batch(num_nodes=128, num_graphs=32, top_k=config['top_k_vec'], device=device)
    s, tp = axolotl(g, e, u, nf, pe, l)

    print(s, tp)
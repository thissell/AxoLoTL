import torch.nn as nn

from model.modified_specformer import ModifiedSpecformer
from model.sine_encoding import SineEncoding


# The AxoLoTL Model
class AxoLoTL(nn.Module):
    def __init__(self, config):
        super(AxoLoTL, self).__init__()

        self.num_layers = config['num_layers']
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=config['embed_dim'], nhead=1, batch_first=True)
        self.pe_pre_encoder_ffn = nn.Linear(config['top_k_vec'], config['embed_dim'])
        self.pe_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.eig_encoder = SineEncoding(config['embed_dim'])
        self.tok_encoder = nn.Embedding(config['vocab_size'], config['embed_dim'])

        self.layers = nn.ModuleList([
            ModifiedSpecformer(config) for _ in range(self.num_layers)
        ])

    def forward(self, g, e, u, nf, pe, l):
        '''
            g       dgl graph
            e       lap. eigvals
            u       lap. eigvecs
            nf      node features
            l       size of graph
        '''

        eig = self.eig_encoder(e)

        pe = self.pe_pre_encoder_ffn(pe)
        nf = self.tok_encoder(nf) + self.pe_encoder(pe)

        for layer in self.layers:
            nf, eig = layer(eig, e, u, g, nf, l)

        return nf
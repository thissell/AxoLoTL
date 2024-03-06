import torch
import torch.nn as nn

from dgl import function as fn
from dgl.ops.edge_softmax import edge_softmax

from model.crosstalk_transformer import CrossTalkTransformer


# Modified from https://github.com/DSL-Lab/Specformer
class ModifiedSpecformer(nn.Module):
    def __init__(self, config):
        super(ModifiedSpecformer, self).__init__()
        self.cross_talk = CrossTalkTransformer(config)

        self.hidden_dim = config['embed_dim']
        self.nheads = config['num_atn_heads']

        self.decoder = nn.Linear(self.hidden_dim, self.nheads)

        self.adj_dropout = nn.Dropout(config['dropout'])
        self.filter_encoder = nn.Sequential(
            nn.Linear(self.nheads + 1, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
        )

        self.pre_conv_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU()
        )

        self.preffn_dropout = nn.Dropout(config['dropout'])
        self.conv_ffn_dropout = nn.Dropout(config['dropout'])
        self.conv_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )
        
    def forward(self, eig, e, u, g, nf, length):
        ut = u.transpose(1, 2)
        e_mask, edge_idx = self.length_to_mask(length)

        nf_mask = torch.full_like(e_mask, False)
        e_mask = torch.concat((e_mask, nf_mask), dim=1)

        eig, nf = self.cross_talk(eig, nf, key_padding_mask=e_mask)
        new_e = self.decoder(eig).transpose(2, 1)
        diag_e = torch.diag_embed(new_e)

        identity = torch.diag_embed(torch.ones_like(e))
        bases = [identity]

        for i in range(self.nheads):
            filters = u @ diag_e[:, i, :, :] @ ut
            bases.append(filters)

        bases = torch.stack(bases, axis=-1)
        bases = bases[edge_idx]
        bases = self.adj_dropout(self.filter_encoder(bases))
        bases = edge_softmax(g, bases)

        nf_flat = torch.flatten(nf, 0, 1)
        with g.local_scope():
            g.ndata['x'] = nf_flat
            g.apply_edges(fn.copy_u('x', '_x'))

            xee = self.pre_conv_ffn(g.edata['_x']) * bases
            g.edata['v'] = xee
            g.update_all(fn.copy_e('v', '_aggr_e'), fn.sum('_aggr_e', 'aggr_e'))

            y = g.ndata['aggr_e']
            y = self.preffn_dropout(y)
            nf_flat = nf_flat + y
            
            y = self.conv_ffn(nf_flat)
            y = self.conv_ffn_dropout(y)
            nf_flat = nf_flat + y
        
        return nf_flat.reshape_as(nf), eig

    def length_to_mask(self, length):
        B = len(length)
        N = length.max().item()
        mask1d = torch.arange(N, device=length.device).expand(B, N) >= length.unsqueeze(1)
        mask2d = (~mask1d).float().unsqueeze(2) @ (~mask1d).float().unsqueeze(1)
        mask2d = mask2d.bool()

        return mask1d, mask2d
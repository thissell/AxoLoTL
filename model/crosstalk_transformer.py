import torch.nn as nn
import torch


# Cross-Talk Self-Attention
class CrossTalkSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(config['embed_dim'], config['num_atn_heads'], batch_first=True)

        self.dense1 = nn.Linear(config['embed_dim'], config['embed_dim'] // 2)
        self.dense2 = nn.Linear(config['embed_dim'], config['embed_dim'] // 2)

    def forward(self, cross1, cross2, key_padding_mask=None):
        h = torch.cat((cross1, cross2), 1)
        h = self.multihead_attn(h, h, h, key_padding_mask)

        output1 = self.dense1(h[0])
        output2 = self.dense2(h[0])

        return output1.reshape_as(cross1), output2.reshape_as(cross2)
    

# Residual Feed-Feedforward for Transformer
class ResidualFeedForward(nn.Module):
    def __init__(self, dropout, embed_dim):
        super(ResidualFeedForward, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        x_norm = self.norm(x)
        x_norm = self.ffn(x_norm)
        return x + self.dropout(x_norm)


# Cross-Talk Transformer
class CrossTalkTransformer(nn.Module):
    def __init__(self, config):
        super(CrossTalkTransformer, self).__init__()
        assert config["embed_dim"] % 2 == 0, "The embedding dimension is not a multiple of two."

        self.config = config        

        self.crstlk_mha = CrossTalkSelfAttention(config)

        self.eig_mha_norm = nn.LayerNorm(config['embed_dim'])
        self.eig_mha_dropout = nn.Dropout(config['dropout'])
        self.eig_ffn = ResidualFeedForward(config['dropout'], config['embed_dim'])

        self.feat_mha_norm = nn.LayerNorm(config['embed_dim'])
        self.feat_mha_dropout = nn.Dropout(config['dropout'])
        self.feat_ffn = ResidualFeedForward(config['dropout'], config['embed_dim'])

    def forward(self, eig, feat, key_padding_mask=None):
        eig = self.eig_mha_norm(eig)
        feat = self.feat_mha_norm(feat)
        
        mha_eig, mha_feat = self.crstlk_mha(eig, feat, key_padding_mask)
        
        eig = eig + self.eig_mha_dropout(mha_eig)
        feat = feat + self.feat_mha_dropout(mha_feat)

        return self.eig_ffn(eig), self.feat_ffn(feat)
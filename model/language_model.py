from torch import nn

from model.axolotl import AxoLoTL


# Entailment Prediction Module
class EntailmentPrediction(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 1)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, x):
        return self.log_sigmoid(self.linear(x[:, 0]))


# Masked Token Predction Module
class MaskedTokenPrediction(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.log_softmax(self.linear(x))

    
# AxoLoTL Language Model
class LanguageAxoLoTL(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.axolotl = AxoLoTL(config)

        self.entailment_prediction = EntailmentPrediction(config['embed_dim'])
        self.masked_token_prediction = MaskedTokenPrediction(config['embed_dim'], config['vocab_size'])

    def forward(self, g, e, u, nf, pe, l):
        h = self.axolotl(g, e, u, nf, pe, l)

        return self.entailment_prediction(h), self.masked_token_prediction(h)
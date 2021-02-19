import mindspore
import mindspore.nn as nn
import mindspore.ops.functional as F
# from src.utils
class Encoder(nn.Cell):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, has_bias=True, batch_first=True, bidirectional=True)
        self.W_h = nn.Dense(hidden_dim * 2, hidden_dim * 2, has_bias=False)

    def construct(self, inputs, seq_lens):
        embedded = self.embedding(inputs)

        # packed = 

import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as P
from layers.rnns import LSTM
from layers.layers import Dense, Embedding
from mindspore import Tensor
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out

def init_lstm_wt(lstm):
    for name, param in lstm.parameters_and_names():
        if name.startwith('weight_'):
            param.set_data(initializer(Uniform(0.02), param.shape))
        elif name.startwith('bias_'):
            # set forget bias to 1
            n = param.shape[0]
            start, end = n // 4, n // 2
            data = Tensor(np.zeros(param.shape), param.dtype)
            data[start:end] = 1.
            param.set_data(data)

def init_linear_wt(linear):
    linear.weight.set_data(initializer(Normal(1e-4), linear.weight.shape))
    if linear.has_bias:
        linear.bias.set_data(initializer(Normal(1e-4), linear.bias.shape))

def init_wt_normal(wt):
    wt.set_data(initializer(Normal(1e-4), wt.shape))

def init_wt_unif(wt):
    wt.set_data(initializer(Uniform(0.02), wt.shape))

class Encoder(nn.Cell):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        init_wt_normal(self.embedding.embedding_table)

        self.lstm = LSTM(embed_dim, hidden_dim, num_layers=1, has_bias=True, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)
        
        self.W_h = Dense(hidden_dim * 2, hidden_dim * 2, has_bias=False)
        self.hidden_dim = hidden_dim
        
    def construct(self, inputs, seq_lens):
        embedded = self.embedding(inputs)
        output, hidden = self.lstm(embedded, seq_length=seq_lens)
        encoder_feature = output.view(-1, 2 * self.hidden_dim)
        encoder_feature = self.W_h(encoder_feature)

        return output, encoder_feature, hidden

class ReduceState(nn.Cell):
    def __init__(self, hidden_dim):
        super().__init__()
        self.reduce_h = Dense(hidden_dim * 2, hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = Dense(hidden_dim * 2, hidden_dim)
        init_linear_wt(self.reduce_c)
        
        self.hidden_dim = hidden_dim

    def construct(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = P.Transpose()(h, (1, 0, 2)).view(-1, self.hidden_dim * 2)
        hidden_reduced_h = P.ReLU()(self.reduce_h(h))
        hidden_reduced_h = P.ExpandDims()(hidden_reduced_h, 0)
        c_in = P.Transpose()(c, (1, 0, 2)).view(-1, self.hidden_dim * 2)
        hidden_reduced_c = P.ReLU()(self.reduce_c(c_in))
        hidden_reduced_c = P.ExpandDims()(hidden_reduced_c, 0)

        return (hidden_reduced_h, hidden_reduced_c)

class Attention(nn.Cell):
    def __init__(self, hidden_dim, is_coverage=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.is_coverage = is_coverage

        if is_coverage:
            self.W_c = Dense(1, hidden_dim * 2, has_bias=False)
        self.decode_proj = Dense(hidden_dim * 2, hidden_dim * 2)
        self.v = Dense(hidden_dim * 2, 1, has_bias=False)

    def construct(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = encoder_outputs.shape

        dec_fea = self.decode_proj(s_t_hat) # (B, 2 * hidden_dim)
        dec_fea_expand = P.ExpandDims()(dec_fea, 1)
        dec_fea_expand = P.Tile()(dec_fea_expand, (b, t_k, n))

        att_features = encoder_feature + dec_fea_expand
        if self.is_coverage:
            coverage_input = coverage.view(-1, 1) # (B * t_k, 1)
            coverage_feature = self.W_c(coverage_input) # (B * t_k, 2 * hidden_dim)
            att_features = att_features + coverage_feature

        e = P.Tanh()(att_features) # (B * t_k, 2 * hidden_dim)
        scores = self.v(e) # (B * t_k, 1)
        scores = scores.view(-1, t_k) # (B, t_k)

        attn_dist_ = P.Softmax(1)(scores) * enc_padding_mask # (B, t_k)
        normalization_factor = P.ReduceSum(True)(attn_dist_, 1)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = P.ExpandDims()(attn_dist, 1) # (B, 1, t_k)
        c_t = P.BatchMatMul(attn_dist, encoder_outputs) # (B, 1, n)
        c_t = c_t.view(-1, self.hidden_dim * 2) # (B, 2 * hidden_dim)

        attn_dist = attn_dist.view(-1, t_k)

        if self.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist
        
        return c_t, attn_dist, coverage

class Decoder(nn.Cell):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pointer_gen=True, is_coverage=False):
        super().__init__()
        self.attention_network = Attention(hidden_dim, is_coverage)
        # decoder
        self.embedding = Embedding(vocab_size, embed_dim)
        init_wt_normal(self.embedding.embedding_table)

        self.x_context = Dense(hidden_dim * 2 + embed_dim, embed_dim)

        self.lstm = LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if pointer_gen:
            self.p_gen_linear = Dense(hidden_dim * 4 + embed_dim, 1)

        self.out1 = Dense(hidden_dim * 3, hidden_dim)
        self.out2 = Dense(hidden_dim, vocab_size)
        init_linear_wt(self.out2)
        
    def construct(self):
        pass
        
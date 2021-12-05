import mindspore
import numpy as np
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor
from mindspore.common.initializer import initializer, Uniform, Normal
from mindspore.ops.primitive import constexpr
from .layers.rnns import LSTM
from .layers.layers import Dense, Embedding

@constexpr
def range_tensor(start, end):
    return Tensor(np.arange(start, end), mindspore.int32)

def init_lstm_wt(lstm):
    for name, param in lstm.parameters_and_names():
        if name.startswith('weight_'):
            param.set_data(initializer(Uniform(0.02), param.shape))
        elif name.startswith('bias_'):
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
        output, hidden = self.lstm(embedded)
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
        h_in = h.swapaxes(0, 1).view(-1, self.hidden_dim * 2)
        hidden_reduced_h = P.ReLU()(self.reduce_h(h_in))
        hidden_reduced_h = P.ExpandDims()(hidden_reduced_h, 0)
        c_in = c.swapaxes(0, 1).view(-1, self.hidden_dim * 2)
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
        dec_fea_expand = P.BroadcastTo((b, t_k, n))(dec_fea_expand).view(-1, n)

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
        c_t = P.BatchMatMul()(attn_dist, encoder_outputs) # (B, 1, n)
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
        
        self.hidden_dim = hidden_dim
        self.pointer_gen = pointer_gen

    def construct(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            h_decoder = h_decoder.view(-1, self.hidden_dim)
            c_decoder = c_decoder.view(-1, self.hidden_dim)
            s_t_hat = P.Concat(1)((h_decoder, c_decoder)) # (B, 2 * hidden_dim)
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embed = self.embedding(y_t_1)
        x = self.x_context(P.Concat(1)((c_t_1, y_t_1_embed)))
        lstm_out, s_t = self.lstm(P.ExpandDims()(x, 1), s_t_1)

        h_decoder, c_decoder = s_t
        h_decoder = h_decoder.view(-1, self.hidden_dim)
        c_decoder = c_decoder.view(-1, self.hidden_dim)
        s_t_hat = P.Concat(1)((h_decoder, c_decoder))

        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if self.pointer_gen:
            p_gen_input = P.Concat(1)((c_t, s_t_hat, x)) # (B, 2 * 2 * hidden_dim + embed_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = P.Sigmoid()(p_gen)
        
        output = P.Concat(1)((lstm_out.view(-1, self.hidden_dim), c_t)) # (B, hidden_dim * 3)
        output = self.out1(output) # (B, hidden_dim)

        output = self.out2(output) # (B, vocab_size)
        vocab_dist = P.Softmax(1)(output)

        if self.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = P.Concat(1)((vocab_dist_, extra_zeros))
            
            # like pytorch scatter_add
            batch_size, attn_len = enc_batch_extend_vocab.shape
            batch_num = range_tensor(0, batch_size)
            batch_num = P.ExpandDims()(batch_num, 1)
            batch_num = P.Tile()(batch_num, (1, attn_len))
            indices = P.Stack(2)((batch_num, enc_batch_extend_vocab))
            shape = (batch_size, vocab_dist_.shape[1])
            attn_dist_ = P.ScatterNd()(indices, attn_dist_, shape)
            final_dist = vocab_dist_ + attn_dist_
        else:
            final_dist = vocab_dist
        
        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

class TrainOneBatch(nn.Cell):
    def __init__(self, encoder, decoder, reduce_state, config):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state
        self.max_dec_steps = config.max_dec_steps
        self.eps = config.eps
        self.cov_loss_wt = config.cov_loss_wt
        self.is_coverage = config.is_coverage

    def construct(self, enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, \
                  extra_zeros, c_t_1, coverage, \
                  dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch):
        encoder_outputs, encoder_feature, encoder_hidden = self.encoder(enc_batch, enc_lens)
        s_t_1 = self.reduce_state(encoder_hidden)
        step_losses = ()
        for di in range(self.max_dec_steps):
            y_t_1 = dec_batch[:, di] # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]
            gold_probs = P.GatherD()(final_dist, 1, P.ExpandDims()(target, 1)).squeeze()
            step_loss = -P.Log()(gold_probs + self.eps)
            if self.is_coverage:
                step_coverage_loss = P.ReduceSum()(P.Minimum()(attn_dist, coverage), 1)
                step_loss = step_loss + self.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses += (step_loss,)
        sum_losses = P.ReduceSum()(P.Stack(1)(step_losses), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = P.ReduceMean()(batch_avg_loss)

        return loss

class TrainOneStep(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, sens=1, max_grad_norm=1.0):
        super().__init__(network, optimizer, sens=sens)
        self.max_grad_norm = max_grad_norm

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = P.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        grads = P.clip_by_global_norm(grads, self.max_grad_norm)
        loss = P.depend(loss, self.optimizer(grads))
        return loss

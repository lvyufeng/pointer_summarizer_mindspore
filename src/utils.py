import mindspore
import numpy as np
import mindspore.numpy as mnp
from mindspore import Tensor
import src.config as config

def get_input_from_batch(batch):
    batch_size = len(batch.enc_lens)
    enc_batch = Tensor(batch.enc_batch, mindspore.int32)
    enc_padding_mask = Tensor(batch.enc_padding_mask, mindspore.float32)
    enc_lens = Tensor(batch.enc_lens, mindspore.int32)
    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = Tensor(batch.enc_batch_extend_vocab, mindspore.int32)
        if batch.max_art_oovs > 0:
            extra_zeros = mnp.zeros((batch_size, batch.max_art_oovs))
    
    c_t_1 = mnp.zeros((batch_size, 2 * config.hidden_dim))
    coverage = None
    if config.is_coverage:
        coverage = mnp.zeros(enc_batch.shape)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

def get_output_from_batch(batch):
    dec_batch = Tensor(batch.dec_batch, mindspore.int32)
    dec_padding_mask = Tensor(batch.dec_padding_mask, mindspore.float32)
    dec_lens = Tensor(batch.dec_lens, mindspore.int32)
    max_dec_len = Tensor(np.max(batch.dec_lens), mindspore.int32)
    dec_lens_var = Tensor(dec_lens, mindspore.float32)
    target_batch = Tensor(batch.target_batch, mindspore.int32)

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch
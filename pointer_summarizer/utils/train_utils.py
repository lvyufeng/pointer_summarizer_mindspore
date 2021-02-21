import mindspore
import numpy as np
from mindspore import Tensor
from mindspore.ops.primitive import constexpr

@constexpr
def get_input_from_batch(batch, hidden_dim, pointer_gen, is_coverage):
    batch_size = len(batch.enc_lens)
    enc_batch = Tensor(batch.enc_batch, mindspore.int32)
    enc_padding_mask = Tensor(batch.enc_padding_mask, mindspore.float32)
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    if pointer_gen:
        enc_batch_extend_vocab = Tensor(batch.enc_batch_extend_vocab, mindspore.int32)
        if batch.max_art_oovs > 0:
            extra_zeros = Tensor(np.zeros(batch_size, batch.max_art_oovs), mindspore.float32)
    
    c_t_1 = Tensor(np.zeros(batch_size, 2*hidden_dim))
    coverage = None
    if is_coverage:
        coverage = Tensor(np.zeros(enc_batch.shape), mindspore.float32)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

@constexpr
def get_output_from_batch(batch):
    dec_batch = Tensor(batch.dec_batch, mindspore.int32)
    dec_padding_mask = Tensor(batch.dec_padding_mask, mindspore.float32)
    dec_lens = batch.dec_lens
    max_dec_len = Tensor(np.max(dec_lens), mindspore.int32)
    dec_lens_var = Tensor(dec_lens, mindspore.float32)
    target_batch = Tensor(batch.target_batch, mindspore.int32)

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch
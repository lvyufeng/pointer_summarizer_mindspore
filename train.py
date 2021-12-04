import time
import mindspore.nn as nn
import mindspore.ops as P
from src.model import Encoder, Decoder, ReduceState, TrainOneBatch, TrainOneStep
from src.utils import get_input_from_batch, get_output_from_batch
from src.data import Vocab
from src.batcher import Batcher
import src.config as config
from mindspore import context

context.set_context(max_call_depth=10000, enable_graph_kernel=True)

def train_iters(trainer, batcher, n_iters, print_interval=1000):
    print("Start network training")
    iter = 0
    start = time.time()
    while iter < n_iters:
        batch = batcher.next_batch()
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch)
        
        loss = trainer(enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, \
                        extra_zeros, c_t_1, coverage, \
                        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch)
        iter += 1
        
        if iter % print_interval == 0:
            print('steps %d, seconds for %d batch: %.2f , loss: %f' % \
                  (iter, print_interval, time.time() - start, loss.asnumpy()))

encoder = Encoder(config.vocab_size, config.emb_dim, config.hidden_dim)
decoder = Decoder(config.vocab_size, config.emb_dim, config.hidden_dim, config.pointer_gen, config.is_coverage)
reduce_state = ReduceState(config.hidden_dim)

network = TrainOneBatch(encoder, decoder, reduce_state, config)
optimizer = nn.Adagrad(network.trainable_params(), accum=config.adagrad_init_acc, learning_rate=config.lr)
trainer = TrainOneStep(network, optimizer, max_grad_norm=config.max_grad_norm)

vocab = Vocab('data/finished_files/vocab', config.vocab_size)
batcher = Batcher('data/finished_files/test.mindrecord', vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
batch = batcher.next_batch()
enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch)
dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
    get_output_from_batch(batch)

# pre-compile
print("Start network compile")
trainer.compile(enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, \
                extra_zeros, c_t_1, coverage, \
                dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch)
train_iters(trainer, batcher, 10000, 100)

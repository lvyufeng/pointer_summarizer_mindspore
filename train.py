import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from src.model import Encoder, Decoder, ReduceState
from src.utils.train_utils import get_input_from_batch, get_output_from_batch
from src.data import Vocab
from src.batcher import Batcher

class TrainOneBatch(nn.Cell):
    def __init__(self, encoder, decoder, reduce_state, config):
        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state
        self.max_dec_steps = config.max_dec_steps
        self.eps = config.eps
        self.cov_loss_wt = config.cov_loss_wt

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
        sum_losses = P.ReduceSum(P.Stack(1)(step_losses), 1)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = P.ReduceMean(batch_avg_loss)

        return loss

vocab = Vocab('data/finished_files/vocab', 20000)
batcher = Batcher('data/finished_files/test.mindrecord', vocab, mode='train',
                               batch_size=32, single_pass=False)
batch = batcher.next_batch()
print(batch.enc_lens)
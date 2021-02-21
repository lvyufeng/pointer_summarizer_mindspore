import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from model import Encoder, Decoder, ReduceState
from utils.train_utils import get_input_from_batch, get_output_from_batch

class TrainOneBatch(nn.Cell):
    def __init__(self, encoder, decoder, reduce_state):
        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

    def construct(self):
        pass


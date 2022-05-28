'''RNN Cells module, include RNNCell, GRUCell, LSTMCell'''
import math
import numpy as np
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Uniform

def rnn_tanh_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''RNN cell function with tanh activation'''
    if b_ih is None:
        igates = P.MatMul(False, True)(inputs, w_ih)
        hgates = P.MatMul(False, True)(hidden, w_hh)
    else:
        igates = P.MatMul(False, True)(inputs, w_ih) + b_ih
        hgates = P.MatMul(False, True)(hidden, w_hh) + b_hh
    return P.Tanh()(igates + hgates)

def rnn_relu_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''RNN cell function with relu activation'''
    if b_ih is None:
        igates = P.MatMul(False, True)(inputs, w_ih)
        hgates = P.MatMul(False, True)(hidden, w_hh)
    else:
        igates = P.MatMul(False, True)(inputs, w_ih) + b_ih
        hgates = P.MatMul(False, True)(hidden, w_hh) + b_hh
    return P.ReLU()(igates + hgates)

def lstm_cell(inputs, hx, cx, w_ih, w_hh, b_ih, b_hh):
    '''LSTM cell function'''
    if b_ih is None:
        gates = P.MatMul(False, True)(inputs, w_ih) + P.MatMul(False, True)(hx, w_hh)
    else:
        gates = P.MatMul(False, True)(inputs, w_ih) + P.MatMul(False, True)(hx, w_hh) + b_ih + b_hh
    ingate, forgetgate, cellgate, outgate = P.Split(1, 4)(gates)

    ingate = P.Sigmoid()(ingate)
    forgetgate = P.Sigmoid()(forgetgate)
    cellgate = P.Tanh()(cellgate)
    outgate = P.Sigmoid()(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * P.Tanh()(cy)

    return hy, cy

def gru_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''GRU cell function'''
    if b_ih is None:
        gi = P.MatMul(False, True)(inputs, w_ih)
        gh = P.MatMul(False, True)(hidden, w_hh)
    else:
        gi = P.MatMul(False, True)(inputs, w_ih) + b_ih
        gh = P.MatMul(False, True)(hidden, w_hh) + b_hh
    i_r, i_i, i_n = P.Split(1, 3)(gi)
    h_r, h_i, h_n = P.Split(1, 3)(gh)

    resetgate = P.Sigmoid()(i_r + h_r)
    inputgate = P.Sigmoid()(i_i + h_i)
    newgate = P.Tanh()(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy

class RNNCellBase(nn.Cell):
    '''Basic class for RNN Cells'''
    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(Tensor(np.random.randn(num_chunks * hidden_size, input_size).astype(np.float32)))
        self.weight_hh = Parameter(Tensor(np.random.randn(num_chunks * hidden_size, hidden_size).astype(np.float32)))
        if bias:
            self.bias_ih = Parameter(Tensor(np.random.randn(num_chunks * hidden_size).astype(np.float32)))
            self.bias_hh = Parameter(Tensor(np.random.randn(num_chunks * hidden_size).astype(np.float32)))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.hidden_size)
        for weight in self.get_parameters():
            weight.set_data(initializer(Uniform(stdv), weight.shape))

class RNNCell(RNNCellBase):
    r"""
    An Elman RNN cell with tanh or ReLU non-linearity.

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Here :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If `nonlinearity` is `relu`, then `relu` is used instead of `tanh`.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        nonlinearity (str): The non-linearity to use. Can be either `tanh` or `relu`. Default: `tanh`.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch_size, `input_size`).
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and shape (batch_size, `hidden_size`).
          Data type of `hx` must be the same as `x`.

    Outputs:
        - **hx'** (Tensor) - Tensor of shape (batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size` or `hidden_size` is not an int or not greater than 0.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `nonlinearity` is not in ['tanh', 'relu'].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.RNNCell(10, 16)
        >>> x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> hx = Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        >>>     hx = net(x[i], hx)
        >>>     output.append(hx)
        >>> print(output[0].shape)
        (3, 16)
    """
    _non_linearity = ['tanh', 'relu']
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh"):
        super().__init__(input_size, hidden_size, bias, num_chunks=1)
        if nonlinearity not in self._non_linearity:
            raise ValueError("Unknown nonlinearity: {}".format(nonlinearity))
        self.nonlinearity = nonlinearity

    def construct(self, inputs, hx):
        if self.nonlinearity == "tanh":
            ret = rnn_tanh_cell(inputs, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        else:
            ret = rnn_relu_cell(inputs, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        return ret

class LSTMCell(RNNCellBase):
    r"""
    A LSTM (Long Short-Term Memory) cell.

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ix} x_t + b_{ix} + W_{ih} h_{(t-1)} + b_{ih}) \\
            f_t = \sigma(W_{fx} x_t + b_{fx} + W_{fh} h_{(t-1)} + b_{fh}) \\
            \tilde{c}_t = \tanh(W_{cx} x_t + b_{cx} + W_{ch} h_{(t-1)} + b_{ch}) \\
            o_t = \sigma(W_{ox} x_t + b_{ox} + W_{oh} h_{(t-1)} + b_{oh}) \\
            c_t = f_t * c_{(t-1)} + i_t * \tilde{c}_t \\
            h_t = o_t * \tanh(c_t) \\
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ix}, b_{ix}` are the weight and bias used to transform from input :math:`x` to :math:`i`.
    Details can be found in paper `LONG SHORT-TERM MEMORY
    <https://www.bioinf.jku.at/publications/older/2604.pdf>`_ and
    `Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling
    <https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43905.pdf>`_.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch_size, `input_size`).
        - **hx** (tuple) - A tuple of two Tensors (h_0, c_0) both of data type mindspore.float32 and shape (batch_size, `hidden_size`).
          Data type of `hx` must be the same as `x`.

    Outputs:
        - **hx'** (Tensor) - A tuple of two Tensors (h', c') both of data shape (batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` is not an int.
        TypeError: If `has_bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.LSTMCell(10, 16)
        >>> x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> h = Tensor(np.ones([3, 16]).astype(np.float32))
        >>> c = Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        >>>     hx = net(x[i], (h, c))
        >>>     output.append(hx)
        >>> print(output[0][0].shape)
        (3, 16)
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias, num_chunks=4)
        self.support_non_tensor_inputs = True

    def construct(self, inputs, hx):
        hx, cx = hx
        return lstm_cell(inputs, hx, cx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

class GRUCell(RNNCellBase):
    r"""
    A GRU(Gated Recurrent Unit) cell.

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ir}, b_{ir}` are the weight and bias used to transform from input :math:`x` to :math:`r`.
    Details can be found in paper
    `Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
    <https://aclanthology.org/D14-1179.pdf>`_.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch_size, `input_size`).
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and shape (batch_size, `hidden_size`).
          Data type of `hx` must be the same as `x`.

    Outputs:
        - **hx'** (Tensor) - Tensor of shape (batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` is not an int.
        TypeError: If `has_bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.GRUCell(10, 16)
        >>> x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> hx = Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        >>>     hx = net(x[i], hx)
        >>>     output.append(hx)
        >>> print(output[0].shape)
        (3, 16)
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias, num_chunks=3)

    def construct(self, inputs, hx):
        return gru_cell(inputs, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
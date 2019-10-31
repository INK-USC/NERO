import tensorflow as tf

INF = 1e30


class Cudnn_RNN:

    def __init__(self, num_layers, num_units, mode="lstm", keep_prob=1.0, is_train=None, scope="cudnn_rnn"):
        self.num_layers = num_layers
        self.rnns = []
        self.mode = mode
        if mode == "gru":
            rnn = tf.contrib.cudnn_rnn.CudnnGRU
        elif mode == "lstm":
            rnn = tf.contrib.cudnn_rnn.CudnnLSTM
        else:
            raise Exception("Unknown mode for rnn")
        for layer in range(num_layers):
            rnn_fw = rnn(1, num_units)
            rnn_bw = rnn(1, num_units)
            self.rnns.append((rnn_fw, rnn_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            rnn_fw, rnn_bw = self.rnns[layer]
            output = dropout(outputs[-1], keep_prob=keep_prob, is_train=is_train)
            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, state_fw = rnn_fw(output)
            with tf.variable_scope("bw_{}".format(layer)):
                inputs_bw = tf.reverse_sequence(output, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
                out_bw, state_bw = rnn_bw(inputs_bw)
                out_bw = tf.reverse_sequence(out_bw, seq_lengths=seq_len, seq_axis=0, batch_axis=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers is True:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        state_fw = tf.squeeze(state_fw[0], [0])
        state_bw = tf.squeeze(state_bw[0], [0])
        state = tf.concat([state_fw, state_bw], axis=1)
        return res, state


def dropout(args, keep_prob, is_train, mode=None):
    if keep_prob < 1.0:
        noise_shape = None
        scale = 1.0
        shape = tf.shape(args)
        if mode == "embedding" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], shape[1], 1]
            scale = keep_prob
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        args = tf.cond(is_train, lambda: tf.nn.dropout(
            args, keep_prob, noise_shape=noise_shape) * scale, lambda: args)
    return args


def softmax_mask(val, mask):
    rank_val = len(val.get_shape().as_list())
    rank_mask = len(mask.get_shape().as_list())
    if rank_val - rank_mask == 1:
        mask = tf.expand_dims(mask, axis=-1)
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def dense(inputs, hidden, use_bias=True, initializer=None, scope="dense"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden], initializer=initializer, dtype=tf.float32)
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable("b", [hidden], initializer=tf.constant_initializer(0.), dtype=tf.float32)
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res


def attention(inputs, hidden, mask=None, keep_prob=1.0, is_train=None, scope="attention"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs = dropout(inputs, keep_prob=keep_prob, is_train=is_train)
        s0 = tf.nn.tanh(dense(inputs, hidden, scope="dense1"))
        s = tf.squeeze(dense(s0, 1, use_bias=False, initializer=tf.constant_initializer(0.), scope="dense2"), [2])
        if mask is not None:
            s = softmax_mask(s, mask)
        a = tf.nn.softmax(s)
        return a


def log(x):
    return tf.log(tf.clip_by_value(x, 1e-5, 1.0))


def mean(seq, mask):
    mask = tf.cast(mask, tf.float32)
    length = tf.expand_dims(tf.reduce_sum(mask, axis=1), axis=1)
    mask = tf.expand_dims(mask, axis=2)
    emb = tf.reduce_sum(seq * mask, axis=1) / length
    return emb


def cosine(seq1, seq2, weighted=True, scope="cosine"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = seq1.get_shape().as_list()[-1]
        if weighted:
            weight = tf.get_variable("weight", [1, dim])
            seq1 = seq1 * weight
            seq2 = seq2 * weight
        norm1 = tf.norm(seq1 + 1e-5, axis=1, keepdims=True)
        norm2 = tf.norm(seq2 + 1e-5, axis=1, keepdims=True)
        sim = tf.matmul(seq1 / norm1, tf.transpose(seq2 / norm2, [1, 0]))
        return sim

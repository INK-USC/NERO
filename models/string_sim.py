import tensorflow as tf
from func import Cudnn_RNN, dropout, attention, cosine, mean


def mean_match(mid, pat, mid_mask, pat_mask, keep_prob, is_train):
    pat_d = dropout(pat, keep_prob=keep_prob, is_train=is_train)
    mid_v = mean(mid, mask=mid_mask)
    pat_v = mean(pat, mask=pat_mask)
    pat_v_d = mean(pat_d, mask=pat_mask)
    sur_sim = cosine(mid_v, pat_v, weighted=False)
    pat_sim = cosine(pat_v, pat_v_d, weighted=False)
    return sur_sim, pat_sim


def att_match(mid, pat, mid_mask, pat_mask, hidden, keep_prob, is_train):
    mid_d = dropout(mid, keep_prob=keep_prob, is_train=is_train)
    pat_d = dropout(pat, keep_prob=keep_prob, is_train=is_train)
    mid_a = attention(mid_d, hidden, mask=mid_mask)
    pat_a = attention(pat_d, hidden, mask=pat_mask)
    mid_v = tf.reduce_sum(tf.expand_dims(mid_a, axis=2) * mid, axis=1)
    pat_v = tf.reduce_sum(tf.expand_dims(pat_a, axis=2) * pat, axis=1)
    pat_v_d = tf.reduce_sum(tf.expand_dims(pat_a, axis=2) * pat_d, axis=1)
    sur_sim = cosine(mid_v, pat_v_d)
    pat_sim = cosine(pat_v, pat_v_d)
    return sur_sim, pat_sim


def lstm_match(mid, pat, mid_mask, pat_mask, mid_len, pat_len, hidden, keep_prob, is_train):

    rnn = Cudnn_RNN(num_layers=1, num_units=hidden // 2)
    mid, _ = rnn(mid, seq_len=mid_len, concat_layers=False)
    pat, _ = rnn(pat, seq_len=pat_len, concat_layers=False)

    mid_d = dropout(mid, keep_prob=keep_prob, is_train=is_train)
    pat_d = dropout(pat, keep_prob=keep_prob, is_train=is_train)
    mid_a = attention(mid_d, hidden, mask=mid_mask)
    pat_a = attention(pat_d, hidden, mask=pat_mask)

    mid_v = tf.reduce_sum(tf.expand_dims(mid_a, axis=2) * mid, axis=1)
    pat_v = tf.reduce_sum(tf.expand_dims(pat_a, axis=2) * pat, axis=1)
    pat_v_d = tf.reduce_sum(tf.expand_dims(pat_a, axis=2) * pat_d, axis=1)
    sur_sim = cosine(mid_v, pat_v_d)
    pat_sim = cosine(pat_v, pat_v_d)
    return sur_sim, pat_sim

import tensorflow as tf
from func import Cudnn_RNN, dropout, attention, dense, log
from models.string_sim import att_match


class Soft_Match(object):
    def __init__(self, config, word_mat, word2idx_dict=None):
        self.config = config
        self.word_mat = word_mat
        self.word2idx_dict = word2idx_dict
        self.global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), dtype=tf.float32)

        self.sent = tf.placeholder(tf.int32, shape=(None, config.length), name="sent")
        self.mid = tf.placeholder(tf.int32, shape=(None, config.length), name="mid")
        self.rel = tf.placeholder(tf.float32, shape=(None, config.num_class), name="rel")
        self.hard = tf.placeholder(tf.int32, shape=(None), name="hard_res")

        self.pats = tf.placeholder(tf.int32, shape=(None, 10), name="pats")
        self.rels = tf.placeholder(tf.float32, shape=(None, config.num_class), name="rels")
        self.weight = tf.placeholder(tf.float32, shape=(None,), name="weight")
        self.num_pat = tf.shape(self.pats)[0]
        self.hard_res = tf.one_hot(self.hard, depth=self.num_pat, dtype=tf.float32)

        self.is_train = tf.placeholder(tf.bool)
        self.lr = tf.get_variable("lr", [], initializer=tf.constant_initializer(config.init_lr), trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32))

        self.ready()

        with tf.variable_scope("optimizer"):
            opt = tf.train.AdagradOptimizer(learning_rate=self.lr)
            grads = opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(gradients, config.grad_clip)
            self.train_op = opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        d = config.hidden

        batch_size = tf.shape(self.sent)[0]
        sent_mask = tf.cast(self.sent, tf.bool)
        sent_len = tf.reduce_sum(tf.cast(sent_mask, tf.int32), axis=1)
        sent_maxlen = tf.reduce_max(sent_len)
        sent_mask = tf.slice(sent_mask, [0, 0], [batch_size, sent_maxlen])
        sent = tf.slice(self.sent, [0, 0], [batch_size, sent_maxlen])

        mid_mask = tf.cast(self.mid, tf.bool)
        mid_len = tf.reduce_sum(tf.cast(mid_mask, tf.int32), axis=1)
        mid_maxlen = tf.reduce_max(mid_len)
        mid_mask = tf.slice(mid_mask, [0, 0], [batch_size, mid_maxlen])
        mid = tf.slice(self.mid, [0, 0], [batch_size, mid_maxlen])

        pat_mask = tf.cast(self.pats, tf.bool)
        pat_len = tf.reduce_sum(tf.cast(pat_mask, tf.int32), axis=1)

        with tf.variable_scope("embedding"):
            sent_emb = tf.nn.embedding_lookup(self.word_mat, sent)
            mid_emb = tf.nn.embedding_lookup(self.word_mat, mid)
            sent_emb = dropout(sent_emb, keep_prob=config.word_keep_prob, is_train=self.is_train, mode="embedding")
            pat_emb = tf.nn.embedding_lookup(self.word_mat, self.pats)

        with tf.variable_scope("encoder"):
            rnn = Cudnn_RNN(num_layers=2, num_units=d // 2)
            cont, _ = rnn(sent_emb, seq_len=sent_len, concat_layers=False)
            pat, _ = rnn(pat_emb, seq_len=pat_len, concat_layers=False)

            cont_d = dropout(cont, keep_prob=config.keep_prob, is_train=self.is_train)
            pat_d = dropout(pat, keep_prob=config.keep_prob, is_train=self.is_train)

        with tf.variable_scope("attention"):
            att_a = attention(cont_d, config.att_hidden, mask=sent_mask)
            pat_a = self.pat_a = attention(pat_d, config.att_hidden, mask=pat_mask)

        with tf.variable_scope("sim"):
            sim, pat_sim = att_match(mid_emb, pat_emb, mid_mask, pat_mask, d, keep_prob=config.keep_prob, is_train=self.is_train)

            neg_idxs = tf.matmul(self.rels, tf.transpose(self.rels, [1, 0]))
            pat_pos = tf.square(tf.maximum(config.tau - pat_sim, 0.))
            pat_pos = tf.reduce_max(pat_pos - (1 - neg_idxs) * 1e30, axis=1)
            pat_neg = tf.square(tf.maximum(pat_sim, 0.))
            pat_neg = tf.reduce_max(pat_neg - 1e30 * neg_idxs, axis=1)
            l_sim = tf.reduce_sum(self.weight * (pat_pos + pat_neg), axis=0)

            with tf.variable_scope("pred"):
                att2_d = tf.reduce_sum(tf.expand_dims(att_a, axis=2) * cont_d, axis=1)
                pat2_d = tf.reduce_sum(tf.expand_dims(pat_a, axis=2) * pat_d, axis=1)

                logit = self.logit = dense(att2_d, config.num_class, use_bias=False)
                pred = tf.nn.softmax(logit)
                l_a = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit[:config.batch_size], labels=self.rel[:config.batch_size]), axis=0)

                xsim = tf.stop_gradient(sim[config.batch_size:])
                pseudo_rel = tf.gather(self.rels, tf.argmax(xsim, axis=1))
                bound = tf.reduce_max(xsim, axis=1)
                weight = tf.nn.softmax(10 * bound)
                l_u = tf.reduce_sum(weight * tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logit[config.batch_size:], labels=pseudo_rel), axis=0)

                logit = dense(pat2_d, config.num_class, use_bias=False)
                l_pat = self.pat_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=self.rels), axis=0)

        self.max_val = tf.reduce_sum(pred * -log(pred), axis=1)
        self.pred = tf.argmax(pred, axis=1)

        self.loss = l_a + config.alpha * l_pat + config.beta * l_sim + config.gamma * l_u
        self.sim_pred = tf.argmax(tf.gather(self.rels, tf.argmax(self.sim, axis=1)), axis=1)
        self.sim_max_val = tf.reduce_max(self.sim, axis=1)
        self.gold = tf.argmax(self.rel, axis=1)
        self.max_logit = tf.reduce_max(self.logit, axis=1)

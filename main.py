import tensorflow as tf
import ujson as json
import numpy as np
import sys
import random
from tqdm import tqdm
from collections import Counter
from util import get_batch, get_feeddict, get_patterns, merge_batch

from models.pat_match import Pat_Match
from models.soft_match import Soft_Match

tqdm.monitor_interval = 0
np.set_printoptions(threshold=np.nan)


def read(config):

    def _read(path, dataset):
        if dataset == "tacred":
            with open(path, "r") as fh:
                return json.load(fh)
        elif dataset == "semeval":
            res = []
            with open(path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if len(line) > 0:
                        d = json.loads(line)
                        res.append(d)
            return res

    train_data = _read(config.train_file, config.dataset)
    dev_data = _read(config.dev_file, config.dataset)
    test_data = _read(config.test_file, config.dataset)
    if config.dataset == "tacred":
        from tacred_loader import read_glove, get_counter, token2id, read_data
    else:
        from semeval_loader import read_glove, get_counter, token2id, read_data
    counter = get_counter(train_data)

    emb_dict = read_glove(config.glove_word_file, counter, config.glove_word_size, config.glove_dim)
    with open(config.emb_dict, "w") as fh:
        json.dump(emb_dict, fh)
    word2idx_dict, word_emb = token2id(config, counter, emb_dict)

    train_data = read_data(train_data)
    dev_data = read_data(dev_data)
    test_data = read_data(test_data)
    return word2idx_dict, word_emb, train_data, dev_data, test_data


def train(config, data):
    word2idx_dict, word_emb, train_data, dev_data, test_data = data
    patterns = get_patterns(config, word2idx_dict)

    with tf.variable_scope("models"):
        if config.dataset == "tacred":
            import tacred_constant as constant
        else:
            import semeval_constant as constant
        regex = Pat_Match(config, constant.LABEL_TO_ID)
        match = Soft_Match(config, word_mat=word_emb, word2idx_dict=word2idx_dict)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    labeled_data = []
    unlabeled_data = []
    for x in train_data:
        batch = [x["tokens"]]
        res, pred = regex.match(batch)
        patterns["weights"] += res[0]
        if np.amax(res) > 0:
            x["rel"] = pred.tolist()[0]
            x["pat"] = np.argmax(res, axis=1).tolist()[0]
            labeled_data.append(x)
        else:
            x["rel"] = 0
            unlabeled_data.append(x)
    patterns["weights"] = patterns["weights"] / np.sum(patterns["weights"])
    random.shuffle(unlabeled_data)
    print("{} labeled data".format(len(labeled_data)))

    dev_history, test_history = [], []

    with tf.Session(config=sess_config) as sess:

        lr = float(config.init_lr)
        sess.run(tf.global_variables_initializer())

        for epoch in tqdm(range(1, config.num_epoch + 1), desc="Epoch"):
            for batch1, batch2 in zip(get_batch(config, labeled_data, word2idx_dict), get_batch(config, unlabeled_data, word2idx_dict, pseudo=True)):
                batch = merge_batch(batch1, batch2)
                loss, _ = sess.run([match.loss, match.train_op], feed_dict=get_feeddict(match, batch, patterns))

            (dev_acc, dev_rec, dev_f1), best_entro = log(config, dev_data, patterns, word2idx_dict, match, sess, "dev")
            (test_acc, test_rec, test_f1), _ = log(
                config, test_data, patterns, word2idx_dict, match, sess, "test", entropy=best_entro)

            dev_history.append((dev_acc, dev_rec, dev_f1))
            test_history.append((test_acc, test_rec, test_f1))
            if len(dev_history) >= 1 and dev_f1 <= dev_history[-1][2]:
                lr *= config.lr_decay
                sess.run(tf.assign(match.lr, lr))

        max_idx = dev_history.index(max(dev_history, key=lambda x: x[2]))
        max_acc, max_rec, max_f1 = test_history[max_idx]
        print("acc: {}, rec: {}, f1: {}".format(max_acc, max_rec, max_f1))
        sys.stdout.flush()
    return max_acc, max_rec, max_f1


def log(config, data, patterns, word2idx_dict, model, sess, label="train", entropy=None):
    golds, preds, vals, sim_preds, sim_vals = [], [], [], [], []
    for batch in get_batch(config, data, word2idx_dict):
        gold, pred, val, sim_pred, sim_val = sess.run([model.gold, model.pred, model.max_val, model.sim_pred, model.sim_max_val],
                                                      feed_dict=get_feeddict(model, batch, patterns, is_train=False))
        golds += gold.tolist()
        preds += pred.tolist()
        vals += val.tolist()
        sim_preds += sim_pred.tolist()
        sim_vals += sim_val.tolist()

    threshold = [0.01 * i for i in range(1, 200)]
    acc, recall, f1 = 0., 0., 0.
    best_entro = 0.

    if entropy is None:
        for t in threshold:
            _preds = (np.asarray(vals, dtype=np.float32) <= t).astype(np.int32) * np.asarray(preds, dtype=np.int32)
            _preds = _preds.tolist()
            _acc, _recall, _f1 = evaluate(golds, _preds)
            if _f1 > f1:
                acc, recall, f1 = _acc, _recall, _f1
                best_entro = t
    else:
        preds = (np.asarray(vals, dtype=np.float32) <= entropy).astype(np.int32) * np.asarray(preds, dtype=np.int32)
        preds = preds.tolist()
        acc, recall, f1 = evaluate(golds, preds)
    return (acc, recall, f1), best_entro


def evaluate(key, prediction):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == 0 and guess == 0:
            pass
        elif gold == 0 and guess != 0:
            guessed_by_relation[guess] += 1
        elif gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        elif gold != 0 and guess != 0:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro

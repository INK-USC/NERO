import random
import numpy as np
import json


def get_pos(start, end, length, pad=None):
    res = list(range(-start, 0)) + [0] * (end - start + 1) + list(range(length - end - 1))
    if pad is not None:
        res += [0 for _ in range(pad - len(res))]
    return res


def get_mask(start, end, length, pad=None):
    res = [0] * start + [1] * (end - start + 1) + [0] * (length - end - 1)
    if pad is not None:
        res += [0 for _ in range(pad - len(res))]
    return res


def get_word(tokens, word2idx_dict, pad=None):
    res = []
    for token in tokens:
        i = 1
        for each in (token, token.lower(), token.capitalize(), token.upper()):
            if each in word2idx_dict:
                i = word2idx_dict[each]
        res.append(i)
    if pad is not None:
        res = res[:pad]
        res += [0 for _ in range(pad - len(res))]
    return res


def get_id(tokens, idx_dict, pad=None):
    if not isinstance(tokens, list):
        tokens = [tokens]
    res = [idx_dict[token] if token in idx_dict else 1 for token in tokens]
    if pad is not None:
        res += [0 for _ in range(pad - len(res))]
    return res


def get_patterns(config, word2idx_dict, filt=None):
    if config.dataset == "tacred":
        import tacred_constant as constant
    else:
        import semeval_constant as constant
    patterns = config.patterns
    rels, pats = [], []
    for pattern in patterns:
        rel, pat = pattern
        rel_id = constant.LABEL_TO_ID[rel]
        if filt is not None:
            if rel_id in filt:
                continue
        rel = [0. for _ in range(config.num_class)]
        rel[rel_id] = 1.

        pat = pat.split()

        pat = get_word(pat, word2idx_dict, pad=10)
        rels.append(rel)
        pats.append(pat)
    num_pats = len(rels)
    rels = np.asarray(rels, dtype=np.float32)
    pats = np.asarray(pats, dtype=np.int32)
    weights = np.ones([num_pats], dtype=np.float32)
    return {"rels": rels, "pats": pats, "weights": weights}


def get_feeddict(model, batch, patterns, is_train=True):
    return {model.sent: batch["sent"], model.rel: batch["rel"], model.mid: batch["mid"],
            model.hard: batch["pat"], model.rels: patterns["rels"], model.pats: patterns["pats"],
            model.weight: patterns["weights"], model.is_train: is_train}


def get_batch(config, data, word2idx_dict, rel_dict=None, shuffle=True, pseudo=False):
    if shuffle:
        random.shuffle(data)
    batch_size = config.pseudo_size if pseudo else config.batch_size
    length = config.length
    for i in range(len(data) // batch_size):
        batch = data[i * batch_size: (i + 1) * batch_size]
        raw = list(map(lambda x: x["tokens"], batch))
        sent = np.asarray(list(map(lambda x: get_word(x["tokens"], word2idx_dict, pad=length), batch)), dtype=np.int32)
        mid = np.asarray(list(map(lambda x: get_word(x["tokens"][x["start"] - 1: x["end"] + 2], word2idx_dict, pad=length), batch)), dtype=np.int32)
        rel = np.asarray(list(map(lambda x: [1.0 if i == x["rel"] else 0. for i in range(config.num_class)], batch)), dtype=np.float32)
        pat = np.asarray(list(map(lambda x: x["pat"], batch)), dtype=np.int32)
        yield {"sent": sent, "mid": mid, "rel": rel, "raw": raw, "pat": pat}


def merge_batch(batch1, batch2):
    batch = {}
    for key in batch1.keys():
        val1 = batch1[key]
        val2 = batch2[key]
        val = np.concatenate([val1, val2], axis=0)
        batch[key] = val
    return batch


def sample_data(config, data):
    random.shuffle(data)
    num = len(data)
    return data[:int(config.sample * num)]

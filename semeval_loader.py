from tqdm import tqdm
from collections import Counter
import numpy as np
import ujson as json
import copy
import semeval_constant as constant


def entity_masks():
    masks = ["SUBJ-O", "OBJ-O"]
    subj_entities = list(constant.SUBJ_NER_TO_ID.keys())
    obj_entities = list(constant.OBJ_NER_TO_ID.keys())
    masks += ["SUBJ-" + e for e in subj_entities]
    masks += ["OBJ-" + e for e in obj_entities]
    return masks


def read_glove(path, counter, size, dim):
    embedding_dict = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, total=size):
            array = line.split()
            word = "".join(array[0:-dim])
            vector = list(map(float, array[-dim:]))
            if word in counter:
                embedding_dict[word] = vector
    return embedding_dict


def token2id(config, counter, embedding_dict):
    vec_size = len(list(embedding_dict.values())[0])
    masks = entity_masks()
    token2idx_dict = {}
    token2idx_dict[constant.PAD_TOKEN] = constant.PAD_ID
    token2idx_dict[constant.UNK_TOKEN] = constant.UNK_ID
    embedding_dict[constant.PAD_TOKEN] = [0. for _ in range(vec_size)]
    embedding_dict[constant.UNK_TOKEN] = [0. for _ in range(vec_size)]
    for token in list(embedding_dict.keys()) + masks:
        if token not in token2idx_dict:
            token2idx_dict[token] = len(token2idx_dict)
        if token not in embedding_dict:
            embedding_dict[token] = [np.random.normal(-1., 1.) for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    word_emb = np.array([idx2emb_dict[idx] for idx in range(len(token2idx_dict))], dtype=np.float32)
    return token2idx_dict, word_emb


def get_counter(data):
    counter = Counter()
    for d in data:
        tokens = d["tokens"]
        for token in tokens:
            counter[token] += 1
    return counter


def read_data(data):
    examples = []
    for d in data:
        tokens = d["tokens"]
        ss, se = d["subj_start"], d["subj_end"]
        os, oe = d["obj_start"], d["obj_end"]
        rel = constant.LABEL_TO_ID[d['relation']]

        start = min(oe, se) + 1
        end = max(os, ss) - 1
        examples.append({"tokens": tokens, "start": start, "end": end, "rel": rel, "pat": -1})
    return examples

import os
import tensorflow as tf
import tacred_constant as constant
import json

from main import train, read


flags = tf.flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


flags.DEFINE_string("dataset", "tacred", "")
flags.DEFINE_string("mode", "regd", "pretrain / pseudo / regd")
flags.DEFINE_string("gpu", "0", "The GPU to run on")


flags.DEFINE_string("pattern_file", "./data/tacred/pattern.json", "")
flags.DEFINE_string("target_dir", "data", "")
flags.DEFINE_string("log_dir", "log/event", "")
flags.DEFINE_string("save_dir", "log/model", "")
flags.DEFINE_string("glove_word_file", "./data/glove/glove.840B.300d.txt", "")

flags.DEFINE_string("train_file", "./data/tacred/train.json", "")
flags.DEFINE_string("dev_file", "./data/tacred/dev.json", "")
flags.DEFINE_string("test_file", "./data/tacred/test.json", "")
flags.DEFINE_string("emb_dict", "./data/tacred/emb_dict.json", "")

flags.DEFINE_integer("glove_word_size", int(2.2e6), "Corpus size for Glove")
flags.DEFINE_integer("glove_dim", 300, "Embedding dimension for Glove")
flags.DEFINE_integer("top_k", 100000, "Finetune top k words in embedding")
flags.DEFINE_integer("length", 110, "Limit length for sentence")
flags.DEFINE_integer("num_class", len(constant.LABEL_TO_ID), "Number of classes")
flags.DEFINE_string("tag", "", "The tag name of event files")

flags.DEFINE_integer("batch_size", 50, "Batch size")
flags.DEFINE_integer("pseudo_size", 100, "Batch size for pseudo labeling")
flags.DEFINE_integer("num_epoch", 50, "Number of epochs")
flags.DEFINE_float("init_lr", 0.5, "Initial lr")
flags.DEFINE_float("lr_decay", 0.95, "Decay rate")
flags.DEFINE_float("keep_prob", 0.5, "Keep prob in dropout")
flags.DEFINE_float("grad_clip", 5.0, "Global Norm gradient clipping rate")
flags.DEFINE_integer("hidden", 200, "Hidden size")
flags.DEFINE_integer("att_hidden", 200, "Hidden size for attention")

flags.DEFINE_float("alpha", 1.0, "Weight of pattern RE")
flags.DEFINE_float("beta", 0.2, "Weight of similarity score")
flags.DEFINE_float("gamma", 0.5, "Weight of pseudo label")
flags.DEFINE_float("tau", 0.7, "Weight of tau")
flags.DEFINE_list("patterns", [], "pattern list")


def main(_):
    config = flags.FLAGS
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    with open(config.pattern_file, "r") as fh:
        patterns = json.load(fh)
    config.patterns = patterns
    data = read(config)
    train(config, data)


if __name__ == "__main__":
    tf.app.run()

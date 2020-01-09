# NERO
Code for DeepLo 2019 paper [Neural Rule Grounding for Low-Resource Relation Extraction](https://arxiv.org/abs/1909.02177).

While deep neural models have gained successes on information extraction tasks, they become less reliable when the amount of labeled data is limited. Thus, we propose to annotate frequent surface patterns to form labeling rules. These rules can be automatically mined from large text corpora and generalized via a soft rule matching mechanism. In this paper, we present a neural approach to ground rules for relation extraction, named NERO, which jointly learns a relation extraction module and a soft matching module to use these rules. The soft matching module extends the coverage of rules on semantically similar instances and augments the learning on unlabeled corpus. Experiments on two public datasets (TACRED and SemEval 2010) demonstrate the ectiveness of NERO when compared with both rule-based and semi-supervised baselines. Additionally, the learned soft matching module is able to predict on new relations with unseen rules, and can provide interpretation on matching results. With extensive user study, we find that the time effiency for a human to annotate rules and sentences are similar (0.30 vs. 0.35 min per label). Particularly, NERO’s performance using 270 rules is comparable to the models trained using 3,000 labeled sentences, yielding a 9.5x speedup.

<p align="center"><img src="figs/REGD.jpg" width="800"/></p>

If you make use of this code or the rules in your work, please cite the following paper:

```bibtex
@article{zhou2019neural,
  title={Neural Rule Grounding for Low-Resource Relation Extraction},
  author={Zhou, Wenxuan and Lin, Hongtao and Lin, Bill Yuchen and Wang, Ziqi and Du, Junyi and Neves, Leonardo and Ren, Xiang},
  journal={arXiv preprint arXiv:1909.02177},
  year={2019}
}
```

## Quick Links
* [Requirements](#requirements)
* [Motivation](#motivation)
* [Rules](#rules)
* [Train and Test](#train-and-test)

## Motivation
Supervised neural models yield state-of-the-art results on relation extraction task, but their performance has heavy reliance on sufficient training labels. To alliveate the problem, recent works (e.g. [Stanford Snorkel](https://www.snorkel.org/)) propose to construct large labeled dataset from labeling rules. They perform exact string matching on unlabeled dataset, and a sentence is either matched or not matched by a rule. However, this hard-matching method fails to annotate sentences with similar meanings but different words, which consequently cause the low-recall problem and data-insufficiency for training neural networks. In this paper, we argue that the rule matching should not be performed solely based on the surface forms (strings), but also on the semantic meanings. We measure the similarity between sentences and rules with matching scores that are calculated on their neural representations (e.g. word embedding), and label each sentence with its most similar rule.
<p align="center"><img src="figs/rule_example.jpg" width="400"/></p>

## Rules
Labeling rules can either be crafted by domain experts, or automatically mines from large corpora with a knowledge base. In this work, we adopt a hybrid apporach. We first extract the frequent patterns from large raw corpora, then ask human annotators to assign labels to the patterns. In this way, we get 270 rules on TACRED dataset (located in``/data/tacred/pattern.json``) and 164 rules on SemEval 2010 dataset (located in``/data/semeval/pattern.json``). 


## Requirements
Tensorflow-gpu == 1.10 \
tqdm \
ujson

## Train and Test
Before running, download dataset from [TACRED](https://nlp.stanford.edu/projects/tacred/) and [SemEval](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50) and put them under ``/data``. Also download [Glove embedding](http://nlp.stanford.edu/data/glove.840B.300d.zip) and unzip it to ``/data/glove/glove.840B.300d.txt``.

To train the relation extraction model on TACRED / SemEval dataset, run

```bash
python3 tacred.py / semeval.py
```

The model will be automatically evaluated on dev and test dataset after each training epoch. After training, the code will choose the model that achieves best performance on dev dataset and return its test score.

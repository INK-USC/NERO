# REGD
Code for DeepLo 2019 paper [Neural Rule Grounding for Low-Resource Relation Extraction](https://arxiv.org/abs/1909.02177).

While deep neural models have gained successes on information extraction tasks, they become less reliable when the amount of labeled data is limited. In this paper, we study relation extraction (RE) under low-resource setting, where only some (hand-built) labeling rules are provided for learning a neural model over a large, unlabeled corpus. To overcome the low-coverage issue of current bootstrapping methods (i.e., hard grounding of rules), we propose a Neural Rule Grounding (REGD) framework for jointly learning a relation extraction module (with flexible neural architecture) and a sentence-rule soft matching module. The soft matching module extends the coverage of rules on semantically similar instances and augments the learning on unlabeled corpus. Experiments on two public datasets demonstrate the effectiveness of REGD when compared with both rule-based and semi-supervised baselines. Additionally, the learned soft matching module is able to predict on new relations with unseen rules, and can provide interpretation on matching results.

<p align="center"><img src="figs/REGD.jpg" width="800"/></p>

If you make use of this code or the rule dataset in your work, please cite the following paper:

```bibtex
@article{zhou2019neural,
  title={Neural Rule Grounding for Low-Resource Relation Extraction},
  author={Zhou, Wenxuan and Lin, Hongtao and Wang, Ziqi and Neves, Leonardo and Ren, Xiang},
  journal={arXiv preprint arXiv:1909.02177},
  year={2019}
}
```


## Requirements
Tensorflow-gpu == 1.10 \
tqdm \
ujson

## Train
Before running, download dataset from [TACRED](https://nlp.stanford.edu/projects/tacred/) and [SemEval](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50). Also download [Glove embedding](http://nlp.stanford.edu/data/glove.840B.300d.zip) and unzip it to ``/data/glove/glove.840B.300d.txt``.

To train the relation extraction model on TACRED / SemEval dataset, run

```bash
python3 tacred.py / semeval.py
```

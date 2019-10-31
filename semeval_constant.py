"""
From Github repo https://github.com/yuhaozhang/tacred-relation.git. No modification made.
"""

"""
Define common constants.
"""
TRAIN_JSON = 'train.json'
DEV_JSON = 'dev.json'
TEST_JSON = 'test.json'

GLOVE_DIR = 'dataset/glove'

EMB_INIT_RANGE = 1.0
MAX_LEN = 100

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9,
                 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6,
             'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22,
             '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21,
                'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

LABEL_TO_ID = {'no_relation': 0, 'Cause-Effect(e1,e2)': 1, 'Cause-Effect(e2,e1)': 2, 'Component-Whole(e1,e2)': 3, 'Component-Whole(e2,e1)': 4, 'Content-Container(e1,e2)': 5, 'Content-Container(e2,e1)': 6, 'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8, 'Entity-Origin(e1,e2)': 9,
               'Entity-Origin(e2,e1)': 10, 'Instrument-Agency(e1,e2)': 11, 'Instrument-Agency(e2,e1)': 12, 'Member-Collection(e1,e2)': 13, 'Member-Collection(e2,e1)': 14, 'Message-Topic(e1,e2)': 15, 'Message-Topic(e2,e1)': 16, 'Product-Producer(e1,e2)': 17, 'Product-Producer(e2,e1)': 18}

ID_TO_LABEL = {0: 'no_relation', 1: 'Cause-Effect(e1,e2)', 2: 'Cause-Effect(e2,e1)', 3: 'Component-Whole(e1,e2)', 4: 'Component-Whole(e2,e1)', 5: 'Content-Container(e1,e2)', 6: 'Content-Container(e2,e1)', 7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)', 9: 'Entity-Origin(e1,e2)',
               10: 'Entity-Origin(e2,e1)', 11: 'Instrument-Agency(e1,e2)', 12: 'Instrument-Agency(e2,e1)', 13: 'Member-Collection(e1,e2)', 14: 'Member-Collection(e2,e1)', 15: 'Message-Topic(e1,e2)', 16: 'Message-Topic(e2,e1)', 17: 'Product-Producer(e1,e2)', 18: 'Product-Producer(e2,e1)'}
INFINITY_NUMBER = 1e12

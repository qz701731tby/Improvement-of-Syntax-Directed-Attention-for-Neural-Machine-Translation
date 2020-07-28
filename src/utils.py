# coding: utf-8

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(data_path, lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    #lines = open(data_path + '%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    lines = open(data_path + 'syntax_info_train_data.txt', encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    # pairs = [[s for s in l.split('\t')] for l in lines]
    pairs = []
    for i in range(0, len(lines), 2):
        sentence_pair = lines[i].split('\t')
        syntax_matrix = eval(lines[i+1])
        pairs.append([sentence_pair, syntax_matrix])
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p[0])) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    flag = True
    if p[0][-6:] == ' æ—¥ : .' or p[0][-4:] == ' æ—¥ .':
        flag = False
    return flag

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair[0])]

def prepareData(data_path, lang1, lang2, reverse=False):
    max_length = 0
    input_lang, output_lang, pairs = readLangs(data_path, lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0][0])
        output_lang.addSentence(pair[0][1])
        input_length = len(pair[0][0].split(' '))
        max_length = max(max_length, input_length)
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, max_length

if __name__ == "__main__":
    data_path = '/content/drive/My Drive/colab qz_seq2seq/data/'
    input_lang, output_lang, pairs, max_length = prepareData(data_path, 'zh', 'en', False)
    print(random.choice(pairs))
    print(pairs[:10])
    print(max_length)

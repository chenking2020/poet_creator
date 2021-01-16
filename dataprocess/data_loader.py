import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import codecs
import math
import random
import re
import numpy as np
from tqdm import tqdm
import importlib
import json


def init_processor(processor_name):
    global processor_module
    processor_module = importlib.import_module('write_provider.dataprocess.process_{}'.format(processor_name))


def load_data(path):
    sentences = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num += 1
        line = line.rstrip()
        if len(line) > 0:
            sentences.append(line)
    random.shuffle(sentences)
    return processor_module.extract_all_features(sentences)


def word_mapping(sentences, word_pretrain_type, word_pretrain_path):
    if word_pretrain_type is None or len(word_pretrain_type) == 0:
        word_pretrain_type = "pretrain_default"
    word_processor_module = importlib.import_module(
        'write_provider.dataprocess.process_{}'.format(word_pretrain_type))
    _, word_to_id, id_to_word = word_processor_module.word_mapping(sentences, word_pretrain_path)
    _, py_to_id, id_to_py = word_processor_module.pinyin_mapping(sentences, word_pretrain_path)
    _, ton_to_id, id_to_ton = word_processor_module.ton_mapping(sentences, word_pretrain_path)
    return word_to_id, id_to_word, py_to_id, id_to_py, ton_to_id, id_to_ton


def prepare_dataset(sentences, word_to_id, py_to_id, ton_to_id, word_pretrain_type, train=True):
    if word_pretrain_type is None or len(word_pretrain_type) == 0:
        word_pretrain_type = "pretrain_default"
    word_processor_module = importlib.import_module(
        'write_provider.dataprocess.process_{}'.format(word_pretrain_type))

    data = []
    for sentence in sentences:
        words = word_processor_module.sentence_to_word_seq(word_to_id, sentence[0])
        pinyins = word_processor_module.sentence_to_pinyin_seq(py_to_id, sentence[1])
        tons = word_processor_module.sentence_to_ton_seq(ton_to_id, sentence[2])
        tags = word_processor_module.sentence_to_word_seq(word_to_id, sentence[3])
        data.append([words, pinyins, tons, tags])

    return data


class BatchManager(object):
    def __init__(self, data, batch_size, is_sorted=True):
        self.batch_data = self.sort_and_pad(data, batch_size, is_sorted)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size, is_sorted):
        num_batch = int(math.ceil(len(data) / batch_size))
        if is_sorted:
            sorted_data = sorted(data, key=lambda x: len(x[0]))
        else:
            sorted_data = data
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        words = []
        pinyins = []
        tons = []
        targets = []
        max_word_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            word_seq, py_seq, ton_seq, target = line
            word_padding = [0] * (max_word_length - len(word_seq))
            py_padding = [0] * (max_word_length - len(py_seq))
            ton_padding = [0] * (max_word_length - len(ton_seq))
            target_padding = [0] * (max_word_length - len(target))
            words.append(word_seq + word_padding)
            pinyins.append(py_seq + py_padding)
            tons.append(ton_seq + ton_padding)
            targets.append(target + target_padding)
        return [words, pinyins, tons, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]

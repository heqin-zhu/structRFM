import collections
import argparse
import numpy as np
import json

import torch


def get_config(file_path):
    from attrdict import AttrDict
    config_file = file_path  # "./weights/bert_config.json"
    json_file = open(config_file, 'r')
    json_object = json.load(json_file)
    config = AttrDict(json_object)
    return config


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def str2list(v):
    if isinstance(v, list):
        return v
    elif isinstance(v, str):
        vs = v.split(",")
        return [v.strip() for v in vs]
    else:
        raise argparse.ArgumentTypeError(
            "Str value seperated by ', ' expected.")


def seq2kmer(seq, kmer=1):
    kmer_text = ""
    i = 0
    while i < len(seq):
        kmer_text += (seq[i: i + 1] + " ")
        i += 1
    kmer_text = kmer_text.strip()
    return kmer_text


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class Stack(object):

    def __init__(self, axis=0, dtype=None):
        self._axis = axis
        self._dtype = dtype

    def __call__(self, data):
        data = np.stack(data, axis=self._axis).astype(
            self._dtype) if self._dtype else np.stack(data, axis=self._axis)
        return data

def get_and_set_device(device):
    if device:
        torch.cuda.set_device(device)
        return device
    else:
        if torch.cuda.is_available():
            current_device_id = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device_id)
            print(f"GPU {current_device_id}: {device_name}")
            return torch.device(f"cuda:{current_device_id}")
        else:
            print("No GPU，using CPU")
            return torch.device("cpu")

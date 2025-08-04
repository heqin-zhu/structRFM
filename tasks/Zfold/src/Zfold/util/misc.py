import os
import numpy as np
import string


def parse_a3m(filename, limit=20000, rm_query_gap=True):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    n = 0
    for line in open(filename, "r"):
        if line[0] != '>' and len(line.strip()) > 0:
            seqs.append(
                line.rstrip().replace('W', 'A').replace('R', 'A').replace('Y', 'C').replace('E', 'A').replace('I',
                                                                                                              'A').replace(
                    'P', 'G').replace('T', 'U').translate(table))
            n += 1
            if n == limit:
                break

    # convert letters into numbers
    alphabet = np.array(list("AUCG-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

        # treat all unknown characters as gaps
    msa[msa > 4] = 4
    if rm_query_gap:
        return msa[:, msa[0] < 4]
    return msa


def ss2mat(ss_seq):
    ss_mat = np.zeros((len(ss_seq), len(ss_seq)))
    stack = []
    stack1 = []
    stack2 = []
    stack3 = []
    stack_alpha = {alpha: [] for alpha in string.ascii_lowercase}
    for i, s in enumerate(ss_seq):
        if s == '(':
            stack.append(i)
        elif s == ')':
            ss_mat[i, stack.pop()] = 1
        elif s == '[':
            stack1.append(i)
        elif s == ']':
            ss_mat[i, stack1.pop()] = 1
        elif s == '{':
            stack2.append(i)
        elif s == '}':
            ss_mat[i, stack2.pop()] = 1
        elif s == '<':
            stack3.append(i)
        elif s == '>':
            ss_mat[i, stack3.pop()] = 1
        elif s.isalpha() and s.isupper():
            stack_alpha[s.lower()].append(i)
        elif s.isalpha() and s.islower():
            ss_mat[i, stack_alpha[s].pop()] = 1
        elif s in ['.', ',', '_', ':', '-']:
            continue
        else:
            raise ValueError(f'unk not: {s}!')
    allstacks = stack + stack1 + stack2 + stack3
    for _, stack in stack_alpha.items():
        allstacks += stack
    if len(allstacks) > 0:
        raise ValueError('Provided dot-bracket notation is not completely matched!')

    ss_mat += ss_mat.T
    return ss_mat


def parse_ct(ct_file, length=None):
    seq_ct = ''
    if length is None:
        length = int(open(ct_file).readlines()[0].split()[0])
    mat = np.zeros((length, length))
    for line in open(ct_file):
        items = line.split()
        if len(items) >= 6 and items[0].isnumeric() and items[2].isnumeric() and items[3].isnumeric() and items[
            4].isnumeric():
            seq_ct += items[1]
            if int(items[4]) > 0:
                mat[int(items[4]) - 1, int(items[5]) - 1] = 1
                mat[int(items[5]) - 1, int(items[4]) - 1] = 1
    return mat

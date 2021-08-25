from __future__ import division
from torch.autograd import Variable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_cfg(cfgfile):
    '''
    :param cfgfile: configuration file
    :return: a list of blocks. blocks are represented as a dict
    '''
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    # rid empty lines
    lines = [x for x in lines if len(x) > 0]
    # rid comments
    lines = [x for x in lines if x[0] != '#']
    # rid whitespaces
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

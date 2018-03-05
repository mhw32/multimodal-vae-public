from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import string
import random
import time
import math
import torch
from torch.autograd import Variable

max_length = 4  # max of 4 characters in an image
all_characters = '0123456789'
n_characters = len(all_characters)
# add 2 characters; b/c we always generate a fixed number
# of characters, we do not need an EOS token
SOS = n_characters
FILL = n_characters + 1  # placeholder for nothing
n_characters += 2


def char_tensor(string):
    """Turn a string into a tensor.

    @param string: str object
    @return tensor: torch.Tensor object. Not a Variable.
    """
    tensor = torch.ones(max_length).long() * FILL
    for c in xrange(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor


def charlist_tensor(charlist):
    """Turn a list of indexes into a tensor."""
    string = ''.join([str(i) for i in charlist])
    return char_tensor(string)
    

def tensor_to_string(tensor):
    """Identical to tensor_to_string but for LongTensors."""
    string = ''
    for i in range(tensor.size(0)):
        top_i = tensor[i]
        string += index_to_char(top_i)
    return string


def index_to_char(top_i):
    if top_i == SOS:
        return '^'
    # FILL is the default character
    elif top_i == FILL:
        return ''
    else:
        return all_characters[top_i]

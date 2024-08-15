# utils

import re
from collections import defaultdict, UserDict

# get batch indices
def batch_indices(length, batch_size):
    return [(i, min(i+batch_size, length)) for i in range(0, length, batch_size)]

# split text into chunks
def list_splitter(text, maxlen):
    for i, j in batch_indices(len(text), maxlen):
        yield text[i:j]

def glob_strings(pattern, strings):
    # make regex pattern
    reg = '^' + pattern.replace('.', '\\.').replace('*', '.*') + '$'

    # get matching objects
    matches = [re.match(reg, s) for s in strings]

    # return matching strings
    return [m.string for m in matches if m is not None]

# dictionary that allows getattr access
# only allows string keys for correctness
class AttrDict(UserDict):
    def __setitem__(self, key, val):
        # key type validation
        if type(key) is not str:
            raise ValueError('Only string keys allowed')
        elif '*' in key:
            raise ValueError('Wildcards (*) not allowed in keys')

        # get the key
        super().__setitem__(key, val)

    def __getitem__(self, key):
        # handle tuple accessor case (prevent recursion)
        if type(key) is tuple:
            return [super().__getitem__(k) for k in key]

        # key type validation
        if type(key) is not str:
            raise ValueError('Only string keys allowed')

        # get the key
        return super().__getitem__(key)

    # allows globbing with *
    def subset(self, keys):
        # expand wildcards to lists
        if type(keys) is str:
            keys = glob_strings(keys, self)

        # get subset dict
        return {k: self[k] for k in keys}

# = defaultdict(list)
# + handles popping off maximal list
# + handles deletion on empty list
class SizeDist(dict):
    def __init__(self, data):
        sdist = defaultdict(list)
        for i, size in enumerate(data):
            sdist[size].append(i)
        super().__init__(sdist)

    def pop(self, max_size=None):
        if max_size is None:
            size = max(self, default=None)
        else:
            size = max((s for s in self if s <= max_size), default=None)
        if size is None:
            return
        ret = self[size].pop(0)
        if len(self[size]) == 0:
            del self[size]
        return ret

def pack_batches(sizes, max_len):
    # get size distribution
    n_seq = len(sizes)
    sdist = SizeDist(sizes)
    assert max(sdist) <= max_len

    # plan batch contents
    batches = []
    bidxs = []
    bsize = 0
    for _ in range(n_seq):
        # get a maximal sample
        idx = sdist.pop(max_len-bsize)

        # if none we commit batch and retry
        if idx is None:
            batches.append(bidxs)
            bidxs = []
            bsize = 0
            idx = sdist.pop(max_len)

        # append to batch
        bidxs.append(idx)
        bsize += sizes[idx]

    # append final batch
    batches.append(bidxs)

    return batches

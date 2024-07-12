# utils

from collections import defaultdict

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

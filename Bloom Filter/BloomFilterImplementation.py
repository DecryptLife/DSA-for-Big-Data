import math
import random
import csv
import pandas as pd

from sklearn.utils import murmurhash3_32
from bitarray import bitarray
from random import shuffle

random.seed(6)
# implement a simple hash function
def hashfunc(m, key,seed):
    return murmurhash3_32(key,seed) % m


class BloomFilter():
    # primary constructor
    # n => expected number of keys to store
    # c => false positive rate
    def __init__(self, c, n):

        # storing the fp rate
        self.c = c

        # find the size of hash table R (m)
        self.m = math.ceil(self.getTableSize(n, c))
        # find the number of hash functions (K)
        self.k = math.ceil(self.getHashNum(n, self.m))

        # create a bit array of size m
        self.bit_array = bitarray(self.highest2(self.m))

        # set all elements initially 0
        self.bit_array.setall(0)

    def insert(self, key):

        # Adding elements to the filter
        for i in range(self.k):
            bit = hashfunc(len(self.bit_array), key, i)
            self.bit_array[bit] = True

    def test(self, key):

        for i in range(self.k):
            bit = hashfunc(len(self.bit_array), key, i)
            if (self.bit_array[bit] == False):
                return False
        return True

    def highest2(self,m):
        self.p = int(math.ceil(math.log2(m)))
        return int(pow(2,self.p))

    def getTableSize(self, n, c):
        return n * math.log(c) / math.log(0.618)

    def getHashNum(self, n, m):
        return m / n * math.log(2)


n = 10000
c_vals = [0.01, 0.001, 0.0001]

data = [random.randint(10000, 99999) for i in range(10000)]
test = [random.randint(1000, 5000) for i in range(1000)]

bloomf = []

for c in c_vals:
    bloomf.append(BloomFilter(c, n))
    for i in data:
        bloomf[len(bloomf) - 1].insert(i)

fp_count = [0, 0, 0,0,0]

for index, c in enumerate(c_vals):
    for i in test:
        if bloomf[index].test(i):
            fp_count[index] += 1
    print("For false positive rate {} the theoritical positive rate is {}".format(c, fp_count[index] / 1000))
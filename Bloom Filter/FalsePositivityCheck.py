import csv
import os
import math
import random
import string
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import murmurhash3_32
from bitarray import bitarray

filename = "/user-ct-test-collection-01.txt"
data2 = pd.read_csv(os.getcwd() + filename, sep="\t")
urllist = data2.ClickURL.dropna().unique()

random.seed()


def hashfunc(m, key, seed):
    return murmurhash3_32(key, seed) % m


class BloomFilter():

    def __init__(self, m, n):
        self.m = m
        self.n = n

        # calculate false positive
        self.fp = self.cal_fp()

        # find number of hash functions
        self.k = self.get_hashNum()

        # create bit array of size m
        self.bit_array = bitarray(m)

        # set all elements to 0
        self.bit_array.setall(0)

    def get_hashNum(self):
        return math.ceil(0.7 * (self.m / self.n))

    def cal_fp(self):
        return 0.618 ** (self.m / self.n);

    def getMemUse(self):
        return sys.getsizeof(self.bit_array)

    def insert(self, key):

        for i in range(self.k):
            bit = hashfunc(self.m, key, i)

            self.bit_array[bit] = True

    def check(self, key):
        for i in range(self.k):
            bit = hashfunc(self.m, key, i)

            if (self.bit_array[bit] == False):
                return False

        return True


bloomf = BloomFilter(100, 377871)

sampled_data = [urllist[random.randint(0, 377870)] for i in range(1000)]
sampled_test = [''.join(random.choices(string.ascii_lowercase, k=10)) for i in range(1000)]

for url in urllist:
    bloomf.insert(url)

# varying m finding fp rates
fp_values = []
m_values = []

ht = {}
for url in urllist:
    ht[url] = 1

for i in range(15, 25):
    m_val = 1 << i

    bloomf1 = BloomFilter(m_val, 377871)
    fp_values.append(bloomf1.cal_fp())
    m_values.append(bloomf1.getMemUse())
    print(
        "Memory usage of hashtable is {} and that of bloom filter is {}".format(sys.getsizeof(ht), bloomf1.getMemUse()))

plt.plot(m_values, fp_values)
plt.xlabel("Memory Size")
plt.ylabel("False positive rate")
plt.show()

# storing to hashtable

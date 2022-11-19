import itertools
import math
import os
import random
import sys

import pandas as pd
import numpy as np
from sklearn.utils import murmurhash3_32
import matplotlib.pyplot as plt

filename = "/user-ct-test-collection-01.txt"
data2 = pd.read_csv(os.getcwd() + filename, sep="\t")
queries = data2.Query.dropna()

Freq_100 = {}
Rand_100 = {}
dictionary = {}
Infreq_100 = {}

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])


# returns the top 100 frequently used queries
def freq_100(key_value_pairs):
    sorted_dictionary = sorted(key_value_pairs, reverse=True)
    for (value, key) in sorted_dictionary[0:100]:
        Freq_100[key] = value


# returns the least 100 frequently used queries
def infreq_100():
    sorted_tuples = sorted(dictionary.items(), key=lambda item: item[1])
    for (key, value) in sorted_tuples[0:100]:
        Infreq_100[key] = value


# stores 100 random queries from the dictionary
def rand_100(dictionary):
    key_vals = [(key, value) for key, value in dictionary.items()]
    rand_elements = random.sample(list(key_vals), 100)
    for key, value in rand_elements:
        Rand_100[key] = value


def murmurhash1(x, size):
    return int(murmurhash3_32(x, seed=5000) % size)


def murmurhash2(x, size):
    return int(murmurhash3_32(x, seed=5100) % size)


def murmurhash3(x, size):
    return int(murmurhash3_32(x, seed=5200) % size)


def murmurhash4(x, size):
    return int(murmurhash3_32(x, seed=5300) % size)


def murmurhash5(x, size):
    return int(murmurhash3_32(x, seed=5400) % size)


def hash(x):
    return int(murmurhash3_32(x, seed=5500) % 2)


def error(estimated_count, true_count):
    return abs(estimated_count - true_count) / true_count


def estimated_count_min(test, R):
    hash_counts = []
    hash_counts.append(counters[0, int(murmurhash1(test, R))])
    hash_counts.append(counters[1, int(murmurhash2(test, R))])
    hash_counts.append(counters[2, int(murmurhash3(test, R))])
    hash_counts.append(counters[3, int(murmurhash4(test, R))])
    hash_counts.append(counters[4, int(murmurhash5(test, R))])
    return hash_counts


def estimated_count_sketch(test, R):
    sign_hash_counts = []
    sign_hash_counts.append(sign_hash[0, int(murmurhash1(test, R))])
    sign_hash_counts.append(sign_hash[1, int(murmurhash2(test, R))])
    sign_hash_counts.append(sign_hash[2, int(murmurhash3(test, R))])
    sign_hash_counts.append(sign_hash[3, int(murmurhash4(test, R))])
    sign_hash_counts.append(sign_hash[4, int(murmurhash5(test, R))])
    sorted_shc = sorted(sign_hash_counts)
    for i in range(0, len(sorted_shc)):
        if hash(test) % 2 == 1:
            sorted_shc[i] = sorted_shc[i] * -1

    return sorted_shc


def hash_count(counters, query, R):
    words = query.split(" ")
    for word in words:
        counters[0, int(murmurhash1(word, R))] += 1
        counters[1, int(murmurhash2(word, R))] += 1
        counters[2, int(murmurhash3(word, R))] += 1
        counters[3, int(murmurhash4(word, R))] += 1
        counters[4, int(murmurhash5(word, R))] += 1


def signed_hash_count(sign_hash, query, R):
    words = query.split(" ")
    for word in words:
        if hash(word) == 1:
            sign_hash[0, int(murmurhash1(word, R))] -= 1
            sign_hash[1, int(murmurhash2(word, R))] -= 1
            sign_hash[2, int(murmurhash3(word, R))] -= 1
            sign_hash[3, int(murmurhash4(word, R))] -= 1
            sign_hash[4, int(murmurhash5(word, R))] -= 1

        else:
            sign_hash[0, int(murmurhash1(word, R))] += 1
            sign_hash[1, int(murmurhash2(word, R))] += 1
            sign_hash[2, int(murmurhash3(word, R))] += 1
            sign_hash[3, int(murmurhash4(word, R))] += 1
            sign_hash[4, int(murmurhash5(word, R))] += 1


# finding counter values of each hash function for the test
for query in queries:
    words = query.split(" ");
    for word in words:
        if (word in dictionary.keys()):
            dictionary[word] += 1
        else:
            dictionary[word] = 1

key_value_pairs = ((value, key) for (key, value) in dictionary.items())


def median(val):
    if len(val) % 2 == 0:
        return (val[len(val) // 2] + val[len(val) // 2 - 1]) / 2
    else:
        return val[len(val) // 2]


freq_100(key_value_pairs)
infreq_100()
rand_100(dictionary)


for R in [2 ** 10, 2 ** 14, 2 ** 18]:
    x1 = []
    y1 = []
    y2 = []
    y3 = []
    print("At ",R)
    counters = np.zeros((5, R))
    sign_hash = np.zeros((5, R))
    for query in queries:
        hash_count(counters, query, R)
        signed_hash_count(sign_hash, query, R)

    for key, value in Infreq_100.items():
        x1.append(key)
        y1.append(error(min(estimated_count_min(key, R)), value))
        y2.append(error(median(estimated_count_min(key, R)), value))
        y3.append(error(median(estimated_count_sketch(key, R)), value))
    plt.figure()
    plt.title("Least-frequent 100 words: %s" % R)
    plt.plot(x1, y1, 'r', label='Count-min')
    plt.plot(x1, y2, 'g', label='Count-median')
    plt.plot(x1, y3, 'b', label='Count-Sketch')
    plt.xticks(rotation=90)
    plt.legend(loc="upper left")

    x1 = []
    y1 = []
    y2 = []
    y3 = []
    for key, value in Freq_100.items():
        x1.append(key)
        y1.append(error(min(estimated_count_min(key, R)), value))
        y2.append(error(median(estimated_count_min(key, R)), value))
        y3.append(error(median(estimated_count_sketch(key, R)), value))
    plt.figure()
    plt.title("Frequent 100 words: %s" % R)
    plt.plot(x1, y1, 'r', label='Count-min')
    plt.plot(x1, y2, 'g', label='Count-median')
    plt.plot(x1, y3, 'b', label='Count-Sketch')
    plt.xticks(rotation=90)
    plt.legend(loc="upper left")

    x1 = []
    y1 = []
    y2 = []
    y3 = []

    for key, value in Rand_100.items():
        x1.append(key)
        y1.append(error(min(estimated_count_min(key, R)), value))
        y2.append(error(median(estimated_count_min(key, R)), value))
        y3.append(error(median(estimated_count_sketch(key, R)), value))
    plt.figure()
    plt.title("Random 100 words: %s" % R)
    plt.xlabel(" Words ")
    plt.ylabel("Error")
    plt.plot(x1, y1, 'r', label='Count-min')
    plt.plot(x1, y2, 'g', label='Count-median')
    plt.plot(x1, y3, 'b', label='Count-Sketch')
    plt.xticks(rotation=90)
    plt.legend(loc="upper left")

plt.show()

print("Space occupied by the dictionary is ", convert_size(sys.getsizeof(dictionary)))
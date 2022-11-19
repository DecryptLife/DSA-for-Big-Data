import heapq
import os
import statistics
from heapq import heapify, heappop, heappush

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import murmurhash3_32

dictionary = {}
filename = "/user-ct-test-collection-01.txt"
data2 = pd.read_csv(os.getcwd() + filename, sep="\t")
queries_list = data2.Query.dropna()
queries = []
x = []

x_axis = []
y_axis= []
intersection = [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]]

for query in queries_list:
    words = query.split(" ")
    for word in words:
        queries.append(word)


heap = [[[], [], []],
        [[], [], []],
        [[], [], []]]


dict_heap = [[{}, {}, {}],
             [{}, {}, {}],
             [{}, {}, {}]]

Top100 = {}


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


def top_100(keys_and_values):
    top100 = {}
    sorted_dictionary = sorted(keys_and_values, reverse=True)
    for (value, key) in sorted_dictionary[:500]:
        Top100[key] = value
    return top100


def hash_count(counters, query, R):
    for word in query:
        counters[0, int(murmurhash1(word, R))] += 1
        counters[1, int(murmurhash2(word, R))] += 1
        counters[2, int(murmurhash3(word, R))] += 1
        counters[3, int(murmurhash4(word, R))] += 1
        counters[4, int(murmurhash5(word, R))] += 1


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
        else:
            sorted_shc[i] = sorted_shc[i] * 1

    return sorted_shc


def estimated_count_min(test, R):
    hash_counts = []
    hash_counts.append(counters[0, int(murmurhash1(test, R))])
    hash_counts.append(counters[1, int(murmurhash2(test, R))])
    hash_counts.append(counters[2, int(murmurhash3(test, R))])
    hash_counts.append(counters[3, int(murmurhash4(test, R))])
    hash_counts.append(counters[4, int(murmurhash5(test, R))])
    return hash_counts


def signed_hash_count(sign_hash, query, R):
    word = query
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


for query in queries:
    words = query.split(" ")
    for word in words:
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1

key_value_pairs = ((value, key) for (key, value) in dictionary.items())
sorted_dictionary = sorted(key_value_pairs, reverse=True)
for (value, key) in sorted_dictionary[0:100]:
    Top100[key] = value
for k in [2 ** 10, 2 ** 14, 2 ** 18]:
    x.append(k)

for query in queries:
    if query in dictionary:
        dictionary[query] += 1
    else:
        dictionary[query] = 1

for i in range(0, len(x)):
    counters = np.zeros((5, x[i]))
    sign_hash = np.zeros((5, x[i]))
    print("run ", i)
    for query in queries:
        hash_count(counters, query, x[i])
        signed_hash_count(sign_hash, query, x[i])

    for query in queries:
        c_list = [min(estimated_count_min(query, x[i])),
                  statistics.median(estimated_count_min(query, x[i])),
                  statistics.median(estimated_count_sketch(query, x[i]))]

        for index in range(0, len(c_list)):
            c = c_list[index]
            if query in dict_heap[i][index]:
                dict_heap[i][index][query][0] += 1
                heapq.heapify(heap[i][index])
            else:
                dp = [c, query]
                if len(heap[i][index]) == 0:
                    dict_heap[i][index][query] = dp
                    heapq.heappush(heap[i][index], dp)
                else:
                    root = heap[i][index][0]
                    if len(heap[i][index]) < 500:
                        dict_heap[i][index][query] = dp
                        heapq.heappush(heap[i][index], dp)
                        heapq.heapify(heap[i][index])
                    elif c > root[0] and len(heap[i][index]) == 500:
                        top = heapq.heappop(heap[i][index])
                        del dict_heap[i][index][top[1]]
                        heapq.heappush(heap[i][index], dp)
                        heapq.heapify(heap[i][index])
                        dict_heap[i][index][query] = dp


for i in range(3):
    for j in range(3):
        print(dict_heap[i][j])
for i in range(3):
    for j in range(3):
        for word in dict_heap[i][j]:
            if word in Top100:
                intersection[i][j] += 1
color = ["r", "g", "b"]
label = ["Count-min", "Count-median", "Count-Sketch"]
for i in range(3):
    print(x[i])
    x_axis.append(x[i])
    y_axis.append(intersection[i])
plt.scatter(x_axis[0], intersection[0][0],c=color[0],label=label[0])
plt.scatter(x_axis[0], intersection[0][1],c=color[1],label=label[1])
plt.scatter(x_axis[0], intersection[0][2],c=color[2],label=label[2])
for i in [1,2]:
    plt.scatter(x_axis[i], y_axis[i][0],c=color[0])
    plt.scatter(x_axis[i], y_axis[i][1],c=color[1])
    plt.scatter(x_axis[i], y_axis[i][2],c=color[2])

plt.xlabel("Size")
plt.ylabel("Intersection")
plt.title("Intersection of Count-min, Count-Median, Count-Sketch with Top 100 words")
plt.legend(loc="lower right")
plt.show()

print("Intersection: ", intersection)

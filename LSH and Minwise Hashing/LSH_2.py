import heapq
import os
import random
import pandas as pd
from sklearn.utils import murmurhash3_32
from heapq import heapify, heappop, heappush
import time
import matplotlib.pyplot as plt
from scipy.special import expit as logistic

filename = "/user-ct-test-collection-01.txt"
data2 = pd.read_csv(os.getcwd() + filename, sep="\t")
queries = data2.Query.dropna().unique()
J_vals = []
P_vals = []
top_10 = []




def murmurhash(query, seed):
    return murmurhash3_32(query, seed=seed, positive=True)


def minhash(A, k):
    minhash_values = []
    anagrams = []
    query_anagram_hash = {}
    minhash_queries = []
    if(len(A)>=3):
        for index in range(len(A) - 2):
            anagrams.append(A[index:index + 3])
    else:
        anagrams.append(A)

    for i in range(k):
        for query in anagrams:
            if query not in query_anagram_hash:
                query_anagram_hash[query] = murmurhash(query, 100 + (i * 2))
        minhash_val = min(query_anagram_hash, key=query_anagram_hash.get)
        minhash_queries.append(minhash_val)
        minhash_values.append(query_anagram_hash[minhash_val])
        query_anagram_hash = {}

    return [minhash_values,minhash_queries]


def calculateJaccard(test,buckets, hashcodes, K):
    total_jsim =0
    for bucket in buckets:
        jsim = 0
        minhash_ele= []
        for ele in bucket:
            anagrams = []
            minhash_set = []
            if len(ele) < 3:
                anagrams.append(ele)
            else:
                for pos in range(len(ele) -2):
                    anagrams.append(ele[pos:pos+3])
            for n in range(K):
                ele_hash_set = set()
                for anagram in anagrams:
                    ele_hash_set.add(murmurhash(anagram, 100 + (n * 2)))
                minhash_set.append(min(ele_hash_set))
                minhash_ele.append(minhash_set)
        for ele_hash in minhash_ele:
            if any(value in ele_hash for value in hashcodes):
                    jsim += 1
        jsim = jsim/K


        jsim = jsim/len(bucket)
        total_jsim += jsim
    return total_jsim/len(buckets)




def jbrute_heap(test,buckets):
    test_anagram = []
    heap = []
    dict_heap = {}

    if len(test)<3:
        test_anagram.append(test)
    else:
        for pos in range(len(test) -2):
            test_anagram.append((test[pos:pos+3]))


    total_bsim = 0
    for bucket in buckets:
        bucket_bsim = 0
        for query in bucket:
            b_sim = 0
            queries = [test,query]
            union = []
            intersection = []
            anagrams = []

            for anagram in test_anagram:
                if anagram not in union:
                    union.append(anagram)

            if(len(query)<3):
                anagrams.append(query)
            else:
                for pos in range(len(query) -2):
                    anagrams.append(query[pos:pos+3])

            for anagram in anagrams:
                if anagram in test_anagram and anagram not in intersection:
                    intersection.append(anagram)
                if anagram not in test_anagram:
                    union.append(anagram)

            j_val = len(intersection)/len(union)

            bucket_bsim += j_val

            if query not in dict_heap.keys():
                if (len(heap) < 10):

                    dp = [j_val, query]
                    dict_heap[query] = j_val
                    heapq.heappush(heap,dp)
                    heapq.heapify(heap)
                else:
                    heapq.heapify(heap)
                    min_sim = heap[0][0]
                    key = heap[0][1]
                    if j_val > min_sim:
                        dp = [j_val, query]
                        heapq.heappop(heap)
                        del dict_heap[key]
                        heapq.heappush(heap, dp)
                        dict_heap[query] = j_val
                        heapq.heapify(heap)
    top_10_list = list(dict_heap.values())
    return top_10_list,sorted(dict_heap, key=dict_heap.get, reverse=True)

def brute_jaccard(test,buckets):
    test_anagram = []

    if len(test) < 3:
        test_anagram.append(test)
    else:
        for pos in range(len(test) - 2):
            test_anagram.append((test[pos:pos + 3]))

    total_bsim = 0
    for bucket in buckets:
        bucket_bsim = 0
        for query in bucket:
            union = []
            intersection = []
            anagrams = []

            for anagram in test_anagram:
                if anagram not in union:
                    union.append(anagram)

            if (len(query) < 3):
                anagrams.append(query)
            else:
                for pos in range(len(query) - 2):
                    anagrams.append(query[pos:pos + 3])

            for anagram in anagrams:
                if anagram in test_anagram and anagram not in intersection:
                    intersection.append(anagram)
                if anagram not in test_anagram:
                    union.append(anagram)

            j_val = len(intersection) / len(union)

            bucket_bsim += j_val


        bucket_bsim = bucket_bsim / len(bucket)
        total_bsim += bucket_bsim

    total_bsim = total_bsim / len(buckets)

    return total_bsim

class HashTable():

    def __init__(self, K, L, B, R):
        self.K = K
        self.L = L
        self.B = B
        self.R = R
        self.buckets = self.createBuckets(B)
        self.a_constants = self.createConstants(L,K)


    def createBuckets(self, size):
        bucket = []
        for l in range(self.L * self.B):
            bucket.append([])
        return bucket

    def insert(self, hashcodes, id):
        for l in range(self.L):
            index = 0
            for pos in range(self.K):
                index += self.a_constants[l][pos] * hashcodes[pos]

            index %= self.B
            self.buckets[index+(l * self.B )-1].append(id)



    def lookup(self, hashcodes):

        matching_buckets=[]


        check = 0
        for l in range(self.L):
            index = 0
            for pos in range(self.K):
                index += self.a_constants[l][pos] * hashcodes[pos]
            index %= self.B
            matching_buckets.append(self.buckets[index+(l*self.B) -1])
        return matching_buckets


    def display(self):
        print(self.a_constants)

    def createConstants(self, L, K):
        a_constants = []
        for i in range(L):
            a_constants.append(random.sample(range(1,1000),K))
        return a_constants


query_set = []
sample_index = random.sample(range(0,len(queries)),200)
for i in sample_index:
    query_set.append(queries[i])

K_values = [2,3,4,5,6]
L_values = [20,50,100]

for K in K_values:
    for L in L_values:
        print()
        print("When K = ",K," and L = ",L)
        hashTable = HashTable(K,L,64,2**20)

        for query in queries:
            [hashval,minhash_queries] = minhash(query,K)
            hashTable.insert(hashval,query)


        for ele in query_set:
            [hashval,minhash_queries] = minhash(ele,K)
            matched = hashTable.lookup(hashval)

        j_start = time.time()
        mean_j_200 = 0
        for ele in query_set:
            [hashval, minhash_queries] = minhash(ele, K)
            mean_j_200 += calculateJaccard(ele,matched,hashval,K)

        time_taken = time.time() - j_start
        print("Jaccard similarity mean of the sample : ",mean_j_200/200)
        print("Time taken when K = ",K," and L = ",L," is ",time_taken," seconds")





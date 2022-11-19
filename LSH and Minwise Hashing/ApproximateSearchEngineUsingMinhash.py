import numpy as np
from sklearn.utils import murmurhash3_32
# Implement simple MinHash generator. Given an input string and m, return m hashcodes.

def murmurhash(query,seed):
    return murmurhash3_32(query, seed = seed, positive=True)


queries = ["The mission statement of the WCSCC and area employers recognize the importance of good attendance on the job. Any student whose absences exceed 18 days is jeopardizing their opportunity for advanced placement as well as hindering his/her likelihood for successfully completing their program.",
           "The WCSCC’s mission statement and surrounding employers recognize the importance of great attendance. Any student who is absent more than 18 days will loose the opportunity for successfully completing their trade program.}"]


query1_anagram = []
query2_anagram = []
query1_hashcode = []
query2_hashcode = []
query1_anagram_hash = {}
query2_anagram_hash = {}
query_anagram_hash= [{},{}]
union = []
intersection = []

for index in range(len(queries[0]) - 2):
    if queries[0][index:index+3] not in query1_anagram:
        query1_anagram.append(queries[0][index:index+3])

for index in range(len(queries[1]) - 2):
    if queries[1][index:index+3] not in query2_anagram:
        query2_anagram.append(queries[1][index:index+3])

for anagram in query1_anagram:
    if anagram not in union:
        union.append(anagram)

for anagram in query2_anagram:
    if anagram not in union:
        union.append(anagram)

for anagram in query1_anagram:
    if anagram in query2_anagram and anagram not in intersection:
        intersection.append(anagram)



minhash = [[],[]]
for index,query in enumerate(queries):
    for n in range(100):
        for pos in range(len(query)-2):
            anagram = query[pos:pos+3]
            if anagram not in query_anagram_hash[index]:
                query_anagram_hash[index][anagram] = murmurhash(anagram,100+(n*2))
        minhash_val =  min(query_anagram_hash[index], key=query_anagram_hash[index].get)
        minhash[index].append(query_anagram_hash[index][minhash_val])
        query_anagram_hash[index] = {}

similar = 0
for ele in minhash[0]:
    if ele in minhash[1]:
        similar = similar+1

actual = len(intersection)/len(union)
print("Similar: ", similar/100)
print("Actual Jaccard", actual)



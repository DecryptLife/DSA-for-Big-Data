import itertools
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import murmurhash3_32

p = 1048573

x_values = []
hash_values_2 = []
hash_values_3 = []
hash_values_4 = []
murmur_values = []
bin_x = []

random.seed(7)
a = random.randint(1, 1048573)
b = random.randint(1, 1048573)
c = random.randint(1, 1048573)
d = random.randint(1, 1048573)

print("Value of a: ", a)
print("Value of b: ", b)
print("Value of c: ", c)
print("Value of d: ", d)


def hash_function_2(x):
    return ((a * x + b) % p) % 1024


def hash_function_3(x):
    return ((a * (x ** 2) + b * x + c) % p) % 1024


def hash_function_4(x):
    return ((a * (x ** 3) + b * (x ** 2) + c * x) % p) % 1024


def murmurhash(x):
    return murmurhash3_32(x, seed=1000) % 1024

def binaryToDecimal(num):
    return int(num,2)

def mod_binary_10(bin_val):
    if len((bin_val)) == 10:
        return bin_val
    else:
        return ((10 - len(bin_val)) * '0') + (bin_val)


def mod_binary_31(bin_val):
    if len(bin_val) == 31:
        return bin_val
    else:
        return (31 - len(bin_val)) * '0' + bin_val


# print(binaryToDecimal('1000'))

x = [random.getrandbits(31) for i in range(5000)]

# converting x to binary values
for val in x:
    x_to_bin = bin(val)[2:]
    bin_x.append(mod_binary_31(x_to_bin))

# storing probabilities for each hash function
p1 = np.zeros([31, 10])
p2 = np.zeros([31, 10])
p3 = np.zeros([31, 10])
p4 = np.zeros([31, 10])

# generating hash functions in binary for each value in x
for value in x:
    hash_values_2.append(mod_binary_10(bin(hash_function_2((value)))[2:]))
    hash_values_3.append(mod_binary_10(bin(hash_function_3((value)))[2:]))
    hash_values_4.append(mod_binary_10(bin(hash_function_4((value)))[2:]))
    murmur_values.append(mod_binary_10(bin(murmurhash((value)))[2:]))


# change each bit of x then compare the change with the hash values
for i in range(0, 31):
    # flipping each bit one by one and then finding the total changes
    for index, value in enumerate(bin_x):
        if value[i] == '0':
            new_val = value[:i] + '1' + value[i + 1:]
        else:
            new_val = value[:i] + '0' + value[i + 1:]

        # find hash of new flipped bit
        # should we convert back to int?
        new_val_hash = mod_binary_10(bin(binaryToDecimal(bin(hash_function_2(binaryToDecimal(new_val)))[2:]))[2:])
        new_val_hash3 = mod_binary_10(bin(binaryToDecimal(bin(hash_function_3(binaryToDecimal(new_val)))[2:]))[2:])
        new_val_hash4 = mod_binary_10(bin(binaryToDecimal(bin(hash_function_4(binaryToDecimal(new_val)))[2:]))[2:])
        new_val_mhash = mod_binary_10(bin(binaryToDecimal(bin(murmurhash(binaryToDecimal(new_val)))[2:]))[2:])

        for k in range(0, 10):
            if new_val_hash[k] != hash_values_2[index][k]:
                p1[i, k] += 1
            if new_val_hash3[k] != hash_values_3[index][k]:
                p2[i,k] += 1
            if new_val_hash4[k] != hash_values_4[index][k]:
                p3[i,k] += 1
            if new_val_mhash[k] != murmur_values[index][k]:
                p4[i,k] += 1


# sampling the p1
p1 = p1 / 5000
p2 = p2/5000
p3 = p3/5000
p4 = p4/5000


plt.figure(figsize=(31,10))
heatmap = sns.heatmap(p1.T,cmap="twilight",linewidth = 1, annot= False)
plt.title("2 - Universal Hash Function")
plt.xlabel("Input Bits")
plt.ylabel("Output Bits")

plt.figure(figsize=(31,10))
heatmap = sns.heatmap(p2.T,cmap="twilight",linewidth = 1, annot= False)
plt.title("3 - Universal Hash Function")
plt.xlabel("Input Bits")
plt.ylabel("Output Bits")

plt.figure(figsize=(31,10))
heatmap = sns.heatmap(p3.T,cmap="twilight",linewidth = 1, annot= False)
plt.title("4 - Universal Hash Function")
plt.xlabel("Input Bits")
plt.ylabel("Output Bits")

plt.figure(figsize=(31,10))
heatmap = sns.heatmap(p4.T,cmap="twilight",linewidth = 1, annot= False)
plt.title("Murmurhash ")
plt.xlabel("Input Bits")
plt.ylabel("Output Bits")
plt.show()




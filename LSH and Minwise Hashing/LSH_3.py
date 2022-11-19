import matplotlib.pyplot as plt

J_vals = []
start = 0.01
for i in range(100):
    J_vals.append(round((i * start), 2))
J_vals.append(1.00)

L = 50
K_vals = [1, 2, 3, 4, 5, 6, 7]

plt.title("Probability of retrieving upon varying no of hashes(K)")
for K in K_vals:
    y_vals = []
    for J in J_vals:
        P = 1 - ((1 - J ** K) ** L)
        y_vals.append(P)

    plt.plot(J_vals, y_vals, label=K)
    plt.ylabel("Probability of retrieving")
    plt.xlabel("Jaccard similarity values")
plt.legend()
plt.show()

K = 4
L_vals = [5, 10, 20, 50, 100, 150, 200]
plt.title("Probability of retrieving upon varying Hash Tables (L)")
for L in L_vals:
    y_vals = []
    for J in J_vals:
        P = 1 - ((1 - J ** K) ** L)
        y_vals.append(P)

    plt.plot(J_vals, y_vals, label=L)
    plt.xlabel("Probability of retrieving")
    plt.ylabel("Jaccard similarity values")
plt.legend()
plt.show()

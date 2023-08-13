import numpy as np

np.random.seed(10)

a = np.random.randn(2, 1)
b = np.random.randn(2, 2, 2)

results = np.zeros((2, 2))

for i in range(2):
    results[:, i] = a.T @ (b[:, :, i])
    
print(results)

# print(np.einsum('ij,jk->ik', a.ravel(), b))
print(np.matmul(a.T, b.reshape(2, -1)).reshape(results.shape))


result_left = a.T @ (b[:, :, 1])
result_right = np.squeeze(a.T @ b)[:, 1]

print("result_left:\n", result_left)
print("result_right:\n", result_right)

assert np.allclose(result_left, result_right)
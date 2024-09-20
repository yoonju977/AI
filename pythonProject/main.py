import time
import numpy as np
import torch

# 1. 일반 Python 리스트 연산 시간 측정
start_time = time.time()

X = [1] * 10000
Y = [0.5] * 10000
Z = [None] * 10000
for i in range(10000):
    Z[i] = X[i] * Y[i]

end_time = time.time()
print(f"Execution time for Python list operation: {end_time - start_time} seconds")

# 2. numpy 배열 연산 시간 측정
start_time = time.time()

X = np.full((10000,), 1)
Y = np.full((10000,), 0.5)
Z = X * Y

end_time = time.time()
print(f"Execution time for numpy operation: {end_time - start_time} seconds")

# 3. PyTorch 연산 시간 측정 (CPU 사용)
start_time = time.time()

X = torch.full((10000,), 1.0, requires_grad=True)
Y = torch.full((10000,), 0.5, requires_grad=True)
Z = X * Y

end_time = time.time()
print(f"Execution time for PyTorch CPU operation: {end_time - start_time} seconds")

# 역전파 (backward) 계산
Z.sum().backward()

# 그라디언트 확인
dx = X.grad
print(dx)
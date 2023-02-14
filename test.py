import torch

z = torch.zeros(5,3)
print(z)
print(z.dtype)

i = torch.ones((5,3), dtype=torch.int16)
print(i)

torch.manual_seed(1729)
r1 = torch.rand(2,2)
print('랜덤 tensor 값:')
print(r1)

r2 = torch.rand(2,2)
print('\n다른 랜덤 tensor 값:')
print(r2)

torch.manual_seed(1729)
r3 = torch.rand(2,2)
print('\nr1과 일치:')
print(r3)

ones = torch.ones(2,3)
print(ones)

twos = torch.ones(2,3) * 2
print(twos)

threes = ones + twos
print(threes)
print(threes.shape)

r1 = torch.rand(2,3)
r2 = torch.rand(3,2)
# 런타임 오류
# r3 = r1 + r2

r = (torch.rand(2,2) - 0.5) * 2 # -1과 1사이의 값
print("랜덤 행렬값, r:")
print(r)

# 절댓값
print('\nr의 절댓값:')
print(torch.abs(r))

# 삼각함수
print('\nr의 사인 함수:') # sin x
print(torch.sin(r))
print('\nr의 역 사인 함수:') # arcsin x
print(torch.asin(r))

# 선형대수 연산
# 행렬식 (determinant): 행렬을 대표하는 값. 2x2일때 행렬의 요소 값이 a,b,c,d 일 때 행렬 식은 ad - bc 이다.
print('\nr의 행렬식:')
print(torch.det(r))
# 특이값 분해
print('\nr의 특이값 분해:')
print(torch.svd(r))

# 통계 및 집합 연산 등
print('\nr의 평균 및 표준편차:')
print(torch.std_mean(r))
print('\nr의 최대값:')
print(torch.max(r))


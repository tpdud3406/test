## Pytorch tensors

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

## Pytorch models

import torch # Pytorch 모든 모듈 가져오기
import torch.nn as nn # torch.nn.Module의 경우 Pytorch model의 부모 객체
import torch.nn.functional as F # 활성화 함수 모듈 가져오기

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 입력 이미지 채널, 6개의 output 채널, 3x3 정방 합성곱 커널 사용
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 아핀 변환: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6*6 이미지 차원
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print('forward conv')
        # 최대 풀링은 (2, 2) 윈도우 크기 사용
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # 크기가 정방 사이즈인 경우, 단일 숫자만 지정할 수 있다.
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = LeNet()
print(net) # 인스턴스한 객체를 출력하면 어떤 값을 보여줄까요?

input = torch.rand(1, 1, 32, 32) # 32x32 크기의 1채널의 흑백 이미지를 만든다.
print('\n이미지 배치 shape:')
print(input.shape)

output = net(input) # 객체로부터 직접 forward() 함수를 호출하지 않습니다.
print('\n: 결과 값')
print(output)
print(output.shape)




---
layout: post
title: Pytorch로 시작하는 딥러닝 입문 (1)Package 구성
categories: [PyTorch]
---

이 글을 시작으로 Wikidocs [Pytorch로 시작한는 딥러닝 입문](https://wikidocs.net/book/2788)을 공부한 내용을 포스팅할 것이다. </br>
자연어처리를 공부하기에 관련된 내용 위주로 올릴 것이다.</br>
</br>

# __1. PyTorch 기초__
PyTorch의 기본 구성

---
</br>

## 1. torch
---
메인 네임스페이스로 tensor 등 다양한 수학 함수 포함됨.</br>
Numpy와 유사한 구조.</br></br>

## 2. torch.autograd
---
자동 미분을 위한 함수가 포함됨. (enable_grad / no_grad)로 자동 미분 On/Off 가능</br></br>

## 3. torch.nn
---
신경망 구축 시 쓰이는 모델 구조나 레이어 정의되어 있음. (ex: RNN, LSTM, ReLU, MSELoss, Conv layer 등)</br>
모델을 직접 구축할 때에는 모델 클래스 선언시 nn.Module 상속해야함

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        ...
        ...
```
</br>

## 4. torch.optim
---
SGD(Stochastic Gradient Descent)를 중심으로한 파라미터 최적화 알고리즘 구현됨.</br>
optimizer object로 만들어지며 계산된 gradient를 이용하여 parameter update 수행함.
```python
from torch import optim

optimizer = optim.SGD(model.parameter(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```
Optimizer를 이용해서 parameter update 할 시 `optim.step()` 이용해서 update함.
```python
for input, target in dataset:
    optimizer.zero_grad()  # 누적된 gradient 없앤 후 진행
    output = model(input)
    loss = loss_fn(output, target)  # Computing loss
    loss.backward()   # Backpropagation
    optimizer.step()  # Parameter update
```
</br>

## 5. torch.utils.data
---
SGD의 반복 연산 때 사용하는 미니 배치용 유틸리티 함수 포함.</br>
</br>

Wikidocs만 봐서는 무슨말인지 잘 모르겠다. PyTorch 공식 document에 따르면 Pytorch에서 데이터 로딩의 핵심이 `torch.utils.data.DataLoader class` 라고 한다.
```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```
블로그 시작 얼마 전, Transformer 코드를 만들려 할 때, 이 `DataLoader` 때문에 꽤나 고생했다. 제대로 한번 알아봐야 된다고 생각하는데 이건 나중에 따로 정리해야겠다.
</br>
</br>

## 6. torch.onnx
---
ONNX(Open Neural Network Exchange)의 포멧. Model export 시 사용한다. ONNX는 서로다른 딥러닝 프레임워크(Tensorflow와 Pytorch) 간 모델 공유 시 사용함.</br></br>
아직은 사용할 일이 없을 것 같아서 나중에 쓸 일이 생기게 된다면 그 때 다시 공부할 생각이다.
---
layout: post
title: PyTorch로 시작하는 딥러닝 입문 (4)-2 선형회귀, Autograd
categories: [PyTorch]
---

---

이번 포스팅은 [PyTorch로 시작하는 딥러닝 입문](https://wikidocs.net/53560)의 선형 회귀 구현 부분을 다룰 것이다. <br>
우선 코딩 실습을 위해 PyTorch의 module import 부터 시작할 것이다. <br>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 코드를 재실행 하더라도 같은 결과 나오도록 랜덤 시드 부여
torch.manual_seed(1)
```

실습 시 필요한 기본적인 셋팅이 끝났다. 다음으로는 훈련데이터 x_train, y_train을 선언한다. <br>

```python
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
```

```python
print(x_train)
print(x_trailn.shape)
```

```python
# x_train과 x_train의 크기
tensor([[1.],
        [2.],
        [3.]])
torch.Size([3,1])
```

훈련 데이터 x_train과 훈련 데이터 라벨 y_train의 크기가 [3x1]임을 알 수 있다. <br><br>

우리가 학습하게 될 선형 회귀는 학습 데이터와 가장 잘 맞는 하나의 직선을 찾는 일이다. <br>
이 때 그 직선을 정의하는 것이 W와 b (Weight와 bias)이다. 우선은 가중치와 편향을 0으로 <br>
초기화한 다음 학습을 통해 변경할 수 있도록 하겠다. <br>

```python
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print(W)
print(b)
```

```python
tensor([0.], requires_grad=True)  # Weight
tensor([0.], requires_grad=True)  # Bias
```

```requires_grad=True ```를 인자로 주어 학습을 통해 변수를 변경할 수 있게 만든다. 그리고 현재 가중치와 편향 모두 0이므로 현재의<br>
직선 방정식은 다음과 같다. <br>

$$ y = 0 \times x + 0 $$

현재 x에 무슨 값이 들어가도 결과는 0이 나오므로 적절한 W와 b 값은 아직 아니다. 그리고 아직 선형 회귀에 대한 가설의 코드는<br>
구현하지 않았으므로 직선의 방적식으로 선형 회귀의 가설을 구현한다. <br>

$$ H(x) = Wx + b $$

```python
hypothesis = x_train * W + b
```

코드에서 ``` W * x_trian ```이 아닌 ```x_train * W```인 이유는 행렬의 곱셈시 차원을 맞춰주기 위함이다. ([3,1] * [1]) <br><br>

다음으로는 손실 함수를 구현해 준다. 손실 함수는 선형 회귀의 손실 함수인 Mean Square Error를 사용한다. <br>

$$ cost(W, b) = {1 \over n} \sum_{i=1}^N{[y^{(i)} - H(x^{(i)})]^2} $$

```python
cost = torch.mean((y_train - hypothesis)**2)
```

<br>
다음으로는 optimizer를 구현한다. Optimizer는 경사하강법(Stochastic Gradient Descent)을 사용한다. <br>

```python
optimizer = optim.SGD([W, b], lr = 0.01)
```

여기서 lr은 학습률(learning rate)을 의미한다. Optimizer를 사용하기 전에 이전에 얻었던 기울기를 0으로 초기화해야 한다. <br>
기울기 초기화를 해야 새로운 가중치와 편향에 대해 기울기를 새로 구할 수 있기 때문이다. 그 후 역전파를 수행하게 되면 <br>
가중치와 편향에 대해 기울기가 계산된다. 계산된 기울기에 optimizer의 학습률을 이용해서 W와 b를 업데이트 한다. <br>

```python
# Gradient 0으로 초기화
optimizer.zero_grad()

# Backpropagate W and b
cost.backward()

# Update W and b
optimizer.step()
```

<br>
전체 코드를 정리하자면 다음과 같다.<br><br>

```python
# 필요 PyTorch module import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 시드 고정으로 항상 같은 결과값
torch.manual_seed(1)

# Train Data
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# Initialize Model Parameter 
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Optimizer
optimizer = optim.SGD([W, b], lr=0.01)

num_epochs = 1999  # SGD 반복할 횟수

for epoch in  range(num_epochs+1):

    # Compute hypothesis H(x)
    hypothesis = x_train * W + b

    # Compute loss function
    loss = torch.mean((y_train - hypothesis)**2)

    # Update W and b
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 100회 반복마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Loss: {:.6f}'.format(epoch, num_epochs, W.item(), b.item(), loss.item()))
```

<br>
훈련의 결과 W와 b는 훈련 데이터를 가장 잘 표현하는 직선의 식을 표현하기 위한 적절한 값으로 변화해간다. 여기서 에폭(Epoch)<br>
이란 훈련 데이터 전체가 학습에 한번씩 사용된 주기를 말한다. 위 코드에서는 각 데이터마다 2000번씩 사용되었다. 코드 실행 <br>
결과를 보면 W는2, b는 0에 가까운 값이 설정됨을 알 수 있다. 실제 훈련 데이터의 정답값을 보면 실제 정답은 W는2, b는 0인 <br>
$$ H(x) = 2x $$ 이므로 잘 찾아감을 알 수 있다. <br><br>

위 코드에서 ``` optimizer.zero_grad() ``` 가 사용되었는데 이는 미분통해 나온 기울기값을 누적시키는 PyTorch의 특징때문이다. 만일<br>
기울기를 0으로 초기화하지 않는다면 계속해서 미분값이 누적되어 잘못된 결과가 나타날 수도 있다. 이러한 일을 방지하기 위해 <br>
optimizer로 파라미터를 업데이트하기 전에 초기화를 해야 한다. <br><br><br>


## Autograd
---

위의 전체 코드를 보면 ```requires_grad=True, backward()``` 등이 나온다. 이들은 PyTorch에서 제공하는 자동 미분(Autograd)기능을<br>
수행하고 있는 것이다. 경사하강법에서는 손실 함수를 미분해 기울기를 구해서 손실값이 최소가 되는 방향을 찾아내 업데이트한다. <br>
예시 코드의 선형 회귀 문제에서는 미분을 직접 구현해도 큰 어려움은 없으나 모델이 복잡해질수록 직접 코딩하는 것은 거의 불가능
에 가까운 일이 될 것이다. PyTorch는 이런 어려운 일을 직접하지 않아도 되도록 자동 미분(Autograd)기능을 제공한다. 이로 인해 <br>
따로 미분 계산을 구현하지 않더라도 쉽게 경사하강법을 이용할 수 있게 된다. 

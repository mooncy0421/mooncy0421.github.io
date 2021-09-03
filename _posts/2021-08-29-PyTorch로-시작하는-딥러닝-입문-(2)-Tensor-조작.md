---
layout: post
title: PyTorch로 시작하는 딥러닝 입문 (2) Tensor 조작
categories: [PyTorch]
---

---

<br>
두번째 포스팅이다. <br>
첫번째 포스팅에서는 PyTorch 패키지 구성에 대해 간단하게 정리해봤고, <br>
이번에는 텐서 조작에 대해 살펴볼 거다.<br>
Wikidocs에서는 2장 2절 ~ 2장 3절 내용에 해당한다. 우선 간단하게 정리할 내용을 살펴보자면 다음과 같다. <br>

+ ### Vector / Matrix / Tensor 각각의 특징
+ ### Tensor Allocation
+ ### Basic Operations & Methods
 위의 순서로 정리할 것이며, [wikidocs](https://wikidocs.net/52460) 의 순서와 같다.
 <br><br>
 # __1. Vector / Matrix / Tensor 각각의 특징__
 PyTorch로 코딩을 하게되면 가장 자주 다루게 되는 단위는 벡터, 행렬, 텐서이다. 자연어 처리에서는 단어를 벡터로 표현하고, 문장을 벡터화된 단어로 표현, 그리고 그를 배치로 만들어 학습에 쓰므로 결국 텐서 단위를 가장 많이 접하고 다룰 것이라 생각한다. <br><br>

 <img src="./assets/img/PyTorch_wikidocs/vector_matrix_tensor_img.png" width="450px" height="300px"></img> <br><br>

+ ## Scalar
위의 그림에는 없으나 간단하게 말하자면 차원이 없는 값이다. 그냥 변수값 하나라고 생각하면 될 것 같다. 
<br><br>

+ ## Vector
벡터는 위의 그림에서 1D로 되어있는 형태로 표시된다. 1차원으로 구성된 값이며, scalar 값들이 일렬로 나열된 구조라 생각하면 될 것 같다.
<br><br>

+ ## Matrix
행렬 (Matrix)는 그림에서 2D로 되어있는 형태로 표시된다. 
<br><br>

+ ## Tensor
텐서는 그림에서 3D 이상의 차원을 가진 형태이다. 4D 이상은 생각해내기 어려워 3D 텐서를 쌓은 것으로 표현했다.
<br><br><br>

# __2. Tensor Shape 표현__
딥러닝에서 행렬이나 텐서를 사용할 때 각 차원의 크기는 특정 정보에 대해 나타내는데 자주 사용되는 2D / 3D Tensor에서 일반적으로 사용되는 특징 정보에 대해 알아볼 것이다.
<br><br>

+ ## 2D Tensor
---

|Tensor| = (batch size, dimension)<br>

![2D Tensor](./assets/img/PyTorch_wikidocs/2d_tensor_img.png "2D Tensor")

2차원 텐서에서 각 행 하나는 하나의 벡터 형태의 데이터를 표현하고 각 열들은 벡터의 feature값을 나타낸다. 
<br><br>

+ ## 3D Tensor (In NLP)
---

|Tensor| = (batch size, sequence length, dimension)<br>
![3D Tensor](./assets/img/PyTorch_wikidocs/3d_tensor_img.png "3D Tensor")
3차원 텐서는 위와 같이 나타낸다. 하나의 배치는 각각 dimension 크기만큼의 feature 수를 갖는 토큰들이 sequence length 길이의 sequence를 갖는다.
<br><br><br>

# __3. Tensor Allocation__

Python에서 텐서를 만드는 방법는 두가지가 있다. 우선 ```numpy``` 의 ```np.array``` 를 이용해서 만드는 방법이 있고, <br>
Pytorch에서 텐서를 선언하는 방법이 있다.
<br><br>

## 1) numpy로 선언
---

numpy로 텐서를 만드는 건 별로 어렵진 않다. 게다가 나는 numpy로 선언한 다음 torch를 이용해서 텐서로 바꿔주는게 큰 의미가 없다고 생각하기도 해서 굳이 안쓸 것 같다. 그래도 간단하게 살펴보자면 [num, num, ...] 처럼 단순하게 만드려는 텐서를 [ ]로 감싸준 다음 ```np.array``` 함수에 넣어주면 된다. <br>
```python
import numpy as np

tensor = np.array([0., 1., 2., 3.])  # 1D tensor
print(tensor)
print(tensor.ndim)  # tensor의 차원 수
print(tensor.shape)  # tensor의 크기
```

```python
[0., 1., 2., 3.]
1
(4,)
```
<br>

## 2) PyTorch로 선언
---

PyTorch는 numpy와 매우 비슷한 방식으로 텐서를 만든다. numpy 처럼 [num, num, ...] 로 만드려는 텐서를 감싸고 ```torch.Tensor``` 함수에 넣어주면 된다. 이 때 텐서 내 원소의 type을 지정하려면 LongTensor, FloatTensor 등의 특정 자료형 텐서로 선언하거나 단순하게 Tensor로 선언한 후 .float 과 같이 바꿔주면 된다. <br>
```python
import torch

tensor = torch.Tensor([0., 1., 2., 3.])
print(tensor)
print(tensor.dim())  # numpy의 ndim과 같음
print(tensor.shape)  # numpy의 shpae과 같음 (tensor.size()로 대체 가능)
```

```python
tensor([0., 1., 2., 3.])
1
torch.Size([4])
```
<br>

+ # __4. Basic Operations & Methods__

Pytorch의 텐서는 numpy의 array보다 더 많은 것을 할 수 있다. numpy에서 사용가능한 기능들은 물론 더 다양한 기능도 제공한다.<br><br>

+ ## 원소 접근 방법
---

만들어진 텐서의 원소에 접근하는 방법은 각 원소의 index를 사용하거나 slicing이라는 기법을 사용해서 접근하는 방법이 있다. <br>
<br>

+ ### Indexing
우선 index를 이용해 접근하는 방법은 
```python
print(tensor[0], tensor[1], tensor[-1])
```
과 같이 각 원소의 index에 바로 접근하면 된다. index는 0부터 시작하기 때문에 ```tensor[0]```을 입력하면 첫번째 위치의 값을 가리킨다. <br>
그리고 index를 넣는 곳에 -1을 넣게 되면 텐서의 해당 차원 마지막 index 값을 가리키게 된다. -2의 경우에는 뒤에서 두번째 위치, -3은 뒤에서 세번째 위치를 가리키며 1씩 감소한 값을 넣을 때 마다 한칸씩 앞으로 간다고 생각하면 된다.<br><br>

+ ### Slicing
Slicing은 텐서에서 원소를 불러올 범위를 지정해서 원소에 접근하는 방법인데 ```[start_idx : end_idx]```로 범위를 지정해서 사용한다. <br>
주의해야될 점은 범위가 start_idx값 부터 접근하는 것은 맞으나 end_idx-1 까지만 접근하고 end_idx 값은 포함되지 않기에 사용 시 주의해서 써야한다.
```python
print(tensor[0:3])  # index 0, 1, 2 위치의 값에 접근
```
여기서 start_idx나 end_idx를 생략한 채로 사용되는 경우도 있는데, start_idx를 생략하게 되면 맨 앞 index부터 end_idx-1 까지의 원소들에 접근하고, end_idx를 생략하면 start_idx부터 마지막 index의 원소들에 접근하게 된다. <br>
만일 2이상의 차원을 가지는 텐서의 경우에 start_idx, end_idx 모두 생략하는 경우가 있을 수 있다.
```python
t = torch.Tensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
print(:, :-1)
```
위의 코드처럼 첫번 째 차원 slicing 시 start_idx와 end_idx 모두 생략한 채 사용하게 된다면 다음과 같이 두번째 차원의 마지막 index 위치의 값만 제외하고 출력한다.
```python
tensor([[ 1.,  2.],
        [ 4.,  5.],
        [ 7.,  8.],
        [10., 11.]])
```
<br>
Slicing은 잘쓰면 상당히 유용할 듯 하다. 입력되는 데이터가 라벨링 된 상태거나 패딩이 추가되거나, 뭐 여러모로 데이터 일부를 걷어내고 사용할 때 쓰기 매우 유용하지 싶다.

<br><br><br>

+ ## Broadcasting
---

Broadcasting 기능도 매우 중요하고 유용한 기능이라 생각하는데, 얘는 행렬 연산 시 크기가 다르더라도 연산이 가능하게 해주는 기능이다. <br><br>
두 배열간의 연산을 할 때 각 배열은 조건을 충족해야 연산이 가능하다. 우선 덧셈과 뺄셈은 두 행렬의 크기가 같아야 하며, 곱셈은 두 행렬의 크기를 각각 [AxB], [CxD]라고 할 때 B와 C의 값이 같아야 한다. (앞 행렬 마지막 차원의 크기와 뒷 행렬 첫 차원 크기가 같아야 된다.)<br>
딥러닝을 하다보면 이런 조건들을 만족 못하는 경우가 한번씩 보이는데, 이는 __브로드캐스팅__ 기능으로 해결할 수 있다. <br>
 <br>
앞서 말했듯이 브로드캐스팅은 크기가 달라 원래라면 연산이 불가능했을 행렬을 연산 가능하도록 해주는 것인데, 행렬의 크기를 덧셈, 뺄셈, 곱셈 각각의 조건에 맞춰 늘려주는 기능이다. <br>
예를 들어, [1x2] 크기의 행렬이 있다고 할 때, 같은 크기의 행렬과의 덧셈은 문제없이 잘 되나, 만약 스칼라 값과 연산을 하려한다면 원래라면 불가능하다. 하지만 브로드캐스팅 기능으로 스칼라 값의 크기와 행렬의 크기를 맞춰 스칼라 값을 [1x2] 크기의 행렬로 늘려준다면 계산이 가능하게 된다. <br>
(ex: [1,2] + [3] => [1,2]+[3,3] = [4,5])
크기가 다른 행렬일 경우에도 브로드캐스팅하게 되면 조건에 맞춰 크기를 바꾼다.<br>

![Broadcasting](./assets/img/braodcast_visualization_img.png "Broadcasting example")

위의 그림은 브로드캐스팅을 시각화 한 것인데, 그림에서 흐릿하게 처리된 부분이 브로드캐스팅 된 부분이라고 생각하면 된다. <br><br>
하지만 브로드캐스팅을 할 때에도 조건이 있다.<br>
1. 대응되는 차원축의 길이가 각각 동일할 것
2. 대응되는 차원축의 길이가 하나라도 1일 것
<br><br>

위의 두 조건 중 하나를 반드시 만족해야 브로드캐스팅이 가능해지며, 하나라도 만족하지 못하면 기능이 동작하지않는다. <br>

```
example)
matrix A size : [1,1,3,4]
amtrix B size : [6,5,3,4]
matrix C size : [2,5,3,4]

matrix A와 B 연산 할 때는 broadcasting 기능 동작하나,
matrix B와 C 연산 할 때는 동작하지 않음.
B와 C 첫번째 차원 값들이 두 조건 모두 만족하지 않기 때문.
```
<br>
브로드캐스팅을 사용할 때는 각별히 주의해야한다. 왜냐하면 브로드캐스팅은 사용자가 따로 명시해서 사용하는 것이 아니라 자동으로 실행되는 기능이라, 실제로 텐서의 크기가 같아야 하는 상황에서 같지 않더라도 연산을 가능하게 해주어 오류를 제대로 포착할 수 없게된다.
<br><br><br>

+ ## 자주 사용되는 다른 기능들
---
Indexing, slicing, broadcasting 외에도 자주 사용되는 여러가지 기능들이 있다.<br>
 행렬간 곱셈에서 행렬곱이 아닌 행렬의 요소별 곱을 할 수 도 있고 행렬의 평균을 구할 수도, 합을 구하거나 최대값의 위치 혹은 최대값을 알아내는 함수가 있다. <br><br>

 + ### Matrix Multiplication & Element-wise Multiplication
 <br>
 행렬의 곱셈에는 두가지 방법이 있는데 일반적인 행렬곱과 행렬의 원소별로 곱하는 요소별 곱이 있다.<br><br>

+ Matrix Multiplication

그냥 행렬간 행렬곱이다. 앞서 말했듯이 앞행렬의 마지막 차원과 뒷 행렬의 첫번째 차원이 같은 크기를 가져야 한다. <br>
```python
Tensor1.matmul(Tensor2)
torch.matmul(Tensor1, Tensor2)
Tensor1 @ Tensor2
```

+ Element-wise Multiplication

행렬간 요소별 곱이다. 행렬에서 같은 인덱스 값을 가지는 위치의 원소들끼리 곱해주는 것이며, 행렬의 크기가 같지 않다면 브로드캐스팅 후 연산한다.<br>
```python
Tensor1.mul(Tensor2)
torch.mul(Tensor1, Tensor2)
Tensor1 * Tensor2
```
<br>

+ ### 평균
<br>

행렬의 평균을 구하는 기능도 있는데, 행렬 전체의 평균을 구할 수도 있고, 차원축을 따라서 평균을 구할 수도 있다. <br>

```python
Tensor.mean()
torch.mean(Tensor)
```

만약 차원축을 따라 평균을 구하고 싶다면 ```mean()``` 함수 안에 인자로 ```dim``` 값을 주면 된다.

```python
Tensor.mean(dim=0)  # 0번째 차원축을 따라 평균
torch.mean(Tensor, dim=1)  # 1번째 차원축을 따라 평균
```

<br>

+ ### 덧셈

평균을 구하는 방법과 매우 유사한 방식으로 합을 구할 수도 있다. 평균 함수와 마찬가지로 행렬 전체의 합을 구할 수도, 행렬 차원축을 따르는 합을 구할 수도 있다.<br>

```python
# 텐서 전체 합
Tensor.sum()
torch.sum(Tensor)

# 차원축 따라서 합
Tensor.sum(dim=0)
torch.sum(Tensor, dim=1)
```

<br>

+ ### 최대값 & 최대값 인덱스
행렬의 최대값과 최대값의 인덱스를 리턴하는 함수도 있다. <br>

```python
# Only Max value
Tensor.max()
torch.max(Tensor)

# (Max_value, Max_value_index)
Tensor.max(dim=0)
torch.max(Tensor, dim=0)
```

위 처럼 ```max()``` 함수에 인자를 아무것도 주지 않으면 최대값만을 리턴하나, 인자로 ```dim```을 같이 준다면 해당 차원축에 따른 최대값과 함께 최대값의 인덱스를 튜플로 리턴한다. 만약 최대값만을 리턴받고 싶다면 ```Tensor.max(dim=0)[0]```을 사용하면 되고, 최대값의 인덱스값만 받고 싶다면 ```Tensor.max(dim=0)[1]``` 을 사용하면 된다.<br><br>

<br><br>

+ ### torch.Tensor.view(*shape)
---
텐서의 원소와 원소 수는 유지하면서 모양을 바꿔주는 함수다. numpy의 ```reshape```과 같은 역할을 하며 딥러닝에서 매우 중요한 역할을 한다. <br>
텐서간 연산을 하거나 데이터 모양을 바꿔야 하는 경우가 발생할 경우 많이 쓰게 될 것이며 써야할 상황이 많이 생기게 될 것이다. <br>
우선 예시로 임의의 3차원 텐서가 있다고 가정할 때,

```python
T = torch.Tensor([ [[0,1,2],
                     [3,4,5]],
                    [[6,7,8],
                     [9,10,11]] ])  # shape : [2,3,3]
```

위의 3차원 텐서에 ```view()``` 함수를 사용해 모양을 바꾼다면

```python
T.view([-1,3])
T.view([-1,3]).shape
```

```python
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]])

torch.Size([4,3])
```
와 같은 결과가 나올 것이다. ```T.view([-1,3]``` 으로 3차원이였던 텐서를 2차원으로 바꾸게 된 것인데, 여기서 주어진 -1은 원소의 개수를 그대로 유지하는 값이 자동으로 설정되게 하는 인자이다. 위의 예시는 [2,2,3]의 크기를 가져 총 12개의 원소를 갖는 텐서를 두번째 차원 축이 3인 2차원 텐서로 바꾸려 하는 것이다. 이 때 원소의 개수를 유지하기 위해서는 첫번째 차원의 크기가 4가 되어야 하며 -1을 view 함수의 첫번째 차원에 해당하는 위치에 전달할 시 자동으로 첫번째 차원의 크기는 4가 된다. <br><br>
이는 같은 차원을 가지는 다른 형태의 텐서로 바꿀 때도 사용할 수 있다.

```python
T.view([-1,1,3])
T.view([-1,1,3]).shape
```

위의 함수를 적용할 시 앞서 말했듯이 텐서의 원소 수를 그대로 유지해야 하기 때문에 -1이 전달된 첫번째 차원의 크기는 4가 된다.

```python
tensor([[[0, 1, 2]],
        [[3, 4, 5]],
        [[6, 7, 8]],
        [[9, 10, 11]]])

torch.Size([4,1,3])
```

<br>

+ ## torch.Tensor.squeeze()
---
텐서의 차원축 길이가 1인 차원을 제거한다. 인자로 ```dim```이 주어지지 않으면 길이가 1인 차원을 모두 제거하기에 만약 batch dimension이 존재한다면 주의해서 사용해야 한다. <br>

```python
T = torch.Tensor([[ [0],
                    [1],
                    [2] ]])     # Shape : [1,3,1]
T.squeeze().shape
```

위와 같이 텐서의 모양이 [1,3,1]일 때 ```T.squeeze()``` 함수를 인자없이 사용한다면 1인 차원 전부가 제거되어 텐서의 모양은 [3]이 될 것이다.<br>

```python
T.squeeze(dim=0)
```

인자로 ```dim```값을 넘겨준다면 해당 차원의 길이가 1이라면 차원을 제거하고 만약 1 이상의 값을 갖는다면 그대로 리턴한다.<br><br>

+ ### torch.Tensor.unsqueeze()
---
```torch.Tensor.squeeze()``` 함수와 반대로 ```unsqueeze()```는 길이가 1인 차원을 하나 끼워 넣는다. 인자로 전달되는 ```dim```로 차원을 끼워넣을 위치를 전달하고, 해당 위치에 크기가 1인 차원을 넣는다. <br>
```torch.Tensor.squeeze()```와 함께 적절히 사용한다면 ```torch.Tensor.view()```와 비슷한 기능을 할 수 있다. 세 함수 모두 텐서의 원소나 원소 수는 변화시키지 않으면서 텐서의 차원을 조절한다. <br>

```python
T = torch.Tensor.([0,1,2])  # Shape : [3]
T.unsqueeze(0)  # 0번째 위치에 크기 1인 차원 추가
T.unsqueeze(1)  # 1번째 위치에 크기 1인 차원 추가
```

```python
tensor([[1, 2, 3]])  # T.unsqueeze(0) result  [1,3]

tensor([[1],
        [2],
        [3]])  # T.unsqueeze(1) result  [3,1]
```

```torch.Tensor.unsqueeze()``` 도 dim 인자에 -1을 줄 수 있다. -1을 주면 제일 마지막 차원 뒤에 크기가 1인 차원을 추가한다.<br><br>

+ ### Type Casting
---
Type casting 기느은 아직은 사용하는 것을 못봤다. 왠지 쓸일이 크게 없을 것 같은 느낌도 들긴하지만 크게 어려워 보이지는 않아서 알아만 보고 넘어가려 한다.<br>
텐서는 여러가지 자료형을 가질 수 있는데, 흔히 쓰는 소수값을 원소로 갖는 float tensor나 정수값을 원소로 갖는 int tensor등 여러가지 자료형을 가질 수 있다.<br>

![Type Casting](./assets/img/typecasting_img.png "Type Casting")

위 표에서 볼 수 있듯이 여러가지 자료형이 있으며 선언은 이 때까지 하던방식과 유사하다. 그저 ```torch.Tensor``` 대신에 ```FloatTensor```, ```LongTensor``` 등을 쓰면 된다. <br><br>
또는 만들어진 텐서의 자료형을 변경하는 방법도 있다. 만들어진 텐서 뒤에 ```.float()```이나 ```.long()```과 같이 붙여주면 자료형을 바꿔 줄 수 있다.<br><br>

+ ### Concatenate / torch.cat(), torch.stack()
---
Concatenate 기능도 꽤 많이 쓰이는 것 같다. 뜻 그대로 텐서를 연결하는 함수인데, 텐서를 연결하는 함수는 ```torch.cat()```, ```torch.stack()``` 두 가지가 있다. 때에 따라 ```cat()```보다 ```stack()```이 더 편리할 경우가 있는데 ```stack()```이 좀 더 많은 함수를 포함하기 때문이다. <br>
두 함수 모두 인자로 ```dim```을 받을 수도, 안받을 수도 있다. 전달받지않게 된다면 기본값인 0으로 자동으로 설정되고 첫번째 차원에 따라 이어준다.<br>

```python
T1 = torch.Tensor([[1, 2], [3, 4]])  # Shape : [2,2]
T2 = torch.Tensor([[5, 6], [7, 8]])  # Shape : [2,2]

torch.cat([T1, T2], dim=0)
torch.stack([T1, T2], dim=0)
```

위 처럼 [2,2]의 크기를 갖는 두개의 텐서를 선언해서 이어붙이면

```python
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])  # torch.cat([T1, T2], dim=0)

tensor([ [[1, 2],
          [3, 4]],
         [[5, 6],
          [7, 8]] ])  # torch.stack([T1, T2], dim=0
```

보이는 것 처럼 cat과 stack의 결과는 약간의 차이를 보인다. 경우에 맞게 잘 쓰는게 중요할 듯하다. 아마 나는 cat을 주로 쓰지않을까 한다. 사실 wikidocs에서 보기 전까지는 stack 함수는 사용하는 것을 본 적이 없다.
<br><br>

+ ### ones, ones_like / zeros, zeros_like
---
0이나 1로 가득 채워진 텐서를 쉽게 만드는 함수도 있다. 함수 인자로 텐서의 크기를 주거나 원하는 크기를 가진 텐서를 인자로 전달하면 해당되는 크기의 텐서를 만들어 준다. <br>
예시로 위의 concatenate 항목에서 만든 T1 텐서를 사용할 것이다. <br>

```python
torch.ones([5])
torch.ones_like(T1)

torch.zeros([2,3])
torch.zeros_like(T1)
```
```python
tensor([1, 1, 1, 1, 1])  # torch.ones([5])

tensor([[1, 1],
        [1, 1]])  # torch.ones_like(T1)


tensor([[0, 0, 0],
        [0, 0, 0]])  # torch.zeros([2,3])

tensor([[0, 0],
        [0, 0]])  # torch.zeros_like(T1)
```

<br><br>

+ ### In-place Operation
---
보통 함수를 이용해서 텐서 연산을 하면 바로바로 저장되는게 아니라 따로 ```T = T1.concat(T2)```와 같이 따로 값을 저장해주어야 한다.<br>
하지만 python에서는 연산 결과값을 기존의 텐서에 덮어씌울 수 있는 In-place operation을 제공한다. 따로 거창하게 할 것은 없고, 그냥 기존의 함수 뒤에 _를 하나 붙여주면 된다.<br>

```python
print(T1.mul(2))
print(T1)

print(T1.mul_(2))  # In-place Operation
print(T1)
```

위처럼 _를 붙여주게 되면 값 저장이 없더라도 자동으로 덮어씌워 연산결과가 저장된다. <br>

```python
tensor([[2, 4],
        [6, 8]])

tensor([[1, 2],
        [3, 5]])


tensor([[2, 4],
        [6, 8]])

tensor([[2, 4],
        [6, 8]])
```


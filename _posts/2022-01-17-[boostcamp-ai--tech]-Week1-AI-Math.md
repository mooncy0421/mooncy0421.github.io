---
layout: post
title: '[boostcamp ai-tech] Week1 AI Math'
categories: [boostcamp]
---

---

# Vector
---
+ **수학적 정의** : 유클리드 공간 내에서 크기, 방향을 가지는 양
+ **AI 분야 정의** : 입력되는 데이터를 나타내는 방법
+ **Python** : 숫자를 원소로 가지는 list

$$ X = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} \quad X^T = \begin{bmatrix} x_1, x_2, \cdots , x_d \end{bmatrix} $$

+ 순서대로 열벡터, 행벡터
+ **d** : 벡터의 차원 (x : d차원 내의 한점)

### Vector의 연산

주의) __벡터간 연산은 두 벡터의 차원과 모양이 같아야함__

> **덧셈/뺄셈** : 다른 벡터로부터 상대적인 위치 이동 <br>
> $$ X+Y = \begin{bmatrix} x_1 + y_1 \\ x_2 + y_2 \\ \vdots \\ x_d + y_d \end{bmatrix} \quad X^T + Y^T = \begin{bmatrix} x_1+y_1, x_2+y_2, \cdots , x_d + y_d \end{bmatrix} $$

> **스칼라곱** : 벡터의 방향은 바꾸지않고 길이만 변화시킴 (단, 음수값 곱하면 방향 반대됨) <br>
> **곱셈** : 같은 모양의 두 벡터는 성분곱(Hadamard/Element-wise Product) 가능함 <br>
> $$ X \odot Y = \begin{bmatrix} x_1 y_1, x_2 y_2, \cdots , x_d y_d \end{bmatrix} $$


### Norm

**Norm** : 벡터의 원점으로부터 거리 (임의의 차원 d에서 사용 가능)<br><br>

**L1 Norm** : 벡터 각 성분 변화량 절대값의 합<br>
$$ \parallel x \parallel _1 = \sum_{i=1}^d |x_i| $$

**L2 Norm** : 벡터 각 성분 제곱 합의 square root(피타고라스 정리 이용한 유클리드 거리) <br>
$$ \parallel x \parallel _2 = \sqrt{\sum_{i=1}^d |x_i|^2} $$

<br>
이외에도 L0 Norm, L-Infinity Norm이 있는데, 간단하게 말하자면 다음과 같다.
+ L0 Norm : 벡터에서 0이 아닌 원소의 총 갯수<br>
+ L-Infinity Norm : 원소 중 절대값 가장 큰 원소 <br><br>

일반적으로 L1과 L2 Norm이 가장 많이 사용된다고 한다. <br>
Norm은 종류에 따라 기하학적 성질이 달라지게 되는데, 각 norm의 성질에 따라 적절하게 사용하면 된다. <br>
**L1 Norm** : Robust 학습, Lasso 회귀 등<br>
**L2 Norm** : Laplace 근사, Ridge 회귀 등<br><br>

L1, L2 Norm은 딥러닝에서 모델의 loss를 구할 때에 사용이 된다. 각각을 L1, L2 loss라고 부른다.<br>
$$ L_1 = \sum_{i=1}^n |y_i - f(x_i)| \\
L_2 = \sum_{i=1}^n (y_i - f(x_i))^2 $$

**L1 loss vs. L2 loss**
> Robustness : Outlier가 등장했을 때, 즉 일반적인 데이터들의 패턴을 많이 벗어난 값이 나타났을 경우 loss function이 얼마나 영향을 받는지를 말한다.
> 수식을 보면 알 수 있듯이, L2 loss는 제곱한 값의 합을 loss로 사용하기 때문에 L1 loss에 비해 loss function의 값이 더 크게 변하게 될 것이다. 
> 그래서 outlier에 의한 영향을 적게 받으려하는 경우에는 L1 loss를 사용하는 것이 낫다. <br>
> **L1이 L2보다 robust**

> Stability : 모델이 비슷한 데이터들에 대해 일관적인 예측을 할 수 있는 정도를 말한다. 
> 학습된 모델이 새로운 데이터에 대해 예측을 시도할 때, 만일 outlier가 아닌 학습에 사용된 데이터와 비슷한 데이터가 입력된다면, loss는 L2가 L1에 비해 작아질 수 <br>
> 있다. ( $$ y_i-f(x_i) <= 1 $$ 일 경우) <br>
> 이 때 모델은 L1에 비해 L2 loss를 사용할 경우 학습시 더 적은 변화를 나타내어 좀 더 stable하다고 하게 된다. <br>
> **L2가 L1보다 stable** 
<br>
<br>
또한 L1, L2 Norm은 딥러닝에서 **Regularization**에도 사용이 되는데, regularization은 추후에 좀 더 자세히 다룰 것이라 자세한 <br>
설명은 나중에 하려한다. 간단하게 regularization은 모델의 overfitting을 막는 방법 중 하나로 사용이 되는데 Loss function에 <br>
Regularization Term을 더해줌으로 사용된다. (가중치가 너무 크지 않은 방향으로 학습되게 함)<br>
$$ \mathsf{L_1 \; Regularization} : \quad cost(W, b) = {1 \over m} \sum_i^m L(\hat{y}_i, y_i) + {\lambda \over 2} |w| \\
\mathsf{L_2 \; Regularization} : \quad cost(W, b) = {1 \over m} \sum_i^m L(\hat{y}_i, y_i) + {\lambda \over 2} |w|^2 $$

[참고 블로그](https://seongkyun.github.io/study/2019/04/18/l1_l2/)
<br><br>


### 벡터간의 거리/각도
> **벡터 사이의 거리** : L1/L2 Norm으로 벡터 사이의 거리를 계산함 (순서 상관 X)
> $$ \parallel x-y \parrallel $$

> **벡터 사이의 각도** : 제 2 코사인 법칙 이용 (d-차원에서도 가능하게 함) <br>
> $$ \cos \theta = { \parallel x \parallel ^2 _2 + \parallel y \parallel ^2 _2 - \parallel x-y \parallel ^2 _2 \over 2 \parallel x \parallel _2 \parallel x \parallel _2} = { 2<x,y> \over 2 \parallel x \parallel _2 \parallel x \parallel _2 } $$ <br>
$$ <x,y> = \sum_{}^{} x_iy_i $$

<br>

### 내적
> **내적(Inner Product)** : 정사영(Orthogonal Projection)된 벡터의 길이와 관련됨 <br>
> $$ Proj(x) = \parallel x \parallel \cos \theta $$ <br>
> 내적 : Proj 길이 $ \cos \theta $ 만큼 조정 $$ <x, y> = \parallel x \parallel _2 \parallel y \parallel _2 \cos \theta $$

<br><br><br>

# Matrix
+ **수학적 정의** : 수 또는 다항식 등을 직사각형 모양으로 배열한 것
+ **Python** : 벡터를 원소로 가지는 2차원 배열 (numpy에서는 행이 기본 단위가 된다)<br><br>

나는 행렬만큼 인공지능 분야에서 데이터를 쉽게 다룰 수 있게 해주는 방법은 없다고 생각한다.

$$ X = \begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_n \end{bmatrix} \quad = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1m} \\ x_{21} & x_{22} & \cdots & x_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ x_{n1} & x_{n2} & \cdots & x_{nm} \end{bmatrix} $$

<br>
위의 행렬은 n행과 m열을 가진 nxm 행렬이라고 부른다.(m차원 벡터 n개) <br>
각 원소들의 인덱스는 $ x_{ij} $ 와 같이 나타낼 수 있는데, 이는 i번째 벡터의 j번째 원소라고 보면 된다.<br>

+ **행렬의 덧셈, 뺄셈, 성분곱, 스칼라곱은 벡터의 연산과 똑같이 진행된다.** <br><br>

### 행렬곱 (Matrix Multiplication)
$$ XY = \begin{pmatrix} \sum_{k} x_{ik}y_{kj} \end{pmatrix} $$ <br>
행렬곱에서는 **X의 열 수**와 **Y의 행 수**가 다르면 계산이 불가능함을 주의해야한다.<br>

+ 행렬의 내적 : $$ XY^T = \sum_{k} x_{ik}y_{jk} $$

<br><br>

### 선형 변환 (Linear Transform)
행렬은 데이터의 차원을 변환시키는 연산자로도 이해할 수 있다. 데이터의 차원을 바꿔 표현하는 것을 선형 변환이라고 한다. (패턴 추출, 데이터 압축)<br>
모든 선형 변환은 행렬의 곱으로 표현이 가능하며, 이는 딥러닝 모델의 일반적 구조인 **선형 변환 + 비선형 함수**에 유용하게 쓰인다. <br>
$$ Z_i = \sum_{j} a_{ij}x_j $$ <br>
위 연산은 데이터 x를 z로 표현하게 만드는 선형 변환 연산이다. <br><br>

### 역행렬 (Inverse Matrix)
행렬 A의 연산(선형 변환)을 원래대로 돌리는 연산으로 볼 수 있다. <br>
$$ AA^{-1} = A^{-1}A = I $$ <br>
이처럼 역행렬과 원래의 행렬을 곱하면 항등행렬이 나온다.<br><br>

### 유사 역행렬 / 무어-팬로즈 역행렬
$$ A = \mathbb{R}_{n \times m}(M) $$ <br>
행렬이 정사각행렬이 아니라 역행렬을 구하지 못할 경우 사용된다. <br>
$$ n \ge m, \qquad A^{+}=(A^TA)^{-1}A^T \qquad A^{+}A = I \\
n \le m, \qquad A^{+}=A^T(AA^T)^{-1} \qquad AA^{+} = I $$ <br>
+ 유사 역행렬을 이용해서 연립방정식의 해를 구하거나 선형 회귀 분석에 이용할 수 있다. <br>
    - 연립 방정식 해 구하기<br>
    - $$ n \le m, \quad Ax=b \\ Ax=b \Rightarrow x=A^{+}b = A^T(AA^T)^{-1}b $$ <br>
    - 선형회귀 분석 <br>
    - $$ n \ge m, \quad X \beta = y \qquad (X : \mathsf{data}, \; \beta: \mathsf{coefficient}, \; y: \mathsf{label}) \\ X \beta = \hat{y} \approx y \Rightarrow \beta =X^{+}y = (X^TX)^{-1}X^Ty \\ \min \parallel y - \hat{y} \parallel _2 $$ <br>

<br><br>
원래 벡터나 행렬은 어느정도 알고 있어서 그렇게 어렵지는 않았다. 하지만 Norm에 관한 부분은 강의를 듣고 좀 더 알아보니 상상 <br>
이상으로 수학적으로 많이 부족하다는 것을 깨달았다. 추후 loss function과 regularization 파트가 나오면 그 때 다시 한번 제대로 정리해야겠다.
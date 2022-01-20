---
layout: post
title: '[boostcamp ai-tech] Week1 AI Math (2)'
categories: [boostcamp]
---

---

이번 강의는 경사하강법과 딥러닝의 학습 방법에 관한 내용이였다. <br><br>

# Gradient Descent
---
경사하강법은 최적화 알고리즘 중 하나이다. <br>
기계 학습에서 대표적인 최적화 기법 중 하나이다. 간단하게 말하면 함수의 접선 기울기를 구해서 함수값 $f(x)$가 감소하는 방향으로 $x$를 서서히 이동시키며 함수의 극소값을 찾아가는 과정을 일컫는다. <br><br>

### Differentiation
앞서 말했듯이 경사하강법은 함수 접선의 기울기를 이용한다. <br>
$$ f'(x) = lim_{h \to 0} {f(x+h) - f(x) \over h} $$
위는 미분을 극한으로 정의한 것이다. <br>
위 식은 함수 $f$의 점 $(x, f(x))$에서 접선의 기울기를 의미한다. 이 기울기를 gradient라고 부른다.<br>
미분값을 구하려면 함수가 연속 함수여야한다. <br><br>

미분을 통해 구한 gradient로 함수의 증가/감소 방향을 알아낼 수 있다. gradient가 음수라면 $x$가 증가할수록 함수값이 감소하고, 양수일 경우 $x$가 증가할수록 함수값이 증가하게 된다. <br>
경사 하강법을 이용한 최적화는 $x - f'(x)$로 함수값을 감소시킨다. <br>
경사 상승법의 경우에는 $x + f'(x)$로 함수값을 증가시킨다. <br><br>

실제 사용시에는 gradient에 학습률(learning rate)을 곱해 함수의 수렴속도를 조절해준다. <br>

$$ x = x - \lambda f'(x) $$

<br><br>

**입력이 벡터인 경우** <br>
입력이 벡터인 다변수 함수의 경우 편미분을 사용한다.<br>

$$ \nabla f = (\partial_{x_1}f, \partial_{x_2}f, \cdots, \partial_{x_d}f) $$

위의 $ \nabla f $ 를 gradient vector라고 한다. <br>
Gradient vector는 함수의 극소값으로 향하는 정확한 방향을 가리키지는 않지만 가장 빠른 속도로 수렴할 수 있는 방향을 나타낸다. <br><br><br>


### Linear Regression (with gradient descent)
---
이전 포스팅에서 설명했듯이 선형회귀 모델은 유사 역행렬을 이용하여 구할 수 있었다. 하지만 과정이 워낙 복잡해 사용하기가 쉽지않다. 하지만 gradient descent를 이용하면 쉽게 선형회귀 모델을 만들 수 있다. <br>
선형회귀 모델은 $ y = xW $ 식이 데이터를 가장 잘 근사하는 $ W $를 찾는 것이 목적이다. 이는 다시말해 $ ||y-xW||_2 $ 값을 최소화하는  $W$를 찾으면 된다는 것이다. 여기서 $ ||y-xW||_2 $를 **선형회귀의 목적식**이라고 한다. $ ||y-xW||_2 $를 최소화 하기 위해서는 함수값이 감소하는 방향으로 $W$ 값을 이동시키면 되는데, 이는 앞서 설명한 gradient descent의 방법이다.<br><br>
$$ \mathsf{Linear Model} : y = X\beta \\
\begin{bmatrix} -x_1- \\ -x_2- \\ \vdots \\ -x_n- \end{bmatrix} \begin{bmatrix} \beta_1 \\ \beta_2 \\ \vdots \\ \beta_m \end{bmatrix} = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{bmatrix} \\ 
\nabla_\beta ||y-X\beta||_2 = (\partial_{\beta_1}||y-X\beta||_2, \partial_{\beta_2}||y-X\beta||_2, \cdots, \partial_{\beta_d}||y-X\beta||_2) $$

<br><br>

Gradient vector를 구하는 계산식은 다음과 같다. <br><br>
$$ \nabla_\beta ||y-X\beta||_2 = (\partial_{\beta_1}||y-X\beta||_2, \partial_{\beta_2}||y-X\beta||_2, \cdots, \partial_{\beta_d}||y-X\beta||_2) \\
\partial_{\beta_k}||y-X\beta||_2 = \partial_{\beta_k} \left\{ {1 \over n} \sum_{i=1}^n \left( y_i - \sum_{j=1}^d X_{ij}\beta_j \right)^2 \right\}^{1 \over 2} \\ 
= -{X_{.k}^T(y-X\beta) \over n||y-X\beta||_2} \\
\nabla_\beta ||y-X\beta||_2 = \left( -{X_{.1}^T(y-X\beta) \over n||y-X\beta||_2}, \cdots, -{X_{.d}^T(y-X\beta) \over n||y-X\beta||_2} \right) \\
= -{X^T(y-X\beta) \over n||y-X\beta||_2}
$$

<br>
선형회귀의 목적식을 최소화하는 $\beta$를 구하는 경사하강법은 다음과 같이 나타낼 수 있다. <br><br>
$$ \beta^{(t+1)} = \beta^{(t)} - \lambda\nabla_\beta ||y-X\beta^{(t)}||_2 \\
\beta^{(t+1)} = \beta^{(t)} + {\lambda \over n}{X^T(y-X\beta^{(t)}) \over ||y-X\beta^{(t)}||} $$
<br>

식을 간소화하려면 $\nabla_\beta ||y-X\beta||_2^2$ 를 사용하면 된다. <br><br>
$$ \beta^{(t+1)} = \beta^{(t)} + {2\lambda \over n}X^T(y-X\beta^{(t)})  $$
<br><br>

### Stochastic Gradient Descent(SGD)
기계학습에서 경사하강법을 그대로 사용하게 되면 데이터를 전부 다 메모리에 업로드하게 되어 Out-of-memory 문제가 발생하게 <br>
된다. 이를 방지하기 위해 모든 데이터가 아닌 일부 데이터인 **mini-batch**를 사용해서 parameter를 업데이트하는 방식을 고안했고, 이를 SGD라고 한다. <br>
SGD는 한번 업데이트시에 데이터 일부만 이용해서 parameter를 업데이트하기 때문에 메모리를 효율적으로 활용하는데 도움이 <br>
된다. 또한 매 업데이트마다 다른 mini-batch를 사용하기 때문에 목적식 그래프의 모양이 가변적이게 된다. 이는 극소점의 위치가 <br>
**확률적**이게 되어 최소값이 아닌 극소점을 탈출하는데에 도움을 준다. <br>
SGD는 알고리즘적 효율성 뿐만아니라 하드웨어적 한계를 고려했을 때 반드시 필요한 방법이다. 
<br><br><br>


# 딥러닝 학습 방법
---
앞 부분에서는 선형 모델을 이용하여 데이터를 해석했다. <br>
하지만 딥러닝에서 사용되는 모델들은 **비선형 구조**를 갖는다. 이는 간단하게 보면 선형 모델에 비선형 함수인 **Activation function**을 이어붙여 만드는 것이라 할 수 있다. <br>

$$ 
H = (\sigma(\mathbf{z}_1), \cdots, \sigma(\mathbf{z}_n)) \ , \quad \sigma(\mathbf{z}) = \sigma(\mathbf{Wx}+\mathbf{b})
$$

이 때 $\sigma$ 는 활성화 함수로 잠재 벡터(latent vector) $\mathbf{z}=(z_1, \cdots, z_q)$ 의 각 노드에 개별 적용해 새 잠재 벡터인 <br>
$ H = (\sigma(\mathbf{z}_1), \cdots, \sigma(\mathbf{z}_n)) $ 를 만든다. <br>

### Activation function
> + 비선형 함수를 사용한다.
> + 실수값을 입력받아 실수값을 출력한다. 
> + 딥러닝에서 매우 중요한 개념 중 하나로 활성화 함수없이 모델을 만들게 되면 아무리 깊은 모델을 만들더라도 선형 모델과 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 다를 바가 없어진다. 
> + 종류로는 sigmoid, tanh, ReLU가 있으며, sigmoid, tanh는 전통적으로 많이 사용되었고, 딥러닝에는 ReLU를 많이 사용한다. <br>
> **Sigmoid** : $ \sigma(x) = {1 \over 1+e^{-x}} $ <br>
> **tanh** : $ \tanh (x) = {e^x - e^{-x} \over e^x + e^{-x}} $ <br>
> **ReLU** : $ \mathrm{ReLU}(x) = \max \{0, x\} $ 

<br><br>

이렇게 선형 모델과 비선형 함수인 활성화 함수를 이어 붙인 하나의 블록을 **Perceptron**이라고 칭하며 이러한 Perceptron을 여러층 쌓아올려 기본적인 딥러닝 신경망인 **MLP**(Multi-layer Perceptron)을 구성하게 된다. <br>
Layer를 왜 여러개 쌓아 올려 모델을 구성할까?<br>
이론적으로는 2층 신경망만으로도 임의의 연속 함수 모델을 근사할 수 있다고 한다. (Universal approximation theorem) 하지만 이는 결국 이론적으로만 가능하기에 실제로 모델을 만들 때에는 절대 성능을 보장하지 못한다. 그래서 Perceptron을 여러층 쌓아올려 깊은 신경망을 만들게 되는데, 그렇다고 무조건 깊게만 쌓는다고 해서 성능이 뛰어난 모델을 만들 수 있는 것은 아니게 된다. 층이 너무 깊어지게 될 경우 신경망에 사용되는 parameter들의 최적화가 어려워져 학습의 난이도가 올라가게 된다. 그래서 **적절한 깊이의 신경망을 구성하는 것이 중요**하며, 적절한 깊이의 신경망을 만들 경우 비교적 적은 parameter 수로 복잡한 함수의 표현이 가능해지게 된다. (층이 너무 얕으면 함수 표현에 필요한 뉴런의 숫자가 늘어나 parameter가 많은 넓은 신경망이 된다)
<br><br>

이렇게 L개의 층을 쌓은 다층 신경망은 다음의 수식과 같이 표현할 수 있다.

$$
\mathbf{O} = \mathbf{Z}^L \\
\vdots \\
\mathbf{H}^{(l)} = \sigma(\mathbf{Z}^{(l)}) \\
\mathbf{Z}^{(l)} = \mathbf{H}^{(l-1)} \mathbf{W}^{(l)} + \mathbf{b}^{(l)} \\
\vdots \\
\mathbf{H}^{(1)} = \sigma(\mathbf{Z}^{(1)}) \\
\mathbf{Z}^{(1)} = \mathbf{H}^{(1)} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}
$$

<br>
이 때 각 층에 사용된 parameter $$ \left\{ \mathbf{W}^{(l)},\mathbf{b}^{(l)} \right\}^L_{l=1} $$ 를 학습하기 위해서는 **Backpropagation** 알고리즘을 사용한다. <br><br>

### Backpropagation (역전파)
> + 역전파 알고리즘은 연쇄법칙에 기반한 자동미분을 사용한다. 
> 
> $$ z = (x+y)^2 \\ {\partial z \over \partial x} = ? \\ z = w^2 , \quad w = x + y \\  \Rightarrow {\partial z \over \partial x} = {\partial z \over \partial w} {\partial w \over \partial x} $$
> 
> + 위의 수식처럼 연쇄법칙을 사용하면 합성함수의 미분을 쉽게 계산할 수 있고, 이를 이용해 MLP 신경망의 각 층별 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; parameter의 gradient를 구해 경사하강법을 수행할 수 있다.
> + 역전파 알고리즘을 수행할 때에는 최상단 layer의 gradient vector부터 역순으로 차례차례 계산하면 되고, 각 층의 gradient는 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 아래층으로 전달되게 된다. 
> + 아마 gradient가 아래층으로 갈 수록 곱해지기 때문에 층이 너무 깊으면 gradient가 너무 커지거나 작아져서 학습이 잘 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 안되는 것 같다. 
> + 딥러닝 모델에서의 역전파는 손실 함수를 L이라 할 때. $ \partial L over \partial W^l $ 을 계산할 때 사용된다. 
> + 이 때 $ W^l $ 은 l번째 층의 가중치 행렬로, 각 성분에 대한 편미분을 이용해 구해야 한다. <br>
> [참고 영상 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8)

<br><br>

이렇게 MLP 신경망을 구성하고나면 출력이 나오게 되는데, 이 출력은 그대로 사용하지 않고 특정 함수를 이용해서 원하는 형태의 <br>
출력이 나오도록 만들어준다. 그 중에서 분류 문제 모델에서는 **softmax**라는 함수를 사용한다. <br><br>

### Softmax
> + 분류 문제에 사용되는 함수이다.
> + 선형 모델에 사용시 출력되는 결과를 원하는 의도대로 바꾸어 해석할 수 있다.
> + 모델의 출력을 확률로 해석할 수 있도록 변환해주는 함수이다.
> + 분류 문제에서 예측 시 다음과 같이 각 출력의 확률 꼴로 나타낸다.
>
> $$ \mathrm{softmax}(\mathbf{o}) = \left( {\exp (o_1) \over \sum_{k=1}^p {\exp (o_k)}}, \cdots, {\exp (o_p) \over \sum_{k=1}^p {\exp (o_k)}} \right) $$
>
> 이는 **확률벡터가 k번째 특정 클래스에 속할 확률**로 해석이 가능하다.

<br><br>
이번 강의에서는 딥러닝 모델의 근본(?)이라고 할 수 있는 경사하강법과 역전파 알고리즘에 대해 학습했다. 예전에 처음 인공지능에 대해 공부할 때 봤던 내용이였고, 아주 익숙한 파트였지만 이렇게 수학적으로 깊게 파고들었던 적은 처음이라 많이 생소하기도 하고 어렵게 다가왔다. *강의 없는 주말에 복습하면서 한번 더 정리해야겠다...*
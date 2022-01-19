---
layout: post
title: '[boostcamp ai-tech] Week1 AI Math (2)'
categories: [boostcamp]
---

---

이번 강의는 경사하강법에 관한 내용이였다. <br><br>

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
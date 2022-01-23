---
layout: post
title: '[boostcamp ai-tech] Week1 AI Math (3)'
categories: [boostcamp]
---

---

이번에 학습한 내용은 기초 확률론, 통계학 그리고 기초 베이즈 통계학이였다. <br>
고등학교 때 부터 수학을 안좋아하기도 했고 이해하는데 힘들어했었기에 이번 강의는 정말 열심히 들어보려고 했다... 그래도 여전히 어렵고 이해도 안되고 힘든 건 어쩔 수가 없었다...<br><br><br>


# 확률론
---
딥러닝은 확률론 기반의 기계학습 이론에 바탕을 두고 있다고 한다. <br>
기계학습에서 사용되는 손실함수들의 작동 원리는 데이터 공간의 통계적 해석으로 유도할 수 있으며, 모델을 통한 예측이 틀릴 위험을 최소화하도록 데이터에 대해 학습하는 것을 기계학습의 기본적인 원리로 볼 수 있다. <br><br>

예를 들어, 회귀 분석의 loss로 사용되는 L2-Norm은 모델의 예측과 데이터 간 발생할 수 있는 오차(예측오차의 분산)를 최소화하는 방향으로 학습된다. <br>
그리고 분류 문제에 사용되는 cross entropy는 모델 예측의 불확실성, 그러니까 모델의 예측결과와 실제 데이터 간의 차이를 최소화하는 방향으로 학습된다. <br>
결국 둘 다 예측이 틀릴 가능성을 최소화시키려는 방향의 학습을 시행하는 것으로 볼 수 있어 확률론에 기반한다고 볼 수 있다. 이 분산과 불확실성을 줄이기 위해서는 이 둘을 측정하는 방법을 알아야한다. (이 방법들이 통계학에서 나온다)
<br><br>

### 확률분포
> + 확률분포는 데이터 공간을 나타내는 것으로 볼 수 있다.
> + 데이터 공간을 $ \mathcal{(X, Y)} $라 표기 하고 데이터를 추출하는 분포를 $ \mathcal{D} $, $ (\mathbf{x}, \mathrm{y}) \in \mathcal{X} \times \mathcal{Y} $를 데이터 공간 상 관측 가능한 데이터<br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;라고 하자.
> + 이 때, $(\mathbf{x},\mathrm{y}) \sim \mathcal{D}$를 확률변수라 하고 $\mathcal{D}$를 확률분포라 한다.
> + $\mathcal{D}$는 이론적으로 존재하는 확률분포이기 때문에 사전에 알 수 없으며 우리는 이 $\mathcal{D}$를 결합분포 $P(\mathbf{x},\mathrm{y})$를 통해 모델링 <br>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;해야한다. <br>
> + 딥러닝 모델의 목표는 데이터 $\mathbf{x}$가 입력되었을 때 데이터의 정답값 $\mathrm{y}$가 나타날 확률, $P(\mathrm{y} \mid \mathbf{x})$를 최대화 하는 것이다. (베이즈 통계학 이용)

<br>

**분류 문제**의 Softmax 함수 $\mathrm{softmax}(\mathbf{W}\phi+b)$는 데이터로부터 추출한 특징패턴과 가중치를 곱해 해당 데이터가 각 클래스에 <br>속할 확률을 조건부 확률로 계산한다. ($P(\mathrm{y} \mid \mathbf{x})$)<br><br>

회귀문제의 경우에는 조건부 기댓값인 $\mathbb{E}[y\mid\mathbf{x}]$를 추정한다. 이 조건부 기댓값은 $ \mathbb{E}\parallel y-f(\mathbf{x})\parallel _2 $ 를 최소화하는 함수 $f(\mathbf{x})$와 일치한다. 증명은 다음과 같다.  [참고영상](https://www.youtube.com/watch?v=7V170gTR1YE)<br><br>

> + 조건부 기댓값 = MSE 최소화와 동일함 <br>
> $$ \mathbb{E}\left[\left(y-\mathbb{E}[y \mid \mathbf{x}]\right)^2\right] \le \mathbb{E}\left[ \left(y-f(\mathbf{x})\right)^2 \right] \, ,\\
 \mathbb{E}\left[ \left(y-f(\mathbf{x})\right)^2 \right] = \mathbb{E}\left[ \left(y-\mathbb{E}[y \mid \mathbf{x}]+\mathbb{E}[y \mid \mathbf{x}]-f(\mathbf{x})\right)^2 \right] \\ 
 = \mathbb{E}\left[\left(y-\mathbb{E}[y \mid \mathbf{x}]\right)^2\right] + \mathbb{E}\left[\left(\mathbb{E}[y \mid \mathbf{x}]-f(\mathbf{x})\right)^2 \right] + 2\,\mathbb{E}\left[y - \mathbb{E}[ y \mid \mathbf{x}]\right]\mathbb{E}\left[ \mathbb{E}[ y \mid \mathbf{x}] -f(\mathbf{x}) \right] \, , \\ 
 \text{By te law of iterated expectations,} \\
 \mathbb{E}\left[ \left(y-\mathbb{E}[y \mid \mathbf{x}]\right) \left( \mathbb{E}(y \mid \mathbf{x})-f(\mathbf{x}) \right) \mid \mathbf{x} \right] = 0 \\ 
 \mathbb{E}\left[ \left(y-f(\mathbf{x})\right)^2 \right] = \mathbb{E}\left[\left(y-\mathbb{E}[y \mid \mathbf{x}]\right)^2\right] + \mathbb{E}\left[\left(\mathbb{E}[y \mid \mathbf{x}]-f(\mathbf{x})\right)^2 \right] \\ 
 \mathbb{E}\left[\left(\mathbb{E}[y \mid \mathbf{x}]-f(\mathbf{x})\right)^2 \right] \ge 0 \, ,\text{해당 항 최소화하려면}, \quad \mathbb{E}[y \mid \mathbf{x}]=f(\mathbf{x}) \\ 
 \therefore \mathbb{E}[y \mid \mathbf{x}]=f(\mathbf{x}) $$

<br>
기계 학습의 대부분의 문제들은 확률 분포를 모르는 경우가 많다. 확률 분포를 모르는 채 데이터를 이용해 기댓값을 구하려면 <br>
Monte-Carlo Sampling을 이용하면 된다. <br>

$$ \mathbb{E}_{\mathrm{x}\sim P(\mathbf{x})}[f(\mathbf{x})] \approx {1 \over N} \sum_{t=i}^N f(\mathbf{x}^{(i)}) $$

<br>
몬테 카를로 샘플링은 각 추출이 독립적임만 보장된다면 해당 함수 $f(\mathbf{x})$로의 수렴을 보장한다. <br><br>

몬테-카를로 샘플링으로 원주율은 어떻게 구할 수 있을까?<br>
몬테-카를로 샘플링은 결국 무한히 많은 수의 데이터를 샘플링하게 되면 원하는 함수에 매우 근접하게 근사할 수 있는 방법이다. <br>
그래서 원주율을 구하기 위해서는 반지름이 1인 원과, 그 원에 접하는 한 변의 길이가 2인 정사각형을 그리고, 정사각형 내에 무수히 많은 점들을 찍어나가 원의 중심으로부터의 거리가 1인 점과 아닌점들의 비율을 구하면 원주율값을 구할 수 있게 된다. <br><br><br>


# 통계학
---
### 통계적 모델링
기계학습과 통계학의 공통적인 목표이다. 적절한 가정아래 확률 분포를 추정해내는 것이다. 하지만 유한 개의 데이터를 관찰하여 모집단의 분포를 완벽하게 알아내는 것이란 사실상 불가능하며, 근사적으로 확률 분포를 추정하는 수 밖에 없다. 이렇게 근사적으로 확률 분포를 추정하는 방법을 크게 두 가지로 나눌 수 있는데, 각각을 **모수적 방법론**, **비모수적 방법론**이라 한다.<br><br>

#### 모수적 방법론
**모수적 방법론**이란 데이터가 특정한 확률분포를 따를 것이라 미리 가정(a priori)하고 그 분포를 추정하는 parameter(모수)를 추정해나가는 방법을 말한다. <br>
#### 비모수적 방법론
**비모수적 방법론**은 모수적 방법론과 다르게 미리 확률 분포를 가정하지 않고 데이터에 따라서 모델 구조와 paramter의 갯수가 바뀌는 것을 말한다. (비모수라고 모수가 없는 것은 아니며, 무수히 많을 수도 있다)<br>
기계학습의 많은 방법론들은 이 방법론에 속한다. <br><br>

### 확률 분포 가정법
비모수적 방법론으로 확률 분포를 추정하는 단계는 다음과 같다. <br><br>

1. 데이터가 분포된 모양을 히스토그램으로 관찰한다.
    + 각 확률 분포는 데이터가 가질 수 있는 값, 혹은 데이터가 갖는 값들의 범위가 정해진 경우가 많다. 
    + 이를 활용해서 데이터의 확률 분포를 추정할 수 있다.
2. 데이터의 생성 원리를 고려하여 확률 분포를 가정한다.
3. 가정한 확률 분포의 parameter를 데이터로 추정한다. 
4. 추정된 parameter를 각 분포에 맞는 검정 방법으로 검정한다.

<br>
위 단계의 parameter 추정 단계에서 사용할 수 있는 방법 중에는 **최대가능도 추정법(Maximum Likelihood Estimation, MLE)**이 있다.<br><br>

### Maximum Likelihood Estimation
이론적으로 가장 가능성이 높은 paramter를 추정하는 방법 중 하나이다.<br>

$$ \hat{\theta}_{\mathsf{MLE}} = \argmax_{\theta}L(\theta ; \mathbf{x}) = \argmax_{\theta}P(\mathbf{x}|\theta) $$

<br>
위 식에서 $ L(\theta ; \mathbf{x}) $ 는 parameter $\theta$를 따르는 분포가 데이터 $\mathbf{x}$를 관찰할 가능성을 뜻한다. (확률로 해석하면 안된다) <br>
MLE는 최적화 시 우선 데이터 집합인 $\mathbf{X}$가 독립 추출되어야하며 조건부확률들의 곱으로 나타낼 수 있다. <br>

$$ L(\theta ; \mathbf{X}) = \prod_{i=1}^n P(\mathbf{x}_i | \theta) $$

<br>
실제로 최적화 시에는 확률의 곱이 아닌 로그가능도(log-likelihood)를 최적화하는 방식으로 진행된다. <br>

$$ \log L(\theta ; \mathbf{X}) = \sum_{i=1}^n \log P(\mathbf{x}_i | \theta) $$

이는 데이터값이 매우 클 경우나 작을 경우, 컴퓨터로는 정확하게 likelihood를 계산할 수 없기에 log를 이용해 덧셈으로 바꿔 계산을 진행한다. 또한 경사하강법을 이용한 likelihood 최적화에서 연산량을 $ O(n^2) \rightarrow O(n) $로 줄여 효율을 올려준다.(대부분의 손실함수는 경사하강법을 이용하여 negative log-likelihood를 최적화함) <br><br>

<br>
항상 나는 확률과 통계를 되게 어려워했다. 이 포스팅에는 없지만 베이즈 통계학의 기본식도 이해하기 꽤 힘들어했었고, 매번 볼 때마다 항상 낯설게 다가왔다. 아마 이번에 학습한 내용도 시간이 어느정도 지나고나면 서서히 잊혀지지않을까 생각이 들긴한다. 그래서 그럴 때 마다 다시 들여다 볼 수 있게 정리를 해두어 나중에도 유용하게 쓸 수 있게 정리해뒀다. 뭐 아예 다까먹으면 또 정리하고 또 공부하고 해야겠지만...<br>
---
layout : post
title : '[boostcamp ai tech] Week 4 VAE'
categories : boostcamp
---

---

이번 주는 Deep Learning Basic 강의를 듣는 주차였다. <br>
Optimization을 비롯해 고전적인 MLP 구조부터 CNN, RNN, Trasnformer, VAE, GAN까지 참 많은 모델을 다루었다. MLP, CNN, RNN은 <br>
꽤나 자주봤었던 모델 구조여서 심하게 어렵게 다가오지는 않았고, Transformer는 팀원들과 논문 스터디를 진행해서 나름 괜찮았다. 하지만 VAE, GAN 특히 VAE는 상당히 어려웠고, 꽤나 많은 시간을 들여 조금 이해한 것 같았다. <br><br>

# AutoEncoder (AE)
---
VAE는 AE(Autoencoder)와 이름이 상당히 비슷해서 AE의 업그레이드 버전이겠거니 생각했는데, 그게 아니라고 한다. <br>
간단히 요약하자면 둘은 목적이 다른데, <br>
> + **Autoencoder** : 원 데이터를 차원축소하여 데이터를 잘 표현하는 Manifold를 학습하는 것
> + **Variational Autoencoder** : 데이터의 확률분포를 근사하여 데이터를 생성해내는 generative model을 학습 하는 것

<br>
둘 다 Encoder-Decoder 구조로 이루어져있고, 모델의 생긴 꼴이 매우 비슷하여 둘을 같은 모델 종류로 헷갈릴 수 있을 것 같다. 하지만 네이버 D2 유튜브 채널에 올라온 강의를 어느정도 본 결과 둘은 다른 모델임을 이해할 수 있었다. <br>

[해당 강의 링크](https://www.youtube.com/watch?v=o_peo6U7IRM&t=2514s) <br><br>

그럼 우선 Autoencoder부터 이해한대로 설명을 적어보자면<br>
Autoencoder는 말했듯이 Encoder-Decoder 구조로 이루어진 모델이다. 각각의 역할을 보면, <br><br>
+ __Encoder__ 

입력으로 데이터를 받아 Encoder network에 통과시키면 데이터가 압축된 latent variable z를 얻는다. 이 z는 입력 데이터의 차원 수 <br>
보다 더 적은 차원을 갖게 되는데 이는 학습이 잘 되었을 경우 원래 데이터의 특성을 잘 표현할 수 있는 vector가 된다. <br>
이렇게 잘 학습된 데이터의 저차원의 벡터들은 **manifold**를 잘 찾도록 해준다. <br>
여기서 manifold는 manifold hypothesis에서 출발한다. <br>
고차원 데이터를 데이터 공간에서 표현하게 된다면 각 sample들의 특성을 잘 반영한 저차원 공간(subspace)가 존재할 것이고, 그 <br>
저차원 공간을 manifold라고 한다. <br>
고차원 데이터 공간에서는 데이터간의 관계를 쉽게 찾기가 힘들다. <br>
유사한 데이터끼리 묶으려 할 때 고차원 데이터 공간상에서 해당 작업을 진행하게 되면 데이터들의 지배적인 특징(dominant feature)을 제대로 반영하지 못하여 제대로된 군집을 형성하기 힘들어진다. 하지만 dominant feature를 잘 반영한 저차원 공간에 <br>
데이터들을 매핑하여 subspace를 표현하게 되면 해당 subspace에서 가까운 거리에 있는 데이터들은 서로 비슷한 특성을 공유하게 될 것이다. <br>
서론이 길었는데, autoencoder에서 encoder는 이러한 manifold를 찾기 위한 저차원의 latent variable을 찾는 역할을 진행한다. <br><br>

+ **Decoder** <br>
Decoder는 encoder에서 만든 latent variable을 이용하여 원래의 데이터를 복원하며 학습한다. <br>
사실상 decoder는 생성 모델의 역할을 수행한다고 보면된다. 복원한 데이터 $y$와 원 데이터 $x$의 차이가 최소가 되는 방향으로 학습을 진행하며 계산된 loss를 reconstruction error라고 부른다. <br>
이러한 학습 방식을 사용하게 되면 decoder는 원본 이미지에 가까운 데이터만을 만들어낸다. <br>
이렇게 되면 같은 label값의 다양한 데이터는 만들어낼 수 없지만 그래도 최소한의 생성 성능은 보장해준다. <br>
학습이 완료된 Autoencoder는 encoder 부분만 따로 떼서 사용한다고 한다. <br>
잘 학습된 autoencoder의 encoder부분은 앞서 말했듯이 데이터의 manifold를 잘 찾아내게 되어 데이터의 feature 추출 시 사용된다. <br><br><br>


# Variational AutoEncoder (VAE)
---
VAE는 generative model이다. Generative model은 단순히 원래의 데이터를 복원하는 것이 아닌, 새로운 데이터를 생성해야 한다. 이 때 필요한 것이 원래 데이터셋의 확률 분포 $p(x)$이다. 하지만 이는 주어지지 않기에 generative model은 데이터의 확률분포를 근사함으로 새로운 데이터를 생성해낸다. <br>
근사한 확률분포에서 목표로 하는 데이터를 생성하기 위해 latent variable $z$가 사용된다. 이 $z$를 샘플링하는 확률분포를 prior distribution $p(z)$라 하는데, 이는 VAE에서 normal distribution이나 uniform distribution으로 가정한다고 한다. <br>
그리고 VAE의 encoder는 데이터 x가 주어졌을 때 latent variable z의 확률분포인 posterior distribution $p_{\theta}(z|x)$를 잘 근사하는 variational distribution $q_{\phi}(z|x)$를 찾는 **Variational Inference**를 수행한다. 이 때 $q_{\phi}(z|x)$는 gaussian distribution으로 가정한다.<br>
이는 두 확률 분포 $p_{\theta}(z|x)$와 $q_{\phi}(z|x)$ 사이의 차이(KL Divergence)를 최소화하는 것으로 보면된다.<br>
또한 VAE는 $p_{\theta}(z|x)$와 $q_{\phi}(z|x)$ 사이의 차이를 최소화하는 동시에 likelihood $p(x|z)$를 즉, $p(x|g_{\theta}(z))$를 최대화하는 것을 목적으로 한다. <br>
Likelihood를 최대화하는 것은 Evidence인 $p(x)$를 최대화하는 것과 마찬가지로 해석된다. <br>

이들을 확률분포 $p(x), \; p_{\theta}(z|x), \; q_{\phi}(z|x)$의 관계식으로 유도하면 다음과 같다.<br>
$$ \log (p(x)) = \int \log(p(x))q_{\phi}(z \mid x)\, dz \quad \leftarrow \int q_{\phi}(z \mid x)\, dz = 1 \\
= \int \log \left( {p(x,z) \over p(z \mid x)} \right)q_{\phi}(z \mid x)\, dz \quad \leftarrow p(x) = {p(x,z) \over p(z \mid x)}  \\
= \int \log \left( {p(x,z) \over q_{\phi}(z \mid x)} \cdot {q_{\phi} \over p(z \mid x)}\right)q_{\phi}(z \mid x)\, dz \\
\int \log \left( {p(x,z) \over q_{\phi}(z\mid x)} \right)q_{\phi}(z \mid x)\, dz + \int \log \left({ q_{\phi}(z\mid x) \over p(z \mid x)} \right)q_{\phi}(z\mid x)\, dz \\
ELBO(\phi) : \int \log \left( {p(x,z) \over q_{\phi}(z\mid x)} \right)q_{\phi}(z \mid x)\, dz \\
KL\left(q_{\phi}(z\mid x) \parallel p(z \mid x) \right) = \int \log \left({ q_{\phi}(z\mid x) \over p(z \mid x)} \right)q_{\phi}(z\mid x)\, dz \quad \text{두 확률분포 간의 거리} \ge 0 $$

<br>

앞서 말했듯이 KL Divergence는 두 확률 분포 사이의 거리를 측정하는 함수로 $KL\ge0$이다. <br>
KL 항을 왼쪽으로 넘겨주면 $\log(p(x)) - KL = ELBO$ 가 되고, 여기서 $\log(p(x))$는 고정된 값이라 KL 값을 줄일수록, <br>
그러니까 확률 분포$q(z\mid x)$와 $p(z\mid x)$의 거리 가깝게 하면 할수록 ELBO 값이 커지게 된다. <br>
즉 최적화된 샘플링 함수를 찾는 것은 KL을 minimize 하는 것이며 이는 동시에 ELBO를 최대화 하는 것이 된다.<br>
그러니까 두가지 optimization을 한번에 푸는 꼴이라고 할 수 있다. <br>
이를 간단한 꼴로 표현해보면

$$ \arg \min_{\phi, \theta} \sum_i -\mathbb{E}_{q_{\phi}(z\mid x_i)}[\log (p(x_i\mid g_{\theta}(z)))]+KL(q_{\theta}(z\mid x_i)\parallel p(z)) $$

가 된다.<br><br>


### Problem
Reconstruction Error 구할 때는 Monte-Carlo Sampling 써서 mean값을 구한다. <br>
하지만 Random sampling은 backpropagation이 불가능하다. <br>
그래서 VAE는 re-parameterization trick이라는 것을 이용하는데 간단하게 말해서 Gaussian distribution을 normal distribution으로 <br>
표현하는 것이다. 

$$ z^{i,j} \sim N(\mu_i,\sigma_i^2I) \rightarrow z^{i,j} = \mu_i + \sigma_i^2 \odot \epsilon \\
\epsilon \sim N(0,I) $$

위 식과 같이 Gaussian distribution을 Normal distribution으로 표현하게 되면 $\mu_i$와 $\sigma_i^2$는 random node $\epsilon$과는 상관없게 되어 backpropagation을 사용할 수 있게 된다.
<br><br><br>



### 끝
진짜 이렇게까지 어려웠던 모델은 처음이였던 것 같다. Transformer는 VAE에 비하면 아주 양반이다... 아직 완전하게 이해하지는 <br>
못한 것 같아 주말을 이용해서 한번 더 이해하려고 해봐야겠다. <br>
그래도 결론적으로 AE와 VAE에 대한 이해는 완료한 것같다. <br>
AE는 manifold 그 자체를 학습하기 위한 것이고 VAE는 생성을 위한 모델이기에 manifold의 결과를 보면 AE는 학습 시에 수시로 manifold의 분포는 유지한채 형태가 변한다. 그에 반해 VAE는 생성이 목적이기에 항상 일관된 normal distribution에 가까운 분포를 유지하고 있었다. 아마 생성되는 데이터의 일관성을 유지하기 위해서가 아닐까 생각한다. <br><br>

이번주는 꽤나 아쉬움이 많은 주간이였다. 초반에 강의를 들을 때는 뭐 MLP, CNN, RNN 다 아는건데 괜찮지 그리고 이정도 강의면 <br>
뒷 내용도 별거없겠지 생각하고 쓸데없이 거만했었다. 절대 그랬으면 안되는 것인데...<br>
또한 Optimization 부분도 분명히 잘 알고 넘어가야되는 것을 아직도 복습하지 않았다. <br>
이번주는 주말이 없을 예정이다...
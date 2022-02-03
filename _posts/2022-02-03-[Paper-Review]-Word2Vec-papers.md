---
layout: post
title: "[Paper Review] Word2Vec papers"
categories: [Paper]
---

---

이번 주에 boostcamp 팀 논문 스터디에서 결정한 논문은 Word2Vec 논문이였다. Skip-gram와 CBOW를 소개하는 논문과 Hierarchical Softmax, Negative sampling과 Subsampling 기법을 소개하는 논문으로 총 두 편으로 구성되어있는데, 따로 포스팅하는 것보다는 한번에 둘 다 하는게 나을 것 같아서 한번에 올리려고 한다. 논문은 다음과 같다. <br>
[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) <br>
[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf)<br><br>

# Introduction
---
Word2Vec 논문은 이름그대로 단어를 벡터로 표현하기위해 만들어진 논문이다. Word2Vec 이전의 단어 표현 방식은 대부분 각 단어를 vocabulary의 index나 one-hot encoding 처럼 개별적인 하나의 단위로 나타냈다. 하지만 이러한 방식에는 문제점이 있었는데, 이런 방식으로 단어들을 표현하게 된다면 단어와 단어 사이의 관계에 대해 알 수 없게 된다. 나라-수도, 형용사의 비교급, 최상급과 같이 단어사이에는 특정한 관계성이 존재하는데, 이를 무시한 word representation은 퀄리티가 상당히 떨어지게 된다. <br>
저자는 당시 논문이 발표되기 전까지의 단어 표현은 아무리 학습데이터를 늘린다해도 성능의 한계가 있다 생각하고 Word2Vec 논문을 발표하게 된다. <br><br>

논문의 핵심 내용은 다음과 같다. <br>
> + 단어를 Distributed Representation으로 표현
> + 비슷한 단어는 가까운 거리(cosine distance)에 표현
> + 한 단어는 여러 단어와 유사성 가질 수 있음 (Multiple degrees of similarity)
> + Word vector간의 연산으로 다른 vector 표현 가능

<br>

### 개선된 점
Word2Vec은 이전의 단어 표현에 비해 training time과 accuracy에 큰 발전을 가져왔다. Distributed representation을 선택해 기존의 sparse vector 방식의 단어 표현보다 훨씬 적은 차원의 dense vector를 사용함으로 계산 복잡도가 크게 감소했으며, word vector들도 각 단어 표현과 함께 관계성 표현도 눈에 띄게 향상되었다. <br><br><br>


# Model Architecture
---
### Computational Complexity
Word2Vec 모델은 기존의 neural network model들에 비해 훨씬 단순한 구조를 띈다. 예시로 NPLM의 경우 input > projection > hidden > output 순서의 layer들을 갖는 모델이다. <br>
NPLM의 훈련시 계산 복잡도를 구하면,

$$ N \times D + N \times D \times H + H \times V $$

로 나타낼 수 있다. 여기서 대부분의 계산 시간을 차지하는 항은 $N \times D \times H$로 hidden layer의 계산에 해당한다. <br>
그리고 RNN base LM은 계산 복잡도를 

$$ H \times H + H \times V $$

로 나타낼 수 있는데 여기서 대부분의 시간은 $H \times H$ 항에서 나온다($H$ : number of hidden layers). <br><br>

이에 반해 논문의 CBOW, Skip-gram 모델은 각각 다음의 복잡도를 띈다.

$$ \text{CBOW} : N \times D + D \times \log_2 (V) \\
\text{Skip-gram} : C \times \left(D + D \times log_2 (V)\right) $$

Word2Vec 모델은 NPLM 모델에서 projection layer를 없애 더 단순화시킨 모델이다. 그로 인해 계산 시간 단축을 끌어낼 수 있었다. <br><br>


## Word2Vec
Word2Vec 논문에서는 CBOW와 Skip-gram 두 가지 모델을 소개한다. 우선 **CBOW**란 Continuous bag-of-words의 약자이다. 학습 방식은 중심단어를 기준으로 이전 N개 단어와 이후 N개 단어를 이용해 중심 단어가 무엇인지를 예측하며 학습한다. <br>

<p align="center">
  <img src="/assets/img/paper/word2vec/CBOW_img.PNG">
</p>

그리고 **Skip-gram**은 CBOW와는 반대로 중심 단어가 주어지고, 해당 중심단어 이전, 이후의 각 N개 이하의 단어들을 예측하며 학습한다. <br>

<p align="center">
  <img src="/assets/img/paper/word2vec/Skipgram_img.PNG">
</p>

저자에 의하면 CBOW는 이전, 이후 각 4개의 단어가 주어졌을 때 가장 성능이 좋았고, Skip-gram은 이전, 이후 각 5개 이하의 랜덤한 단어를 예측할 때 가장 성능이 좋았다고 한다. <br><br>

논문에는 명확하게 모델의 구조를 나타내주지는 않았다. 그래서 Word2Vec 구현 코드가 있을까해서 찾아본 결과 코드를 정리해둔 github를 발견했다.<br>
[해당 github](https://github.com/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec-Skipgram(Softmax).py)<br>
모델의 코드를 보면 알 수 있듯이 단순히 one-hot으로 나타낸 단어를 word vector size로 projection후 다시 각 단어별 값으로 나타내 준다. 상당히 단순한 구조로 좋은 성능을 이끌어내는 것이 꽤 놀라웠다.

<p align="center">
  <img src="/assets/img/paper/word2vec/word2vec_comparison_table.PNG">
  <br>< Comparison on the Semantic-Syntactic Word Relationship Test >
</p>

실험 결과를 보면 전반적으로 Skip-gram의 성능이 가장 좋음을 알 수 있다. 그래서 저자는 이후 성능 개선을 위해 몇가지 기법들을 추가할 때 Skip-gram 모델을 이용해서 실험을 진행했다.<br><br>

###  Extensions
저자는 두번째 논문에서 Word2Vec 모델의 성능을 끌어올리고 훈련 시간을 줄이기 위한 몇가지 기법들을 소개했다. (Hierarchical Softmax, Negative Sampling, Subsampling of Frequent Words)<br>
또한 "Air Canada"와 같은 관용구에 대한 학습이 가능하도록 추가적인 학습 방법도 소개했다. <br>

#### Hierarchical Softmax
우선 Hierarchical Softmax에 대해 이야기하기 전에 Skip-gram 모델부터 다시 짚고 넘어가자. Skip-gram은 간단하게 한 단어를 중심으로 주변의 단어들을 유추하며 학습한다. Skip-gram의 Objective function은 다음 식의 결과를 최대화 하는 것이다. <br>

$$ {1 \over T}\sum_{t=1}^T \sum_{-c \le j \le c, \; j \neq0} \log p(w_{t+j} \mid w_t) $$

식은 각 t번째 단어를 중심으로 앞뒤 각 c개의 단어들이 나타날 확률값들의 평균을 나타낸다. Basic Skip-gram은 위 식의 $p(w_{t+j} \mid w_t)$를 softmax function으로 나타내는데,

$$ p(w_O \mid w_I) = {\exp \left( {v'_{w_O}}^T v_{w_I} \right) \over \sum_{w=1}^W \exp \left( {v'_w}^T v_{w_I} \right) } $$

여기서 $v_w$와 $v'_w$는 각각 $w$의 input과 output word vector를 나타내고 $W$는 vocabulary의 단어 수를 나타낸다. 위 식대로 계산하면 계산 비용이 $W$(보통 $10^5$~$10^7$)에 비례해서 증가하기 때문에 상당히 실용적이지 못하게 된다. <br>
이러한 계산 비용 문제 해결을 위해 저자는 **Hierarchical Softmax**를 대신 사용하기로 했다. <br>
Hierarchical Softmax는 기존의 Softmax 연산은 확률 분포를 얻기위해 $W$개의 node 전체를 계산해야 하는데에 비해 $\log_2 (W)$개의 node만 계산하면 된다는 큰 장점이 있다. Hierarchical Softmax는 이진 트리 구조로 출력 계층의 단어 $W$개를 나타내고, 트리의 leaf node(child 없는 node)는 각각의 단어를 가리킨다. <br>
Hierarchical Softmax는 $p(w \mid w_I)$를 다음과 같이 표현한다. <br>

$$ p(w \mid w_I) = \prod_{j=1}^{L(w)-1} \sigma \left( [[n(w,j+1) = ch(n(w,j))]] \cdot {v'_{n(w,j)}}^T v_{w_I} \right) $$

수식의 각 부분들은 다음을 가리킨다. <br>
$$ n(w,j) : \text{root에서 w까지 가는 path의 j번째 node} \\
n(w,1) : \text{root node} \\
L(w) : \text{root에서 w까지 path의 길이} \\
ch(n) : \text{n의 child, left/right 둘중 하나만 선택} \\ 
[[x]] : \text{x가 참이면 1, 거짓이면 -1} \\
\sigma (x) : {1 \over 1+e^{-x}}
$$

$v_{w_I}$단어 벡터가 입력되면 binary tree에 따라 path상의 각 node들의 weight가 곱해지고 sigmoid를 통해 확률값으로 나타나게 된다. 이러한 방식으로 인해 비교적 적은 양의 계산만으로 $p(w \mid w_I)$값을 얻을 수 있게 된다. <br><br>


#### Negative Sampling
Negative Sampling 기법 또한 마지막 Softmax layer의 계산 비용을 줄이기 위해 고안된 방법이다. Parameter를 업데이트할 때 vocabulary 전체를 업데이트할 필요없이 입력된 단어와 관련된 단어들(context내 단어들)과 관련되지 않은 단어 중 일부(Negative Sample)를 각 그룹으로 만들고 $w_O$가 어느 그룹에 속하는지를 판별하는 이진 분류 문제로 바꾸는 방식으로 진행된다. 이 때, negative sample은 데이터셋의 크기에 따라 5~20(작은 데이터셋)이나 2~5(큰 데이터셋)개를 뽑아 사용한다. Sampling은 단어 등장 확률(Unigram distribution)에 따라 확률적으로 진행되는데 수식은 다음과 같다. <br>

$$ P_n(w) = {U(w_i)^{3/4} \over \sum_{j=0}^n U(w_j)^{3/4}} $$

#### Subsampling of Frequent Words
이 등장 빈도가 높은 단어와 낮은 단어 사이의 학습 불균형을 해결하고자 만들어진 방법이다. 영어에서 in, the, a와 같은 관사들은 매우 높은 빈도로 등장하는데에 반해 몇몇 단어들은 낮은 빈도로 등장하게 된다. 이렇게 되면 많은 수의 단어들이 관사와 관련성이 높게 학습될 가능성이 높아진다. 이러한 경우의 발생을 줄이고자 저자는 학습 과정에서 고빈도 단어의 경우 식에 따라 학습에서 제외하고 학습을 수행했다. <br>

$$ P(w_i) = 1 - \sqrt{t \over f(w_i)} $$

위 식에서 t는 threshold로 t 이상의 frequency($f(w_i)$)로 등장한 단어들은 학습에서 제외한다. <br>
이 방식으로 학습 시간도 줄일 수 있을 뿐만 아니라 희귀 단어의 vector representation도 더 좋게 만들 수 있게 된다. <br><br>

#### Learning Phrases
단어 중에는 "Air Canada"와 같이 전혀 연관성이 없는 단어들이 관용구로 사용되어 특정한 의미를 만들어내는 경우가 더러 있다. 이러한 관용구를 학습하기 위해 저자는 관용구를 찾아 unique token으로 바꿔주며 phrase에 대한 학습을 수행했다. <br>
저자는 bigram별로 score를 주고 해당 bigram의 score가 threshold를 넘으면 구로 취급하며 학습을 진행했다. bigram의 score를 구하는 식은 다음과 같다. <br>

$$ \text{score}(w_i, w_j) = {\text{count}(w_i, w_j) - \delta \over \text{count}(w_i) \times \text{count}(w_j)} $$

위 식에서 $\delta$는 discounting coefficient로 너무 많은 수의 구가 만들어지는 것을 방지하기 위해 추가된 값이다. Training data에서 threshold값을 줄여가며 2~4번 반복적으로 학습을 진행하여 여러 단어로 이루어진 구를 학습하도록 하였다. <br><br><br>


## Results
Word2Vec의 각 Extension에 대한 결과는 다음과 같다. 우선 Negative Sampling과 Hierarchical Softmax의 비교는 Negative Sampling이 좀 더 우세한 것으로 나타났다. <br>

<p align="center">
  <img src="/assets/img/paper/word2vec/NEG_HS_table.PNG">
</p>

NEG-k에서 k는 negative sample의 갯수를 말한다. <br><br>

Phrase 학습의 효과는 다음과 같이 나타났다. <br>

<p align="center">
  <img src="/assets/img/paper/word2vec/phrase_learn_table.PNG">
  <br>< Accuracies on the phrase analogy dataset >
</p>

<br><br><br>


## Conclusion
Word2Vec 모델은 word representation 방식에 큰 변화를 준 논문이라고 생각한다. 간단한 모델 구조로 매우 수준높고 빠르게 단어에 대한 표현을 학습했고, 단어간의 관계도 매우 잘 나타내었다. 예전에 한번 읽었던 논문이긴 하지만 스터디를 진행하면서 다시 읽어보니 꽤 새롭게 다가왔다. (특히 수식 부분이 좀 어렵긴했다...)
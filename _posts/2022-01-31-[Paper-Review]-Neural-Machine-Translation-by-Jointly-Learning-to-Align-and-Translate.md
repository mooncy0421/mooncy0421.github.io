---
layout: post
title: "[Paper Review] Neural Machine Translation by Jointly Learning to Align and Translate"
categories: [Paper]
---

---

boostcamp 같은 팀에서 논문 스터디를 진행하기로 했다. 매주 두편정도 논문을 정해 읽고 발표하는 방식으로 진행하기로 했다.<br>
이번 주 논문은 Attention mechanism을 번역 task에 적용한 *Neural Machine Translation by Jointly Learning to Align and Translate*, Seq2Seq 모델로 잘 알려져있는 *Sequence to Sequence Learning with Neural Networks*이다. 이번 포스팅에는 Attention mechanism 논문을 리뷰하려 한다. <br><br>

## Introduction
이 논문은 이전까지 나왔던 모델의 문제점을 보완하기위해 만들어진 논문이다. 이전 논문들은 대부분이 Encoder-Decoder 구조로 되어있었다. 그 중에서 대표적인 것이 **Seq2Seq** 모델인데, 간단하게 설명하자면 Source sentence를 입력받은 Encoder는 입력을 고정 길이 벡터(context vector)로 변환시키고, Decoder는 이를 다시 Target sentence, 번역 문장으로 만들어내는 과정의 모델이다. 이 논문의 저자는 이러한 Seq2Seq 모델의 문제점을 고정 길이 벡터로 보았고, 이러한 변환 과정이 모델의 성능, 특히 긴 문장에서의 성능을 저하시킨다고 보았다. 그래서 제안한 방법이 매 target 단어 생성 시마다 context vector를 만들어 target 단어를 만들어내는 방식이다. 당시 통계적 번역 방식이 가장 성능이 좋은 시기였는데, attention 방식을 도입한 모델은 거의 통계적 모델에 필적하는 성능을 나타냈다. <br><br>

## Background
기계번역을 확률적 관점으로 나타내면 원래 문장 $\mathbf{x}$가 주어졌을 때 실제 번역 시의 문장인 $\mathbf{y}$가 될 조건부 확률을 최대화하는 것으로 생각할 수 있다.<br>

$$ \arg \max _\mathbf{y} p(\mathbf{y} \mid \mathbf{x}) $$

당시 많은 논문들이 이 조건부 확률분포를 neural network를 통해 바로 학습하고자했고, 그로 인해 제안된 방식이 Encoder-Decoder 방식이다. 이는 Encoder에서 source sentence를 인코딩하고, Decoder에서 디코딩을 통해 target sentence를 만들어내는 방식이였다. 이 중 유명했던 방법으로는 GRU를 이용한 Encoder-Decoder [모델](https://arxiv.org/pdf/1406.1078.pdf)과 LSTM을 이용한 Encoder-Decoder [모델](https://arxiv.org/pdf/1409.3215.pdf)(Seq2Seq)이 있다. 이 두 모델(RNN Encoder-Decoder) 모두 공통점은 입력받은 **가변 길이**의 문장을 **고정 길이** 벡터(fixed-length context vector)로 변환시키고, context vector로 target sentence를 만들어낸다. <br>
RNN Encoder-Decoder 모델은 각 Encoder와 Decoder에 RNN을 이용한다. Encoder는 각 $t$ 시점마다의 hidden state를 이용해 context vector를 만든다. <br>

$$ h_t = f(x_t, h_{t-1}) \\
c = q({h_1, \cdots , h_{T_x}}) \\
f(), \; q() : \mathrm{nonlinear \; functions} $$

Decoder는 이렇게 만들어진 context vector를 이용해 $y_t$를 예측한다. Target sentence $\mathbf{y}$의 번역을 조건부 확률의 곱으로 표현하면 다음과 같다. <br>

$$ p(\mathbf{y}) = \prod_{t=1}^T p \left( y_t \mid \{ y_1, \cdots , t_{t-1} \}, c \right) \; , \\
p \left( y_t \mid \{ y_1, \cdots , t_{t-1} \}, c \right) = g(y_{t-1},s_t,c) \\
g \; : \; \mathrm{nonlinear \; function}
$$

위에서 나타낸 것 처럼 이전까지의 RNN Encoder-Decoder 형태 모델들은 fixed-length context vector 하나를 이용해서 전체 문장의 단어들을 예측한다. 이는 문장 전체 단어들의 정보를 벡터 하나에 전부 녹여내야하기 때문에 문장이 길어질수록 정보가 소실되거나 제대로 담기지않을 가능성이 높아지게 된다. <br><br>

## Attention Mechanism
저자는 고정 길이 벡터의 사용이 성능의 한계를 가져온다고 생각했고, 이를 해결할 방안으로 **Attention Mechanism**을 지목했다. <br>
Attention mechanism은 target sentence의 각 단어들은 source sentence의 특정부분들과 개별적으로 더 깊은 연관을 가질 것이라는 생각에서 출발했다. 그래서 source sentence의 각 부분의 정보에 가중치를 두어 좀 더 집중해서 봐야할 부분과 비교적 덜 자세히 봐도되는 부분을 나누고 그에 따라 번역을 진행한다. <br><br>

모델의 전반적인 구조를 살펴보면 다음과 같다. <br>
### Decoder

<p align="center">
  <img src="/assets/img/paper/attention/decoder_archtecture_img.PNG" height="300" width="230"> <br>
  < Decoder Architecture >
</p>

위의 그림은 논문의 모델인 **RNNsearch**의 decoder부분 연산을 나타낸 것이다. 이를 수식으로 나타내면 아래와 같다. <br><br>
$$ p(y_i \mid y_1, \cdots, y_{i-1}, \mathbf{x}) = g(y_{i-1}, s_i, \mathbf{c}_i) \\
s_i = f(s_{i-1}, y_{i-1}, c_i) \\
c_i = \sum_{j=1}^{T_x} \alpha_{ij}h_j $$

여기서 $ s_i $는 decoder의 i번째 시점 hidden state를, $ \mathbf{c}_i $는 i번째 시점 context vector를 나타낸다. 이 context vector는 encoder의 hidden state sequence로 구해진다. $ h_j $는 encoder j번째 시점의 forward, backward hidden state를 concatenate하여 만들어진 벡터이다(annotation). RNN의 특성상 j번째 단어 주변 단어들의 정보가 많이 담긴 상태의 hidden state가 만들어진다(논문에서는 이를 strong focus라 표현한다). 위 식에서 $ \alpha _{ij} $는 attention weight라는 값인데 이는 encoder j번째 위치의 hidden state와 decoder i번째 단어 사이의 관련 정도를 나타내는 값이다. 이는 alignment model을 이용해 계산한 값들의 비율을 통해 계산되는데 식은 다음과 같다. <br>
$$ \alpha_{ij} = {\exp(e_{ij}) \over \sum_{k=1}^{T_x} \exp(e_{ik})} \\
e_{ij} = a(s_{i-1}, h_j)
$$
<br>
위 식에서 $a$는 alignment model을 나타낸다. 이는 encoder j번째 시점 hidden state와 decoder의 i-1번째까지의 hidden state의 정보를 이용해서 source sentence j번째 단어와 decoder i번째 단어간의 관계를 나타낸 것이다. 구한 값을 exponential으로 퍼센트와 같이 표현해서 source sentence의 각 단어들이 target sentence의 i번째 단어를 번역할 때 얼마만큼의 중요도를 갖는지, 즉 어느정도 focus를 두고 번역을 해야하는지를 나타낸다고 볼 수 있다. <br><br>

이렇게 번역모델이 직접 alignment model을 학습하는 것을 soft-alignment라고 한다. 사람이 직접 source sentence -> target sentence의 정보를 지정한 hard-alignment model에서 벗어난 것이다.<br><br>

말주변이 없어서 못알아듣게 말하고 이해하기가 어려울까봐 계속 같은 말을 반복하는 것 같은데, 다시 한번 요약해서 말하자면,<br>

+ $h_j$ : $j$번째 encoder hidden state, j 주변 정보 강하게 담음.
+ $s_i$ : $i$번째 decoder hidden state
+ $c_i$ : $i$번째 context vector
+ $\alpha_{ij}$ : $i$번째 target sentence 번역 위한 attention weight, $s_{i-1}$과 $h_j$ 이용해서 계산
<br><br>

### Encoder
Encoder는 Bidirectional RNN으로 구성되어 있다. 이는 forward hidden state와 backward hidden state를 이어붙어 각 시점별 hidden state를 생성하게 된다. <br>
이렇게 생성된 encoder의 hidden state는 각 시점별 주변 단어의 정보를 더 많이 담게 된다. <br><br><br>


## Experiments
실험은 영어-프랑스어 데이터셋을 이용해 진행되었다. 비교 대상이 되는 모델은 RNNenc로 이는 GRU를 이용한 RNN Encoder-Decoder 모델이다([논문](https://arxiv.org/pdf/1406.1078.pdf)). <br>
저자는 우선 긴 문장에 대한 성능 평가를 진행했다. <br>

<p align="center">
  <img src="/assets/img/paper/attention/exp_length_img.PNG">
  <br>
</p>

각 모델 이름 뒤의 30과 50은 학습시 사용된 문장들의 최대길이를 나타낸다. -30의 경우 길이가 최대 30인 문장들로만 학습을, -50은 길이가 최대 50인 문장들로만 학습을 진행했다. <br>
결과를 보면 **RNNsearch-50**은 예측할 문장의 길이가 길어지더라도 성능의 변화가 거의 없는데에 반해 **RNNenc** 모델들은 문장 길이가 30이 넘어가기 시작하는 부근부터 성능이 급격히 나빠지는 것을 볼 수 있다. <br><br>

<p align="center">
  <img src="/assets/img/paper/attention/compare_Moses_img.PNG" height="150" width="300"> 
  <br> < BLEU Score compare with SMT >
</p>

위의 표는 당시 통계기반 번역 모델인 Moses의 BLEU score와 RNNsearch, RNNenc 모델을 비교한 표다. RNNsearch 모델은 당시의 통계기반 모델에 거의 필적한 성능을 나타내었다. 표에서 RNNsearch-50* 모델은 기존 모델에서 더 오랜 시간 학습한 모델인데, 이를 보면 학습을 더 길게할 수록 모델의 성능이 올라감을 알 수 있다. <br><br>

<p align="center">
  <img src="/assets/img/paper/attention/alignment_table_img.PNG">
</p>

위의 그림은 RNNsearch 모델이 학습한 alignment table을 나타낸 것이다. 그림의 각 부분은 $\alpha_{ij}$인 attention weight를 나타내고, 밝을수록 높은 attention weight를 나타낸다.<br> 논문의 저자가 학습을 진행한 데이터셋은 영어-프랑스어 데이터셋으로 두 언어의 어순은 일부 형용사, 명사의 순서를 제외하고는 거의 똑같다고 볼 수 있다. 이를 미루어보아, alignment table은 영어-프랑스어 사이의 번역 단어간 관계를 잘 나타내고 있다고 볼 수 있다. <br><br><br>



## 결론
이 논문은 최근 가장 많이 쓰이고 있는 Transformer 구조의 근원이라고도 할 수 있는 Attention Mechanism의 시작이라고도 볼 수 있다. 이 논문은 이전까지 모델들의 문제였던 고정길이 벡터의 인코딩-디코딩 구조에서 어느정도 벗어날 수 있는 계기를 만들어주었다. 긴 문장에서도 성능을 유지할 수 있도록 하였고, source sentence의 단어와 target sentence의 단어들 사이의 관계를 잘 표현했다. <br>
하지만 RNNsearch 모델또한 여전히 완전한 해결책이 되지는 못한 것 같다. 전체 문장을 하나의 고정 길이 벡터로 변환한다는 문제를 어느정도 해결은 했으나 이 모델 또한 각 시점별로 context vector를 만들 뿐 결국에는 일정한 길이의 벡터로 전체 문장을 변환시킨다는 근본적인 문제는 해결되지 않았다고 생각한다. <br>
Seq2Seq 모델과 마찬가지로 문장이 길어질수록 long-term dependency를 포착하지 못할 것이다. <br>
그래도 나는 이 논문이 Transformer가 나올 수 있도록 해준 논문이라 영향력이 상당하다고 생각한다. 
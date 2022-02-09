---
layout : post
title : '[Paper Review] Attention is All You Need'
categories : paper
---

---

이번주에 피어세션에서 공부할 논문은 Transformer로 잘 알려진 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 논문이다. 예전에 한번 읽었던 적이 있긴하지만 이번에 다시 읽어보니 이전에는 넘겼던 내용도 많이 보이고, 전에는 잘 이해되지 않았던 부분도 잘 이해할 수 있어 복습차원에서 매우 좋은 시간이였다. <br><br>

# Introduction
---
이 논문에서 소개하는 Transformer 모델은 recurrence한 구조나 convolution없이 **Attention mechanism**만으로 구성된 모델이다. 이전까지 대세를 <br>
이루던 기계 번역 모델(sequence transduction)은 Encoder-Decoder 구조를 띄고 있어 sequence내 단어 사이 거리가 멀어지게 되면 dependency 포착이 어려워진다. 이러한 문제를 해결하기 위해 [Attention Mechanism](https://arxiv.org/pdf/1409.0473.pdf)을 도입했다. Attention mechanism은 효과적으로 단어들의 long-term dependency를 포착할 수 있었다. 하지만 Attention mechanism을 도입했다 하더라도 근본적으로 RNN의 구조적 특성상 한 가지 문제점이 더 남아있었다. 단어를 time step별로 입력받아 sequential하게 처리하기 때문에 모델의 train data의 parallelization이 불가능하게 된다. <br>
Transformer 모델은 위의 언급한 이전 번역 모델들의 단점을 보완한, Attention 구조만을 사용한 모델이다. Recurrence한 부분을 제거하고 Multi-Head Attention 구조를 이용하여 효과적으로 데이터를 병렬화하여 처리했고, Attention mechanism을 도입한 논문과 마찬가지로 long-term dependency를 잘 포착해냈다. <br><br>

# Model Architecture
---
Transformer 모델 또한 Encoder-Decoder 기반의 구조를 가지고 있다. <br>

<p align="center">
  <img src="/assets/img/paper/Transformer/enc_dec_img.PNG">
  <br> < Transformer Architecture >
</p>

위는 Transformer 모델의 구조를 그림으로 나타낸 것이다. <br>
Transformer는 앞서 말했듯이 convolution이나 recurrence한 구조없이 attention만으로 구성되어있다. 모델에서는 공통적으로 Multi-Head Attention을 사용하고, Multi-Head Attention은 Scaled-Dot Product Attention 구조를 병렬화시켜 사용한다. <br><br>

### Scaled-Dot Product Attention

<p align="center">
  <img src="/assets/img/paper/Transformer/Attn_img.PNG">
  <br> < Scaled Dot-Product Attention >
</p>

위 그림은 Scaled-Dot Product Attention을 나타낸다. Scaled Dot Product Attention은 Encoder, Decoder의 Multi-Head Attention, Masked Multi-Head Attention에 모두 사용되지만 우선은 Encoder 기준으로 설명하겠다. <br>
여기서 Q, K, V는 각각 Query, Key, Value의 약자로 Q, K는 $d_k$, V는 $d_v$차원의 벡터로 이전 encoder layer의 output이다. 그리고 Q와 K의 dot product를 scaling하고 softmax취한 후 V와 dot product를 진행한 결과를 다음 layer로 넘기게 되고 이를 반복한다.<br>
사실 나는 이렇게 설명하는 것 보다는 word embedding을 바로 받는 가장 아래 layer를 예시로 해서 보는 것이 더 잘 이해됐다. <br><br>

우선 문장$X$ 하나가 입력으로 들어왔다고 생각하자. <br>
이 문장 $X$는 각 Query, Key, Value별 Weight인 $ W^Q, \; W^K, \; W^V $에 곱해져 Q, K, V로 만들어진다. <br>

<p align="center">
  <img src="/assets/img/paper/Transformer/enc_QKV_img.png">
  <br> < Query, Key, Value >
</p>

만들어진 Q와 K는 MatMul 단계에서 곱해져 $QK^T$가 되는데, 이는 문장내 각 단어들이 서로와 얼마나 연관성을 갖는지를 나타낸다. <br>

<p align="center">
  <img src="/assets/img/paper/Transformer/enc_attn_MatMul_img.png">
  <br> < Scaled Dot-Product Attention MatMul >
</p>

<p align="center">
  <img src="/assets/img/paper/Transformer/enc_attn_MatMul_res_img.png">
  <br> < MatMul Result >
</p>

결과로 나온 $QK^T$는 각 위치가 단어간의 연관성을 나타내게 된다. 이는 내적 연산의 특성 덕분에 가능하다. 두 벡터의 내적 연산은 한 벡터를 다른 <br>
벡터에 정사영한 것을 곱하는 것으로 해석이 가능하고, 이 때 두 벡터가 유사할 경우 내적의 결과 값이 커지게 되고, 유사도가 낮을 경우 내적 값이 <br>
작아지게 돼서 내적을 두 단어간의 유사도로 사용할 수 있다. <br>
하지만 내적 연산은 잘못하면 값이 너무 커질 수 있어 모델의 안정성을 위해 $\sqrt{d_k}$로 나누어 각 내적 결과값을 줄여준다. <br><br>

이제 이렇게 계산된 $ {QK^T \over \sqrt{d_k}} $를 Softmax function에 넣어 V에 대한 weight로 만들어준다. 이 weight는 각 단어가 같은 문장 내부의 단어들과 얼마만큼의 <br>
유사함을 가지고 있는지를 나타낸 값을 0~1사이의 값으로 바꾼 것이라고 보면된다.<br>
만들어진 weight를 이제 V와 곱해 Scaled Dot-Product Attention의 출력으로 내보내게 된다. <br>
이렇게 encoder의 Attention처럼 Q, K, V가 같은 input sequence인 attention을 **Self Attention**이라고 부른다.<br>
(Decoder도 self attention layer있으나 masking이 추가됨)<br><br>

### Multi-Head Attention
Transformer 논문 저자는 하나의 Attention을 사용하는 것 보다 Attention function을 병렬화하는 것이 더 좋다고 판단하여 Multi-Head Attention 구조를 만들었다. <br>

<p align="center">
  <img src="/assets/img/paper/Transformer/MultiheadAttn_img.PNG">
  <br>< Multi-Head Attnetion >
</p>

Multi-Head Attention은 Scaled Dot-Product Attention을 h번 병렬화 한 꼴과 같다. <br>
병렬화는 Linear projection layer에서 진행되어 Scaled Dot-Product Attention layer에 들어가는데, Linear layer에서 가중치 $W_i^Q, W_i^K, W_i^V$를 곱하는 과정을 병렬화 시킨다. <br>
실제 구현시에는 Weight matrix를 여러개 만드는 것이 아니라 입력되는 단어 벡터들의 dimension을 h등분하여 사용한다. <br>
만들어진 Attention 결과를 head라 하며 각각을 concatenation하여 Multi-Head Attention의 출력을 만들어낸다. 수식으로 표현하면 다음과 같다. <br>

$$ \text{MultiHead}(Q,K,V)= \text{Concat}(\text{head}_1, \cdots, \text{head}_h)W^O \\
\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

여기까지 사용된 가중치와 parameters의 값을 정리하자면 다음과 같다. <br>

> + $d_k$ : Dimension of Q, K, 64
> + $d_v$ : Dimension of V, 64
> + $d_{model}$ : Dimension of word embedding, 512
> + ${1 \over \sqrt{d_k}}$ : Scaling value, $\sqrt{64}$
> + $h$ : Number of head
> + $W_i^Q, W_i^K$ : $\mathbb{R}^{d_{model} \times d_k}$
> + $W_i^V$ : $\mathbb{R}^{d_{model} \times d_v}$
> + $W^O$ : $\mathbb{R}^{hd_v \times d_{model}}$
> + Q, K, V : Sequence length X $d_{model}$

<br>

**왜 Multi-head로 나눠서 attention을 수행할까?** <br>
내 개인적인 의견으로는 512 dimension의 word vector를 weight matrix 하나로 한번에 다 연산하게 되면 각 feature에 대해 디테일한 학습을 못하게 되지않을까 생각한다. 그래서 64 dimension을 갖도록 나눠 각 64개의 feature마다 좀 더 세세하게 학습할 수 있는 weight matrix를 개별적으로 사용하는 것이 더 효과적이라 사용했다고 본다. <br>

[Wikidocs](https://wikidocs.net/31379)를 보니 아마 이 생각이 맞는듯하다.

<br>

### Layer Normalization
Layer Normalization은 Bath Normalization과 같이 모델의 학습 속도를 높히고 값 분포의 안정화를 목적으로 사용된다. Batch Normalization과 동일한 맥락으로 계산되는데, batch normalization의 경우에는 각 입력 벡터들의 feature들의 평균과 분산을 이용해서 정규화를 시행한다. <br>
하지만 입력의 길이가 동일하지않은 자연어의 경우 batch normalization을 이용하기보단 layer normalization을 적용하는 것이 더 좋다. <br>
Layer Normalization은 feature별 통계 정보가 아닌 각 input별 통계 정보를 사용해서 정규화를 시행하기 때문에 입력의 길이가 다양하더라도 쉽게 <br>
정규화가 가능해진다. <br><br>

#### Residual Connection
그리고 Transformer 모델의 그림을 보면 LayerNorm에 해당하는 sublayer에 Add&Norm이라 적혀있는 것을 볼 수 있는데, 이는 **Residual Connection**을 의미한다. Residual connection은 한 layer의 출력과 이전 layer의 출력을 더해 다음 layer의 입력으로 사용하는 것을 말한다. <br>
Residual connection은 마치 얕은 신경망들의 앙상블처럼 작동하여 학습시 gradient가 망가지는 망가지는 경우를 방지해준다. <br>
[해당 논문](https://arxiv.org/pdf/1702.08591.pdf)<br>
[참고 사이트](https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55)<br><br>

### Position-wise Feed-Forward Network
Scaled Dot-Product Attention과 같이 Transformer의 encoder, decoder 둘 다에 사용된다. <br>
Position-wise Feed-Forward Network는 linear transformation 두개와 ReLU activation function으로 구성되어 있다. <br>

$$ \text{FFN}(x) = \max (0,xW_1+b_1)W_2+b_2 $$

이름이 Position-wise인 이유는 sequence 전체에 적용되는 것이 아닌, 각 개별 단어별로 linear transform이 적용되어 이름을 붙였다. $W_1, W_2$의 dimension을 보면 각각 $d_{model} \times d_{ff}, \; d_{ff} \times d_{model}$로 출력 결과는 입력된 값의 dimension과 동일하게 나온다($d_{ff}=2048$). <br><br>



## Encoder

<p align="center">
  <img src="/assets/img/paper/Transformer/encoder_img.PNG">
  <br> < Transformer Encoder >
</p>

위 그림은 Transformer의 Encoder 구조를 나타낸 그림이다. Encoder는 총 6개로 그림에 나타난 layer를 쌓아올려 만들어져 있다. <br>
각 layer는 2개의 sub-layer인 *Multi-Head Attention, Position-wise Feed-Forward Network*와 각 sub-layer 위에 Layer Normalization을 추가하였다. <br>
각 sublayer를 거쳐 나온 output은 다음 layer로 전달되며, 최상단 encoder layer의 출력은 decoder의 Multi-Head Attention의 Key와 Value로 사용된다. <br>
Encoder Multi-Head Attention의 Query, Key, Value는 모두 이전 layer의 출력이 사용된다. 최하단 encoder layer의 경우에는 word embedding이 Q, K, V로 사용된다. <br><br><br>


## Decoder

<p align="center">
  <img src="/assets/img/paper/Transformer/decoder_img.png">
  <br> < Transformer Decoder >
</p>

Decoder layer도 encoder와 마찬가지로 6개를 쌓아올려 decoder로 사용된다. Encoder와 다르게 decoder는 **Masked Multi-Head Attention**을 추가한다. 기본적인 구조는 Multi-Head Attention을 사용하지만, 중간에 Masking 단계를 추가하여 backward 방향의 정보를 차단한다. Source sentence를 받아 <br>
새로운 target sentence를 출력할 때는 auto regressive한 특성을 갖게 되는데 만약 이 때 문장 내 각 단어의 뒷부분도 함께 attention을 취하게 되면 이는 치팅이 되기 때문에 $-\infty$로 masking하여 방지한다. <br>

<p align="center">
  <img src="/assets/img/paper/Transformer/dec_masking_img.png">
  <br>< Masked Multi-Head Attention >
</p>

그리고 Decoder의 Multi-Head Attention은 이전 decoder layer의 출력을 Query로, encoder의 출력을 Key와 Value로 받아 계산된다. <br><br>

## Word Embedding
Transformer의 word embedding은 이전까지의 sequential한 구조의 모델들과는 약간 다르게 입력된다. Recurrence한 구조가 제거된 Transformer는 sequence의 순서 정보를 표현할 방법이 따로 필요해지게 되었는데, 저자는 **Positional Encoding**을 input embedding에 추가하여 위치정보가 포함된 word embedding을 만들어냈다. <br>
Positional encoding을 추가하는 방법은 단순하게 만들어진 위치 정보를 기존의 input embedding에 더해서 사용해주면 된다. 벡터 덧셈이 가능하기 <br>
위해 positional encoding vector의 dimension은 embedding의 dimension인 $d_model$과 동일하게 설정된다. <br>

$$ PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{model}}) \\
PE_{(pos,2i+i)}=\cos(pos/10000^{2i/d_{model}}) $$

위 식에서 $pos$는 sequence내 단어들의 위치를, $i$는 dimension을 나타낸다. 위와 같은 식을 sinusoidal version의 positional encoding이라 부른다. <br><br><br>


# Why Self-Attention?
그럼 저자는 왜 Self-Attention만을 이용한 Transformer 구조를 만들려 했을까?<br>
저자는 세 가지 이유를 들며 Self-Attention의 장점을 말하는데, <br>
첫째로는 layer당 computational complexity이다. <br>

<p align="center">
  <img src="/assets/img/paper/Transformer/complexity_compare.png">
</p>

위의 표의 $n$은 sequence의 길이, $d$는 word vector의 dimension을 나타낸다. <br>
대부분의 sequence길이는 dimension 크기에 비해 현저히 작은 경우가 많다(길이 512정도 되는 문장이 있긴할까...). 비교하게 되면 $n^2 \cdot d $는 대부분의 경우 $ n \cdot d^2 $에 비해 작으며, Convolutional은 뭐... 말할 필요는 딱히 없을 것 같다. $k$는 kernel 크기를 말하는데 이 값이 0~1의 실수일리는 없으니까. <br><br>

그리고 두 번째 이유로는 병렬화 가능 여부 때문인데, Recurrent layer에서는 각 sequence의 단어들을 순차적으로 받아오기에 sequence 길이만큼의 sequential operation이 요구된다. 하지만 Self-Attention은 전체 문장을 한번에 연산하기에 $O(1)$의 complexity를 가진다. <br><br>

마지막으로 long-term dependency에 관한 부분인데, 문장 내 단어 사이의 dependency 학습에 큰 영향을 주는 요소는 각 단어 사이의 거리라고 할 수 있다. RNN의 경우에도 문제로 꼽혔던 것이 먼 거리에 있는 단어 사이 dependency 학습이 어려웠던 이유가 sequential하게 연산을 진행하면서 이전의 정보들이 갈수록 희미해져간다고 했다. <br>
Table에서 나타낸 Maximum Path Length는 문장 내 첫 단어와 마지막 단어간 dependency를 계산할 때의 연산 횟수를 나타낸 것인데, Self-Attention의 경우 구조상 sequential하게 한 단어씩 넘어가며 연산할 필요가 없어진다. <br><br><br>


# Conclusion
---
Attention Is All You Need 논문을 이제 두번째 읽게 되었는데, 처음 읽었을 때에 비해 attention layer에 대한 이해도가 꽤 높아진 것을 알 수 있었다. <br>
Transformer는 BERT, GPT, XLNET등 아주 많이 사용되는 기본 구조로 자리한 만큼 최근 연구들에 대한 이해하려면 기본적인 Transformer에 대한 깊은 이해가 필요하다고 생각든다. <br>
하지만 position-wise FFN에 많은 parameter가 사용돼서 모델이 무거워질 것으로 생각이 들며, FFN을 대체하거나 parameter수를 줄이는 방향으로 <br>
진행된 연구가 있는지 찾아봐야겠다. 
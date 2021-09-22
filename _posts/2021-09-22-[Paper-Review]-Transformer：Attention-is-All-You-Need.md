---
layout: post
title: "[Paper Review] Transformer：Attention is All You Need"
categories: [Paper]
---

---
이번에 정리할 논문은 **Attention is All You Need**이다. 사실 BERT 리뷰 이전에 이 논문부터 정리하고 가는게 맞았을 건데 블로그를 <br>
시작할 쯤 읽던 논문이 BERT라서 먼저 포스팅했다. <br>
Attention is All You Need 논문은 Transformer 구조를 처음 제안한 논문으로 상당히 유명하며 최근의 NLP 분야에서 핵심으로 쓰이는 <br>
모델 구조이다. <br><br>

## INTRODUCTION
---
Transformer 등장 이전의 NLP task 모델들은 RNN 계열의 구조를 사용하는 것이 대부분이였다. 하지만 RNN 계열 구조를 사용할 시 <br>
문제가 있는데, 첫 번째로 RNN은 구조상 sequential한 입력이 주어져야 하는데 이는 모델의 병렬화를 막는 원인이 되었다. 그로 인해<br>
모델의 계산 비용이 높은 편이다. 두 번째도 RNN의 sequential한 입력이 문제가 되었는데, RNN에 긴 시퀀스가 입력으로 주어졌을 시 <br>
해당 시퀀스의 long-term dependency를 잘 포착해내지 못하였다. RNN의 특성 상 입력 토큰들의 time-step 차이가 크면 서로 간의 <br>
연관성을 포착하기가 쉽지 않게 된다. 이러한 문제 해결을 위해 attention mechanism이 제안되었는데, 어텐션은 long-term <br>
dependency 문제는 효과적으로 해결하였으나 병렬화 불가 문제는 해결하지 못했다. <br>
저자는 RNN 구조가 갖는 문제점들을 해결하기 위해 어텐션 기법을 응용한 Transformer라는 구조를 제안했고, 이는 parallelization,<br>
long-term dependency 둘 모두 효과적으로 해결하였다. Transformer는 간단히 말하자면 recurrence한 구조는 없애고 오직 attention <br>
기법만을 사용하여 global dependency를 잘 포착하고 parallelization을 잘 수행하도록 만들었다.  <br><br>

## MODEL ARCHITECTURE
---
Transformer는 번역용으로 만들어진 모델이다. Transformer 이전의 번역 모델들은 보통 encoder-decoder 구조로 만들어졌고, 여기서 <br>
encoder는 input sequence $$(x_1,...,x_n)$$을 $$z=(z_1,...z_n)$$로 변환 후 decoder는 $$z$$를 output sequence $$(y_1,...y_m)$$으로 번역한다.<br>
Transformer 또한 전반적인 encoder-decoder구조를 여러개의 encoder, decoder를 쌓아 이용했다. 이제 각 encoder, decoder 구조를<br>
살펴보려한다.<br>

<p align="center">
  <img src="/assets/img/paper/Transformer/enc_dec_img.PNG" height="600" width="450">
</p>

<br><br>

### Encoder
Transformer의 encoder는 6개의 layer가 쌓인 형태로 위 그림의 왼쪽과 같다. 각 layer는 sub-layer인 Multi-Head Attention + <br>
Positionwise Feed Forward 로 구성되어 있으며 각각의 sub-layer에는 residual connection이 적용, sub-layer의 각 출력에는 layer<br>
normalization이 적용되었다. 또한 residual connection 적용을 위해 embedding layer와 모든 sub-layer의 출력은 $$d_{model}=512 $$의 <br>
차원을 갖는다.<br>

$$ LayerNorm(x+Sublayer(x)) $$

<br>

### Decoder
Decoder 또한 encoder와 마찬가지로 6개의 layer가 싸힌 형태이다. 하지만 encoder와 한가지 다른 점이 있는데, 각 lyaer에 추가<br>
sub-layer가 들어간다는 것이다. 그림에서 오른쪽 부분을 보게 되면, encoder의 두 sub-layer 이전에 Masked-Multi-Head Attention이<br>
추가되어 있다. 이는 decoder stack의 self-attention이 번역 문장의 뒷부분을 컨닝하는 것을 방지하기 위해 추가된 것이다. <br>
Decoder도 residual connection을 적용시켰으며 각 sub-layer 사이에 layer normalization을 추가하였다.<br><br>

여기까지 Transformer의 encoder-decoder의 대략적인 구조에 대해 살펴보았고, 이제는 조금 더 세부적인 내용에 대해 정리해보려한다.<br><br><br>


### Attention
Attention mechanism은 2016년에 나온 [논문](https://arxiv.org/pdf/1409.0473.pdf)에서 처음 제안된 방법이다. Attention은 원래 RNN 계열 모델에서 보조적인 수단으로 <br>
사용되었으나 Transformer 논문에서는 핵심 구조로 동작한다. 저자는 Attention mechanism을 Scaled Dot-Product Attention이라는 <br>
구조로 사용했다.<br>
저자는 self-attention을 사용한 이유를 세가지로 정리했는데, layer당 총 계산 복잡도, 병렬화할 수 있는 계산량 그리고 마지막은 <br>
신경망 안의 long-term dependency간 경로 길이 문제이다. Long-term dependency는 번역 작업에서 상당히 중요한 부분인데, 이에<br>
가장 큰 영향을 주는 요소는 신경망 내에서 요소간 의존성을 학습할 때 순방향, 역방향 각각 횡단 거리이다. 이 거리는 recurrence한<br>
신경망에서는 각 토큰간의 dependency를 구하려면 토큰 사이의 sequential 연산을 다 해야하는 반면, self-attention은 그 과정이 <br>
필요없다. 그로 인해서 더 수월하게 long-term dependency를 학습할 수 있게된다.<br>
<p align="center">
  <img src="/assets/img/paper/Transformer/Attn_table_img.PNG">
</p>
위의 표는 각 layer 종류별 복잡도, sequential operation 수, 최대 경로 길이를 나타낸 것이다. 표에서 n은 sequence length, d는 <br>
dimension 수, k는 kernel size를, r은 restricted self-attention에서 window size 이다. Restricted self-attention은 매우 긴 sequence가<br>
있을 때 계산 성능을 높이기 위해서 output token 주변 r개의 토큰만 고려하도록 하였다. 일반적인 경우에서 sequence의 길이는 <br>
dimension보다 작기 때문에 self-attention이 recurrent 보다 빠르다. 만약 sequece 길이가 더 길다하더라도 restirct를 쓰면 더 좋은 <br>
성능을 얻어낼 수 있다. <br><br>


#### Scaled Dot-Product Attention
<p align="center">
  <img src="/assets/img/paper/Transformer/Attn_img.PNG">
</p>

$$ Attention(Q,K,V) = softmax({QK^T \over {\sqrt{d_k}}})V $$

Scaled Dot-Product Attention의 입력은 Key(K), Query(Q), Value(V)로 이루어져 있다. 여기서 K와 Q는 같은 dimension $$d_k$$를 가지며 <br>
V는 $$d_v$$의 dimension을 가진다. 계산은 우선 Q와 K의 dot-product를 한 후, $$\sqrt{d_k}$$로 나누어준다. 그 다음에는 softmax를 통해 weight를<br>
계산하게 되는데, 이를 attention weight라고 한다. 이 attention weight는 V에 곱해지며 각 V에 어느정도 가중치를 곱해 계산하는지,<br>
그러니까 각각의 단어 토큰에 어느정도 집중을 해서 번역을 수행하는지를 결정해 준다. 이 계산은 모델에서matrix로 묶어서 동시에 <br>
계산한다. <br><br>

#### Multi-Head Attention
<p align="center">
  <img src="/assets/img/paper/Transformer/MultiheadAttn_img.PNG">
</p>

저자는 하나의 scaled dot-product attention을 $$d_{model}$$ 차원의 Q, K, V에서 수행하는 것 보다 각기 다른 차원의 선형 변환을 이용하여 <br>
h번 병렬 수행하는 것이 낫다고 한다. 그렇게 선형 변환된 각 값들로 attention function을 수행하여 얻은 $$d_v$$차원의 출력을 <br>
concatenation과 linear projection하여 최종 값을 얻어낸다. 이를 설명한 그림이 위의 그림이다.<br>

$$ MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O $$

$$ where\;head_i = Attention(QW_i^Q, KW_i^K, VW_i^V) $$

식을 보면 Q, K, V 각각 대응되는 W, $$W_i^Q, W_i^K, W_i^V$$에 linear projection후 attention, concat을 거쳐 한번더 $$W^O$$로 linear projection한 <br>
값을 Multi-Head Attention의 출력으로 사용한다. 여기서 각 파라미터들의 크기는 다음과 같다.<br>

$$ W_i^Q \in \mathbb{R}^{d_{model}\times d_k}, \; W_i^K \in \mathbb{R}^{d_{model}\times d_k}, \; W_i^V \in \mathbb{R}^{d_{model}\times d_v}, \; W^O \in \mathbb{R}^{hd_v \times d_{model}} $$

$$ h=8 $$

$$ d_k = d_v = d_{model}/h = 64 $$

<br><br>

#### Applications of Attention in Transformer

Transformer에서 attention mechanism은 여러 위치에서 서로 다른 목적으로 사용된다. encoder, decoder, encoder-decoder 총 세 위치<br>
에서 사용되며 각 위치에서 사용되는 방식이 조금씩 다르다. 설명할 순서는 그림에서 encoder의 Multi-Head Attention, decoder의 <br>
Multi-Head Attention 그리고 decoder의 Masked Multi-Head Attetion이다. <br><br>

__ENCODER__<br>
우선 encoder의 attention은 key, query, value 모두 이전 encoder layer의 해당 위치 출력이 된다. Self-attention이라고도 부르며 현재 <br>
layer의 모든 위치에서 이전 encoder layer의 모든 위치를 볼 수 있다.<br><br>

__ENCODER-DECODER__<br>
Encoder와 decoder를 연결하는 attention이다. 여기서 key와 value는 encoder의 출력을 사용하고, query는 decoder 이전 layer에서 <br>
가져온다. Decoder의 모든 위치에서 input sequence의 모든 위치를 볼 수 있게된다. 이는 Seq2Seq 모델의 attention mechanism 방식과<br>
매우 유사하다. <br><br>

__DECODER__<br>
Decoder의 self-attention은 encoder와 마찬가지로 이전 layer의 모든 위치를 볼 수 있게 한다. 하지만 그렇게 된다면 아직 예측하지<br>
않은 뒷 부분의 내용을 컨닝하게 되므로 이를 방지하기 위해 저자는 masking 방식을 채택하여 아직 예측하지않은 뒷부분을 전부 <br>
$$-\infty$$ 로 교체해 컨닝을 막았다.<br><br><br>


### Position-wise Feed-Forward Networks
Encoder와 decoder의 attention sub-layer 뒤에는 FFN layer가 있다. <br>

$$ FFN(x)=max(0,xW_1+b_1)W_2+b_2 $$

식에서 볼 수 있듯이, FFN layer는 2개의 선형변환과 ReLU activation으로 구성되어 있다. parameter는 layer 마다 다르게 존재하며<br>
input, output dimension은 $$d_{model}=512$$, inner-layer dimension은 $$d_{ff}=2048$$이다.<br><br><br>


### Positional Encoding
Transformer는 recurrence한 부분이나 convolution이 없기 때문에 입력되는 시퀀스의 순서 정보를 알 방법이 없다. 그래서 저자는<br>
positional encoding을 이용해서 시퀀스내 각 토큰의 위치 정보를 얻고자했다. Positional encoding은 token embedding에 더해져서<br>
transformer의 입력으로 사용된다. 
<p align="center">
  <img src="/assets/img/paper/Transformer/pos_enc_img.PNG">
</p>
Positional encoding은 token embedding에 더해서 사용하기 때문에 token embedding의 차원 수와 같은 $$d_{model}$$의 차원을 갖는다.<br>
Positional encoding을 만드는 방법에는 absolute position 또는 relative position을 이용하는데 transformer에서는 relative를 <br>
사용했다. 이는 sine, cosine 함수의 서로 다른 frequency로 구현한다.<br>

$$ PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}}) $$

$$ PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}}) $$

위 식에서 pos와 i는 각각 해당 토큰의 position과 i-th dimension을 나타낸다.<br><br><br>

## CONCLUSION
---
Transformer는 NLP model들이 RNN 모델 구조를 벗어날 수 있도록한 매우 중요한 구조이다. 이 논문을 시작으로 수많은 SoTA 모델<br>
들이 Recurrence한 구조를 버리고 self-attention mechanism만을 이용한 Transformer 기반의 구조를 만들었다. 시작은 번역 task<br>
였으나, 현재는 BERT, GPT등 수많은 multi-task model에서 좋은 결과를 나타내고 있다. <br>


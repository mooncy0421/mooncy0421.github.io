---
layout: post
title: "[Paper Review]BART：Denoising Sequence-to-Squence Pre-training for Natural Language Generation, Translation, and Comprehension"
categories: [Project]
---

---

이전의 [포스팅](https://mooncy0421.github.io/project/2021/10/10/%EA%B0%9C%EC%9D%B8-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-%EC%82%AC%EC%A0%84%EA%B3%B5%EB%B6%80-Text-Summarization.html)에서 말했던 [Survey 논문](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9328413)을 읽어보았는데, 논문에서 BART에 대한 언급이 많아서 이번에는 [BART](https://arxiv.org/pdf/1910.13461.pdf)를 읽고 포스팅하려 <br>
한다. 논문을 읽어본 후 나는 BART는 BERT와 GPT를 합쳐서 성능을 개선시키려한 모델이라 생각했다. 논문의 그림을 보면 BERT의 <br>
Bidirectional Encoder와 GPT의 Autoregressive Decoder를 연결시킨 것을 모델의 대략적 구조로 묘사하고 있다. 
<p align="center">
  <img src="/assets/img/paper/BART/architecture_img.PNG">
</p>

저자의 말에 따르면 실험 결과 BART는 text generation task로 fine tuning 할 때 가장 좋은 성능을 나타냈다고 한다. 그 중에<br>
abstractive dialogue, question answering, summarization task에서는 새로운 SoTA를 달성했다고 한다. Papers with Code라는<br>
사이트에서도 현재 summarization task의 성능 순위를 보게 되면 BART 모델들이 높은 순위를 기록하고 있는 것을 알 수 있다.<br>
BART는 text generation tasks뿐만 아니라 다른 여러 task에서도 좋은 성능을 얻어내며 RoBERTa의 성능과 비슷한 결과를 내는 <br>
task들도 존재했다.<br><br>

BART의 학습 과정은 BERT와 비슷하다. 원래의 text에 손상을 가하고, 원래 text로 복구시키며 모델을 학습시킨다. BERT는 손상<br>
기법을 text sequence의 토큰들 중 일부를 [MASK] 토큰으로 바꾸고, 그를 다시 복구하는 방식으로 진행했는데 BART는 이 방법을<br>
그대로 따르기보단 여러가지 방법들을 시도해보고자 하였다. Token Masking, Token Deletion, Text Infilling, Sentence Permutation, <br>
Document Rotation의 방법들을 실험해봤다. <br><br>

## Model Architecture
---

앞서 말했듯 BART는 Bidirectional Transformer Encoder와 Auto-Regressive Transformer Decoder를 합쳐서 구성된 모델이다. Encoder<br>
와 decoder를 연결하여 Sequence-to-Sequence Transformer 모델로 구성되었다. encoder와 decoder는 여러 층으로 쌓아 만들었다.<br>
(base model: 6, large model: 12) BART의 구조는 BERT와 상당히 비슷한 형태를 띄고 있는데 몇 가지 다른 점이 있다. 우선 첫번째로 <br>
BART는 decoder의 각 layer에서 encoder의 마지막 은닉계층과 cross-attention을 수행하고 BERT에서 단어 예측 이전에 사용하는 추가<br>
적인 feed-forward layer를 BART에서는 사용하지 않는다는 차이점이 있다. 
<p align="center">
  <img src="/assets/img/paper/BART/architecture_img.PNG">
</p>
앞서 살펴보았던 BART의 모델 구조인데 우선 pre-training 시에는 encoder에 noise function을 이용해 손상된 document가 입력되고, <br>
autoregressive decoder에서는 원래의 document를 복원하며 학습을 한다. Fine-tuning시에는 손상없는 document를 encoder, decoder <br>
둘 다에 입력하여 decoder의 최종 hidden state를 사용하여 출력을 표현한다.<br><br>

## Pre-training
---

BART의 pre-training은 손상된 문서를 복구하며 진행된다. 손상된 문서를 재구성한 출력값과 원래의 문서간 cross entropy를 계산해<br>
loss를 낮추는 방향으로 학습한다. 저자의 말에 따르면 BART는 특정 noise function만이 아니라 어떤 방법이 됐던 BART에 적용할<br>
수 있다고 하였고, 저자는 5가지 방법을 실험하여 좋은 성능을 이끌어내는 noise function을 찾아내고자 하였다. 
<p align="center">
  <img src="/assets/img/paper/BART/noise_func_img.PNG">
</p>

### Token Masking
BERT의 noising 방식과 같다. 원래의 문장에서 토큰을 random sample하여 [MASK] 토큰으로 대체한다. <br><br>

### Token Deletion
Token masking 방식과 비슷하다. random sample된 토큰들을 제거한다. Token masking과 다르게 없어진 토큰의 위치를 모델이 결정<br>
해야한다.<br><br>

### Text Infilling
Poisson distribution으로 샘플링된 길이의 text span을 하나의 [MASK] 토큰으로 대체한다. SpanBERT로부터 고안된 방법이며 몇개의<br>
토큰이 마스킹된지 예측하게 된다.<br><br>

### Sentence Permutation
문장을 기준으로 하여 문서 내 문장을 임의의 순서로 섞는다.<br><br>

### Document Rotation
임의로 한 토큰을 선택하고, 해당 토큰을 문서의 시작으로 하도록 문서의 순서를 바꾼다(rotate). 이 task에서 모델은 document의 <br>
시작 부분을 알 수 있도록 학습된다.<br><br><br>


## Fine-tuning
---
BART의 fine-tuning은 Sequence/Token Classification, Sequence Generation, Machine Translation task가 진행되었다. <br><br>

### Sequence Classification
Encoder와 decoder에 같은 입력을 주고 decoder의 마자막 token의 마지막 hidden state를 multi-class linear classifier에 넣어 분류<br>
한다. 
<p align="center">
  <img src="/assets/img/paper/BART/classification_img.PNG">
</p>
<br><br>

### Token Classification
Sequence classification과 비슷하다. Encoder, decoder에 같은 입력(complete document)을 주고 decoder의 최종 hidden states를<br>
각 토큰들의 representation으로 사용한다. 이 representation들은 token classification에 사용된다.<br><br>

### Sequence Generation
BART에는 Autoregressive decoder가 있어서 바로 sequence generation task에 적용시킬 수 있다. (ex: Abstractive QA, summarization)<br>
Encoder에 input sequence가 입력되고, decoder에서는 autoregressive하게 output을 생성해낸다. <br><br>

### Machine Translation
Machine translation task의 경우에는 위의 task들과는 달리 모델 구조에 약간의 변화가 있다. 저자의 말에 따르면 pre-trained encoder<br>
의 사용으로 모델의 번역 성능을 높일 수 있다고 한다. 그래서 저자는 BART 모델 전체를 하나의 pre-trained decoder로 두고 추가적인 <br>
bi-directional encoder를 BART의 embedding layer 대신 사용한다. 새로 학습된 encoder는 다른 언어 토큰을 BART가 de-noising할 수 <br>
있는 토큰으로 매핑할 수 있도록 학습된다. <br><br><br>


## Experiments
---

### Pre-training Objectives Comparison & Noise Function Comparison
BART 논문은 pre-training objectives 비교 실험도 진행했다. 이전까지 제안된 대표적인 기법들에 대해 실험을 진행했는데 내용은 <br>
다음과 같다.
+ Language Model (GPT)
+ Permuted Language Model (XLNet)
+ Masked Language Model (BERT)
+ Multitask Masked Language Model (UniLM)
+ Masked Seq-to-Seq (MASS)

Pre-training objective들은 공평하게 비교하기에 약간의 무리가 있다. 대표하는 각 모델마다의 training data, resources 그리고 모델<br>
구조적 차이와 fine-tuning 과정이 다르기 때문이다. 그래서 저자는 각 방법들을 새로 구현하여 비교하고자 하였다. <br>
그리고 저자는 여러가지 noising function들을 BART base 모델에 적용시킨 것도 함께 비교했다. <br><br>

### Tasks
실험에 사용된 task들은 다음과 같다.
+ SQuAD (Extractive Question Answering)
+ MNLI (Bitext Classification)
+ ELI5 (Long-form Abstractive Question Answering)
+ XSum (Abstractive Summarization)
+ ConvAI2 (Dialogue Response Generation)
+ CNN/Daily Mail (News Summarization)
<br><br>

### Results
실험결과는 다음과 같다.

<p align="center">
  <img src="/assets/img/paper/BART/compare_result_img.PNG">
</p>

정리하자면 각 task별 가장 좋은 성능을 이끌어낸 모델은 다음과 같다.
+ SQuAD : BART + Text Infilling
+ MNLI : BERT Base
+ ELI5 : Language Model
+ XSum : BART + Text Infilling
+ ConvAI2 : BART + Text Infilling
+ CNN/Daily Mail : BART + Text Infilling + Sentence Shuffling

<br><br>
이러한 결과로 알 수 있는 점은 <br>

+ Pre-training 방법의 성능은 task에 의해 좌우됨
+ Token masking 방식들은 중요함 (Token masking/deletion, Text infilling)
+ Left-to-right auto-regressive LM은 generation task 성능을 개선함
+ SQuAD task에서 bidirectional encoder는 중요함
+ 성능에 있어 pre-training 방법만이 중요한 요소는 아님
+ ELI5 task에서는 순수 LM이 가장 잘 작동
+ BART는 일관되게 좋은 성능을 보여줌

로 정리할 수 있다. <br><br>

### Large-scale Pre-training Experiments
저자는 또 큰 모델들에 대해서도 BART와 비교를 진행했다. 실험에 사용된 task는 discriminative task를 위해 SQuAD, GLUE, generation <br>
task에 CNN/DailyMail, XSum을 사용했다. 실험 결과는 다음과 같다.<br>

<p align="center">
  <img src="/assets/img/paper/BART/discrim_result_img.PNG">
</p>

<p align="center">
  <img src="/assets/img/paper/BART/gen_result_img.PNG">
</p>

실험 결과를 보면 Discriminative task들 중 많은 수의 task에서 이전 SoTA였던 RoBERTa와 비슷한 성능을 내거나 성능을 능가한 것을 <br>
볼 수 있다. 그리고 Generation task에서는 모든 score에서 BART가 SoTA를 달성한 것을 알 수 있다. 여기서 CNN/DailyMail 데이터셋은 <br>
Extractive model을 쓸 때 좋은 성능을 얻을 수 있고, XSum 데이터셋은 abstractive model을 쓸 때 좋은 성능을 얻어낼 수 있다. 저자는 <br>
뉴스 요약 task만 아니라 conversational response generation task 또한 실험했는데, 이 또한 SoTA를 달성했다.<br>

<p align="center">
  <img src="/assets/img/paper/BART/conv_result_img.PNG">
</p>

그리고 Abstractive Question Answering, Machine Translation task도 SoTA를 달성하였다.<br>
<p align="center">
  < Abstractive QA > <br>
  <img src="/assets/img/paper/BART/QA_result_img.PNG">
</p>

<p align="center">
  < Romanian-English Translation > <br>
  <img src="/assets/img/paper/BART/trans_result_img.PNG">
</p>

<br><br><br>


## Related Work
---
BART와 관련된 모델들과 각 모델의 특징은 다음과 같이 정리할 수 있다.<br><br>

+ GPT : Uni-directional context만 학습
+ ELMo : L-R, R-L 두 개의 uni-directional representation을 concatenate 하여 사용
+ BERT : Masked Language Model (변형 모델 : RoBERTa, ALBERT, SpanBERT)
+ UniLM : Mask ensemble로 fine-tuning한 BERT (논문을 안읽어봐서 잘 모르겠다)
+ MASS : BART와 가장 비슷한 모델
+ XLNet: Permutation Language Model

<br><br>

Transformer 이후의 논문들을 읽기 시작한지 그렇게 오래되지는 않았지만 요즘 나오는 모델들은 대부분 BERT의 변형 버전인 것 <br>
같다.그만큼 BERT가 NLP에 큰 영향을 끼친거 같고, 내 기억에 BERT는 generation task에서는 GPT에 밀리는 모습을 보았던 것 같은데<br>
BART가 이를 해결한 듯 보인다. 저자도 말했듯이 auto-regressive한 구조가 generation task의 성능 개선에 큰 역할을 하기 때문인 것 <br>
같다. 이 다음에는 [MASS](https://arxiv.org/pdf/1905.02450.pdf)를 한번 읽어볼까 한다.
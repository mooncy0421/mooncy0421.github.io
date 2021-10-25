---
layout: post
title: "[Paper Review]MASS：Masked Sequence to Sequence Pre-training for Language Generation"
categories: [Project]
---

---

이번에 포스팅할 논문은 MASS라는 논문이다. 이전에 포스팅했던 BART와 컨셉이 매우 유사한 논문이며 BART보다 약간 먼저 나온 <br>
논문이다. Encoder-decoder 기반 구조의 생성 모델이며 BERT에서 영감을 받아 masking 기법을 적용시켜 masking되지 않은 부분을<br>
이용해 masking된 부분을 재구축하면서 학습한다. 실험은 generation task (translation, summarization, conversational response <br>
gneration)에서 진행되었으며, 당시의 SoTA를 달성했다. 자세한 내용은 논문 내용을 따라가며 한번 살펴보려한다. <br><br>


# INTRODUCTION
---
MASS는 pre-training + fine-tuning 방법 중 하나이다. 이전까지 GPT나 BERT, ELMo등 수많은 pre-training + fine-tuning 방법이 나왔고, <br>
그 중 BERT가 양방향 정보를 학습하는 가장 대표적인 방식으로 알려져 왔다. 하지만 BERT는 NLU tasks에 초점이 맞춰져 있어, NLG <br>
tasks에는 딱히 적합한 방법은 아니였다. 그래서 저자는 BERT의 아이디어를 이용해, NLG tasks에 알맞은 모델을 새로 만들고자 했고, <br>
Seq2Seq 구조와 BERT의 Masking 기법을 합쳐 MASS를 발표했다. MASS의 encoder는 원래 문장의 연속된 토큰 일부를 마스킹한 것을<br>
입력으로 받고, decoder는encoder의 representation에 따라 masking된 부분을 예측한다. MASS는 encoder에서 masking된 부분을 <br>
decoder에서 예측하기 위해 encoder로 하여금 masking되지 않은 부분에 대한 이해를 더 잘하도록 할 수 있고, decoder의 입력을 <br>
encoder 입력과 반대 (encoder에서 masking안된 것을 masking)로 하여 target sentence의 이전 토큰에 대한 표현 의존보다 source <br>
sentence의 표현에 대한 의존을 높게 하여 encoder와 함께 학습하는 것을 쉽게 할 수 있다.<br><br><br>



# MASS
---
앞서 말했듯 MASS는 기존의 sequence to sequence learning 방식이 아닌 masked sequence to sequence pre-training 방식을 사용한다.<br>
원래의 sequence to sequence learning 방식은 문제가 있었다. 대부분의 NLG task들이 필요로 하는 paired data의 양은 부족해 학습 시 <br>
성능 저하의 원인이 되었다. 그에 반해 unpaired data는 많은 양이 있었고, unpaired data를 이용해 pre-training 후 적은 양의 paired <br>
data로 fine-tuning하는 방식이 주목받게 되었다. MASS 또한 이러한 pre-training + fine-tuning 방식을 사용했다. MASS의 objective <br>
function은 log likelihood를 쓰며 다음과 같이 표현된다.

$$  \begin{matrix}
    L(\theta;\mathcal{X}) &=& {1 \over |\mathcal{X}|} \sum\nolimits_{x \in \mathcal{X}} \log P(x^{u:v}|x^{\backslash u:v}; \theta) \\
                          &=& {1 \over |\mathcal{X}|} \sum\nolimits_{x \in \mathcal{X}} \log \prod_{t=u}^v P(x_t^{u:v}|x_{<t}^{\backslash u:v},x^{\backslash u:v}; \theta)
    \end{matrix} $$

여기서 $ x \in \mathcal{X} $ 는 unpaired source sentence를 나타내고, $ x^{\backslash u:v} $ 는 position $ u $ 에서 $ v $ 까지가 masking된 $ x $ 를 의미한다. 그리고 $ x^{u:v} $ 는 <br>
$ u $ 에서 $ v $ 까지의 masking된 부분을 칭한다. MASS의 sequence to sequence model은 masking된 문장 $ x^{\backslash u:v} $ 가 주어졌을 때 masking된 <br>
$ x^{u:v} $ 를 예측하며 학습한다. 이 때 모델은 masking된 부분만 예측하며 decoder는 encoder에서 masking되지 않은 부분을 masking한 <br>
것을 입력으로 받게 된다. 

<p align="center">
  <img src="/assets/img/paper/MASS/MASS_framework_img.PNG">
  < encoder-decoder framework>
</p>

그림에서 "_" 토큰은 마스킹 토큰을 의미한다. 그리고 MASS에는 얼마만큼의 source sentence를 masking할지 정하는 hyperparameter <br>
k가 있는데, 이 k값에 따라 모델의 성능이 차이나게 된다. 그리고 GPT와 BERT는 어떻게 보면 MASS의 케이스 중 하나라고 볼 수도 <br>
있는데, 왜냐하면 BERT의 경우 개별적인 토큰을 masking하는 방식으로 k=1일 경우, GPT의 경우 encoder의 input sentence가 모두 <br>
masking된 standard LM의 형태를 띄고 있어 k=100%(=sentence length)인 경우라고 생각할 수 있게 되기 때문이다. k값에 따른 성능<br>
차이는 이후의 실험 결과에 정리되어 있다. 

<p align="center">
  <img src="/assets/img/paper/MASS/BERT_GPT_table.PNG">
</p>

<p align="center">
  <img src="/assets/img/paper/MASS/BERT_GPT_framework_img.PNG">
  < Left: BERT, Right: GPT >
</p>

MASS의 pre-training 방법은 몇가지 이점이 있는데, 우선 masking된 토큰들만을 예측하여 encoder로 하여금 masking되지않은 토큰<br>
들의 뜻을 더 잘 이해하도록 할 수 있고 decoder가 encoder 측에서 좀 더 유용한 정보를 얻어낼 수 있도록 할 수 있다. 그리고 decoder<br>
에서 연속적인 토큰들을 예측하도록 하여 각각의 토큰들을 예측하는 것 보다 더 좋은 language modeling capability를 얻을 수 있게 <br>
된다. 마지막으로 decoder 입력 토큰들 중 encoder에서 masking되지 않았던 토큰들을 masking하여 입력받아 decoder에서는 이전 <br>
토큰들에서 얻는 것보다 encoder측의 더 유용한 정보들을 뽑아낼 수 있게된다. 이 덕에 MASS는 encoder와 decoder를 generation task<br>
에 잘 맞게 학습시킬 수 있게 된다. <br><br><br>



# EXPERIMENTS
---
앞부분에서 말했듯이 MASS는 NMT(Neural Machine Translation), Text Summarization, Conversational Response Generation 총 세가지<br>
task에 대해 실험을 진행했다. 그리고 masking할 토큰 비율에 대한 hyperparameter k의 적정값, ablation study 또한 진행했다. <br><br>


## NMT
NMT 실험은 bilingual data 없이 monolingual data만으로 fine-tuning을 진행했으며 총 6가지 번역 task에 대한 성능을 평가했다. 그리고<br>
low-resource setting에서도 실험을 진행했는데, 같은 번역 task에서 각각 10K, 100K, 1M bilingual data를 이용해 fine-tuning해 성능을<br>
측정했다.

<p align="center">
  <img src="/assets/img/paper/MASS/NMT_res_img.PNG">
  < BLEU score comparisons on unsupervised NMT >
</p>

<p align="center">
  <img src="/assets/img/paper/MASS/NMT_low_res_img.PNG">
  < BLEU score comparisions on low-resource NMT >
</p>

표에서 볼 수 있듯이 MASS가 이전의 모델들보다 좋은 성능을 보이며 SoTA를 달성했다. <br>
저자는 또한 MASS의 pre-training 방법에 대해서도 비교를 진행했다. 비교 대상은 BERT(enc)+LM(dec), DAE(Denoising auto-encoder)<br>
(enc,dec)를 선택했고, 이 또한 MASS가 가장 뛰어난 성능을 보였다. 

<p align="center">
  <img src="/assets/img/paper/MASS/NMT_pre_res_img.PNG"> <br>
  < BLEU score comparisons between MASS and other pre-training methods >
</p>

<br><br>

## Text Summarization
Text summarization task는 Gigaword corpus를 이용해 fine-tuning했으며 fine-tuning data의 양에 따른 실험 결과와 pre-training 방법에 대한 비교도 진행했다. 

<p align="center">
  <img src="/assets/img/paper/MASS/sum_res_img.PNG"> <br>
  < ROUGE score comparisons on Text Summarization >
</p>

<p align="center">
  <img src="/assets/img/paper/MASS/sum_pre_res_img.PNG"> <br>
  < ROUGE score comparisons between MASS and other pre-training methods >
</p>

표를 보면 알 수 있듯이 MASS가 이전의 방법보다 좋은 결과를 얻어냄을 알 수 있다.
<br><br>

## Conversational Response Generation
Conversational Response Generation task의 실험 결과는 다음과 같다.

<p align="center">
  <img src="/assets/img/paper/MASS/conv_res_img.PNG"> <br>
  < PPL score comparisons between MASS and other pre-training methods >
</p>

표에 나와있듯이 MASS는 실험했던 모든 task에서 이전보다 좋은 성능을 얻어냈다.<br><br>


## Study of different k
MASS의 저자는 masking할 토큰의 비율에 대한 hyperparamter 비교도 진행했다. 비교는 k=1인 경우의 BERT, k=m인 경우인 GPT와 <br>
나머지는 각각 10% 간격으로 측정을 진행했다.

<p align="center">
  <img src="/assets/img/paper/MASS/dif_k_res_img.PNG">
</p>

실험 결과는 순서대로 각각 English pre-trained model(PPL), French pre-trained model(PPL), EN-FR translation(BLEU), Summarization <br>
(ROUGE), Conversational response generation validation set(PPL)이다. 결과를 보면 모든 경우에 대략 50% 부근에서 가장 좋은 성능을 <br>
나타냄을 알 수 있다. 이는 encoder와 decoder의 masked token 비율이 균형을 이루기 때문이라고 저자는 말한다. <br><br>


## Ablation Study
Ablation study는 두 가지 기법에 대해 진행되었다. Encoder에서 연속된 토큰을 masking하는 것 (Discrete), encoder에서 masking되지<br>
않은 토큰들을 decoder에서 masking한 후 학습을 진행하는 것 (Feed)에 대해 실험을 했으며, 실험 결과 두 기법 모두 성능에 영향을 <br>
주는 것으로 나타났다.

<p align="center">
  <img src="/assets/img/paper/MASS/abl_res_img.PNG">
</p>
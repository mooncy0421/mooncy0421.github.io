---
layout: post
title: '[Paper Review]BERTSUM：Text Summarization with Pretrained Encoders'
categories: [Project]
---

---

이번에 포스팅할 논문은 BERTSUM이다. 꽤 오랜만에 포스팅을 하는데 이사도 가고 이것 저것 때문에 되게 오랜만에 논문을 읽었다. <br><br>


# INTRODUCTION
---
BERTSUM은 간단하게 말하자면 BERT를 Summarization task에 적용시키기 위해 BERT구조에 변형을 약간 가한 것이다. BERT 기반의 document-level encoder를 만들고, 이를 이용해서 Extractive Summarization과 Abstractive Summarization 둘 다에 사용 가능한 general framework를 만들었다. Extractive model과 Abstractive model은 약간의 차이가 있는데, 우선 Extractive model은 BERT base encoder의 맨 위에 문장간 Transformer layer를 몇개 쌓아 만들었고, Abstractive model은 일반적인 Encoder-Decoder 구조로 만들었으며, 새로운 fine-tuning 기법을 사용했다. Encoder, Decoder에 서로 다른 optimizer를 사용해 둘 사이의 mismatch를 줄이고자 하였다. <br>
그리고 Abstractive model의 성능을 올리기 위해 two-stage find-tuning 기법을 적용했는데, 우선 Encoder를 Extractive objective로 fine-tuning 후 Abstractive objective로 fine-tuning하여 Abstractive Summarization의 성능을 끌어올리고자 하였다. <br><br>

저자는 또한 이 논문이 세 부분에서 기여를 했다고 하는데, <br>
1. Summarization task에서 document encoding의 중요성을 강조함
2. Summarization task에 효율적으로 Pre-trained LM을 적용하는 방법 제시함
3. BERTSUM model이 이후 summarization의 성능을 개선하는 디딤돌이 될 것임
<br><br><br>


# BACKGROUND
---
## Pretrained Language Models
BERTSUM은 이름에서부터 알 수 있듯이 Pretrained Language Model인 BERT를 사용했다. BERT는 간단히 말하자면 Masked Language Modeling과 Next Sentence Prediction을 이용해서 language representation을 학습하는 언어 모델이며 일반적인 모델 구조는 아래의 그림과 같다. 

<p align="center">
  <img src="/assets/img/paper/BERTSUM/Original_BERT_img.PNG"> <br>
  < Architecture of Original BERT model >
</p>

BERT의 입력은 원래 text에 두 가지의 토큰이 추가되는데, 우선 text 맨 앞에 [CLS] 토큰이 추가되어 sequence 전체의 정보를 담는다. (분류 task에 사용됨) 그리고 각 문장의 뒤에 [SEP] 토큰을 추가해 문장의 경계를 나타낸다. <br>
또한 sequence를 토큰으로 표현할 때 세 종류의 임베딩이 할당되는데 token embedding, segmentation embedding, position embedding이 있다. token embedding은 각 token의 의미를, segmentation embedding은 속하는 문장의 구분을, position embeeding은 문장 내 각 토큰의 위치를 나타낸다. 세 임베딩은 모두 더해져 BERT layer에 입력되어 사용된다. <br>

$$ \tilde{h}^l = \mathrm{LN}(h^{l-1} + \mathrm{MHAtt}(h^{l-1})) $$

$$ h^l = \mathrm{LN}(\tilde{h}^l + \mathrm{FFN}(\tilde{h}^l)) $$

<br>
위 식에서 $ h^0 $ 은 입력 벡터를 가리키고, LN은 Layer Normalization, MHAtt는 Multi-head attention을, $ l $은 몇번째 layer인지를 가리킨다. 
맨 위 layer에서 BERT는 각 토큰에 대한 출력 벡터 $t_i$를 생성한다. <br><br>

## Extractive Summarization
Extractive summarization(추출 요약)은 문서 내에서 가장 중요한 문장들을 찾아내거나 이어붙여 요약문을 만들어낸다. 신경망 모델에서는 이를 sentence classification task로 취급한다. 왜냐면 모델에 문서를 입력하여 각 문장의 representation에 따라 문장의 중요도를 판별하여 중요한 문장을 선택하기 때문이다. <br><br>

## Abstractive Summarization
Abstractive summarization(생성 요약)은 신경망 모델에서 Sequence-to-Sequence 문제로 취급된다. encoder는 source document를 encoding하여 연속적 representation으로 변환하고, 이를 decoder의 입력으로 사용한다. Decoder는 입력받은 representation으로 auto-regressive하게 target summary를 생성한다. <br><br><br>


# Fine-tuning BERT for Summarization
---
BERT는 여러 NLP task들에 fine-tuning되어 사용되긴 하지만, summarization에 바로 적용하기에는 적합하지 않다. 왜냐하면 BERT는 Masked LM으로 학습되기 때문에 출력 벡터가 토큰 기반이 되는 반면 추출 요약은 sentence-level represenation에 기반하여 작동되기에 바로 적용하기보다는 변경이 필요하다. 또한 BERT의 segmentation embedding은 sentence-pair에만 사용되기에 multi-sentential input을 사용하는 summarization에 적합하지 않다. <br>
이러한 문제점들을 해결하기위해 저자는 BERTSUM이라는 새로운 구조를 제안했고, 그 구조는 아래의 오른쪽 그림과 같다. <br>

<p align="center">
  <img src="/assets/img/paper/BERTSUM/BERTSUM_img.PNG"> <br>
  < Left: Architecture of Original BERT / Right: Architecture of BERTSUM >
</p>

위에서 말한 문제들을 해결하고자 저자는 각 sentence별 representation을 얻기위해 각 문장의 앞에 [CLS] 토큰을 삽입하여 앞 문장의 feature를 수집했다. 또한 여러 문장을 입력으로 받는 summarization task이기에 interval segment embedding으로 홀수번째, 짝수번째 문장에 각각 $ E_A, E_B $segment embedding을 할당해 사용했다. 또한 original BERT에서 최대 길이 512의 position embedding의 한계를 극복하고자 최대 position embedding을 늘려서 사용했다. <br><br>

## Extractive Summarization
Extractive summarization은 문서 내의 각 문장에 label 0,1을 할당하는 작업으로 정의된다. 이 label은 해당 문장이 요약문에 속하는지 아닌지를 판별하는 값이다. BERTSUM에서는 최상단 layer i번째 [CLS] 토큰의 벡터를 i번째 문장에 대한 representation으로 사용한다. 그리고 그 위에 여러층의 문장간 Transformer layer를 쌓아 추출 요약문 생성을 위한 document-level feature를 얻어낸다. 

$$ \tilde{h}^l = \mathrm{LN}(h^{l-1} + \mathrm{MHAtt}(h^{l-1})) $$

$$ h^l = \mathrm{LN}(\tilde{h}^l + \mathrm{FFN}(\tilde{h}^l)) $$

여기서 $h^0 = \mathrm{PosEmb}(T)$를 나타내고, $T$는 BERTSUM의 출력인 sentence vector를 가리킨다. 그리고 PosEmb는 $T$애 Transformer에 사용된 것과 마찬가지로 positional embedding을 더해준다. 모델의 최종 출력단에는 sigmoid classifier가 적용된다. 

$$ \hat{y}_i = \sigma(W_oh_i^L + b_o) $$

여기서 $h_i^L$은 최상단 Transformer layer에서의 i번째 문장 벡터를 나타낸다. 저자는 Transformer layer 수를 1, 2, 3으로 실험했는데 2개의 layer를 쌓았을 때 가장 성능이 좋다고 한다. <br>
이렇게 BERTSUM을 이용해 만들어진 Extractive Summarization model을 __BERTSUMEXT__ 라고 부른다. <br>

### Hyperparameters
+ __Loss function__ : Binary Classification Entropy
+ __Optimizer__ : Adam ($\beta_1=0.9, \beta_2=0.999$)
+ __Learning rate schedule__ : $lr = 2e^{-3}\cdot \mathrm{min}(\mathrm{step}^{-0.5},\mathrm{step}\cdot \mathrm{warmup}^{-1.5}) $
+ __warmup__ : 10,000

<br><br>

## Abstractive Summarization
Abstractive summarization에서는 standard encoder-decoder 구조를 사용한다. Encoder는 pretrained BERTSUM을, decoder는 random initialize된 6-layered Transformer를 사용한다. 하지만 이렇게 사용할 경우 encoder는 pretrain된 반면 decoder는 처음부터 학습해야하기 때문에 둘 사이에서 mismatch가 생길 수 있다. Fine-tuning시 둘 중 하나는 overfit될 때 다른 하나가 underfit되게 만들어 불안정하게 만들 수 있다. 이 문제를 해결하고자 저자는 encoder, decoder의 optimizer를 구분하는 새로운 fine-tuning 기법을 제안했다.<br>

$$ lr_\varepsilon = \tilde{lr}_\varepsilon \cdot \mathrm{min}(step^{-0.5}, \mathrm{step} \cdot \mathrm{warmup}_\varepsilon^{-1.5}) $$

$$ lr_\mathcal{D} = \tilde{lr}_\mathcal{D} \cdot \mathrm{min}(step^{-0.5}, \mathrm{step} \cdot \mathrm{warmup}_\mathcal{D}^{-1.5}) $$

위의 수식 중 첫번째는 encoder, 두번째는 decoder를 나타낸다. <br>
각각의 hyperparameter 값은 다음과 같다.

__Encoder__
<br> $$ \tilde{lr}_\varepsilon = 2e^{-3} $$ <br>
$$ \mathrm{warmup}_\varepsilon = 20,000 $$ <br>

__Decoder__
<br> $$ \tilde{lr}_\mathcal{D} = 0.1 $$ <br>
$$ \mathrm{warmup}_\mathcal{D} = 10,000 $$ <br>

이러한 hyperparameter 설정은 pretrain된 encoder를 작은 learning rate와 더 부드러운 decay로 학습을해야 decoder가 안정화되는 <br>동안 더 정확한 gradient로 학습할 수 있을 것이라는 가정에서 비롯되었다. <br><br>

그리고 저자는 2단계에 걸친 fine-tuning 기법을 제안했는데, 우선 encoder를 Extractive summarization task에 대해 fine-tune 후 Abstractive summarization task로 한 번 더 fine-tuning하는 방식이다. 이렇게 하면 extractive objective가 abstractive summarization의 성능을 더 좋게 할 수 있다고 한다. Architecture의 큰 변화없이 두 task에서 공유되는 정보를 이용할 수 있다는 점에서 이득을 볼 수 <br>있게 된다. <br>
여기서 기본 abstractive model을 __BERTSUMABS__ 라 하고, two-stage fine-tuning model을 __BERTSUMEXTABS__ 라 부른다.<br><br><br>


# EXPERIMENTS
---
저자는 BERTSUM 모델들을 여러가지 데이터셋에 대해서 실험했다. <br>

<p align="center">
  <img src="/assets/img/paper/BERTSUM/Dataset_img.PNG"> <br>
  < Summarization Datasets detail >
</p>
<br>

+ CNN/DailyMail : 뉴스 기사 데이터셋이다. 뉴스 기사 및 간략한 개요를 제공하는 하이라이트가 포함되어 있다.
+ NYT : 뉴스 기사와 abstractive summary를 제공하는 데이터셋이다.
+ XSum : 뉴스 기사와 무엇에 관한 기사인지에 대한 한 문장 대답을 제공하는 데이터셋이다. CNN/DailyMail과 NYT는 extractive에 <br>
가까운 반면 XSum은 매우 abstractive한 데이터셋이다.
<br><br>

## Results
저자는 우선 ROUGE score (ROUGE-1, ROUGE-2, ROUGE-L)로 성능을 평가했다. <br>

<p align="center">
  <img src="/assets/img/paper/BERTSUM/ROUGE_res_img.PNG" width="600" heigth="800"><br>
  < ROUGE score results on CNN/DailyMail >
</p>

위의 표는 CNN/DailyMail 데이터셋에 대한 성능의 ROUGE score 측정 결과이다. 첫번째 블록의 ORACLE은 extractive ORACLE system의 결과로 상한선을 나타낸다. 그리고 그 밑의 LEAD-3은 단순히 문서 첫 세 문장을 선택한 것을 가리키며 baseline이 된다. <br>
두 번째 블록은 CNN/DailyMail 데이터셋으로 훈련한 여러종류의 추출 요약 모델들의 ROUGE score를 나타내고, 세 번째는 생성 요약 모델들의 score를 나타낸다. 또한 두, 세 번째 블록 맨 아래 항목을 보면 TransformerEXT와 TransformerABS가 있는데, 이는 BERTSUM 모델과 같은 architecture를 쓰는 pretrain되지 않은 Transformer baseline을 말한다. <br>
가장 아래의 블록을 보면 BERT-based 모델들이 있는데, 여기에는 BERTSUMEXT와 그 변형들, BERTSUMABS, BERTSUMEXTABS가 있다. <br>
결과를 보면, BERTSUMEXT, 특히 large 모델이 가장 좋은 성능을 기록했다. 이는 딱히 놀라울만한 발견은 아니다. 우선 CNN/DailyMail 데이터셋은 원래 Extractive 모델들의 성능이 좋게 나타나는 데이터셋이며, BERTSUM 논문 이전부터 많은 논문에서 말하였듯이 언어 모델은 크기가 커질수록 성능이 좋아진다. 이러한 점들을 볼 때 BERTSUM 모델 중 추출 요약 모델인 BERTSUMEXT가 BERTSUMABS나 BERTSUMEXTABS에 비해 성능이 좋을 것이며 모델의 크기가 클 수록 더 좋은 성능을 발휘할 것이라 충분히 예상이 가능하기 때문에 CNN/DailyMail 데이터셋을 이용한 실험에서는 당연히 BERTSUMEXT, 그 중에서도 크기가 큰 것이 가장 좋은 성능을 낼 것임을 예측할 수 있다. <br><br>


<p align="center">
  <img src="/assets/img/paper/BERTSUM/ROUGE_NYT_res_img.PNG" width="600" height="500"><br>
  < ROUGE score results on NYT >
</p>

위의 표는 NYT 데이터셋에 대한 성능의 ROUGE score 측정 결과이다. 우선 NYT 데이터셋은 Abstractive Summarization task라고 볼 수 있는 데이터셋이기 때문에 Abstractive model들의 성능이 비교적 좋게 나올 것이라 예측된다. 표의 각 블록들은 앞서 CNN/DailyMail 데이터셋 실험 결과와 같은 항목들이며 가장 아래의 BERT-based 블록은 BERTSUM 모델들의 성능을 나타낸다. 그 결과 BERTSUMABS 모델이 좋은 성능을 얻어내는데, 저자가 제안한 two-stage fine-tuning 기법을 사용한 BERTSUMEXTABS 모델이 가장 좋은 성능을 얻어내는데 성공한다. <br><br>

<p align="center">
  <img src="/assets/img/paper/BERTSUM/ROUGE_XSum_res_img.PNG" width="600" height="400"><br>
  < ROUGE score results on XSum >
</p>

위 표는 XSum 데이터셋에 대한 성능의 ROUGE score 측정 결과다. XSum 데이터셋은 매우 abstractive한 한 문장의 요약문을 정답으로 하는 데이터셋이다. 그래서 Abstractive model들의 성능이 더 좋게 나타난다. 실험 결과를 보면 BERTSUMEXTABS가 BERTSUMABS보다 약간의 성능 상승이 있음을 볼 수 있다. 저자는 XSum 데이터셋으로 Extractive model들의 성능을 측정하는 것이 의미없다 판단하여 Extractive model 성능은 없앴다. <br><br>

<p align="center">
  <img src="/assets/img/paper/BERTSUM/EXT_POS_img.PNG" width="600" height="400"><br>
  < Position of Extracted Sentences >
</p>

저자는 또한 extractive model을 사용 할 때 자주 요약문으로 사용되는 문서 내 문장 위치 비율 측정에 대한 실험도 진행했다. 그 결과 요약문에 사용된 많은 수의 문장들이 문서 초반에 분포되어 있었고, ORACLE 요약의 경우 문서 전체적으로 고르게 분포되어 있었다. 그에 반해 TransformerEXT는 대부분이 문서 첫 문장에 쪽에 집중되어 있었고, BERTSUMEXT의 경우 이 보다는 조금 더 ORACLE의 결과에 비슷하였다. <br><br>

<p align="center">
  <img src="/assets/img/paper/BERTSUM/Novel_n_gram_img.PNG" width="500" height="600"><br>
  < Proportion of Novel N-gram >
</p>


그리고 저자는 ROUGE score 측정뿐만 아니라 사람이 직접 모델의 성능을 평가하게 하기도 하였는데, 실험 방식은 다음과 같다. 우선 원래 문서는 제공하지 않고 모델로 생성한 요약문만 제공한다. 그리고 원래 문서에 관한 몇가지 질문을 하는데 이 질문들에 대해 잘 대답할수록 더 좋은 요약문이라 평가하게 된다. 

<p align="center">
  <img src="/assets/img/paper/BERTSUM/Human_eval_EXT_img.PNG" width="400" height="200" ><br>
  < Human Evaluation Results for Extractive Model >
</p>

<p align="center">
  <img src="/assets/img/paper/BERTSUM/Human_eval_ABS_img.PNG" width="530" height="210"><br>
  < Human Evaluation Results for Abstractive Model >
</p>

위는 각각 Extractive, Abstractive 모델에 대한 성능 평가 결과이다. 둘 모두에서 BERTSUM 모델이 SoTA 성능을 기록했다. 아래 표의 GOLD는 정답 summary를 나타내며, LEAD의 경우 앞서 말했던 LEAD-3와 같이 맨 앞의 문장들을 요약문으로 사용한 것이다.

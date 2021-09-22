---
layout: post
title: "[Paper Review] BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding"
categories: [Paper]
---

---

처음 포스팅할 논문은 BERT이다. 이전에 읽었던 논문들도 좀 있으나 우선 지금 읽고 있는 BERT부터 한번 정리해보려 한다. <br>
BERT는 당시 제안됐던 다른 모델들을 뛰어넘고 11개의 NLP task에서 SoTA 성능을 달성했다. 그로인해 많은 관심을 받았으며, <br>
현재까지도 BERT의 다양한 변형 버전이 나오고 있는 것으로 알고 있다. 논문을 계속 읽고 정리하다보면 BERT base의 모델을 <br>
많이 볼 수 있으리라 생각한다. <br><br>
BERT는 _**B**idirectional **E**ncoder **R**epresentations from **T**ransformers_ 의 약자로, Transformer Encoder를 이용해서 만든 양방향 <br>
멀티태스팅 언어모델이다. 논문의 Abstract 부분에서는 BERT의 특징에 대해서 간단하게 설명하고 있다. 요약하자면 BERT는 <br>
+ __Pre-train__
+ __Bidirectional Representation__
+ __Training by Using Unlabeled Text__
+ __Fine-tuning__
+ __Multitasking__
<br><br>
으로 나타낼 수 있다. 이는 GPT-1모델과 달리 단어 토큰들의 표현에 양방향 정보를 담았으며, Transformer Encoder를 사용했다는 <br>
점에서 차이를 보인다. 또한 학습 방식에서도 차이를 보이는데 이는 잠시 뒤에 다루도록 하겠다. 이제 논문을 따라 BERT에 대해 <br>
정리하려한다. <br><br>

## 1) BERT가 뭔데?
---

BERT는 우선 언어모델이다. 언어모델이 뭐냐면, 단어 시퀀스에 확률을 할당하는 일을 하는 모델이다. 이게 무슨 말이냐면, 단어가 <br>
하나 주어진다면 그 다음 단어로 가장 자연스러운 단어가 오도록 해당 단어에 확률을 부여하는 것이다. 간단하게 말하면 주어진 <br>
단어의 다음 단어 예측을 학습하는 모델이다. 그런데 BERT는 그냥 언어모델이 아니라 Multitasking이 가능한 언어모델이다. <br>
그러니까 다음 단어를 예측해 문장을 만드는 일만 하는 언어모델이 아닌 번역, 질의응답, 문장 분류와 같은 여러가지 task를 수행할 <br>
수 있는 그런 모델이다. <br>
#### BERT : Multitasking Language Model

그리고 BERT는 Pre-train - Fine-tuning의 접근법을 가지고 있다. 대량의 Unlabeled text data로 사전학습을 거친 후에, 소량의 <br>
task별 데이터셋을 이용한 추가 학습과 하이퍼파라미터 재조정을 통해 성능을 끌어올리는 방식을 말한다. ELMo나 GPT-1과 같은 <br>
방식을 사용하며 이로 인해서 좀 더 좋은 성능을 끌어낼 수 있게 된다. <br>
#### BERT : Pre-training + Fine-tuning Model

<br>

## 2) BERT의 모델 구조는?
---

BERT는 앞서 말했듯이 Transformer Encoder를 이용해 모델을 설계했는데, Encoder만을 떼서 여러 층으로 쌓아 BERT를 구성했다. <br>
<p align="center">
  <img src="/assets/img/paper/BERT/model_architecture_img.PNG" >
</p>
그림과 같이 단어 임베딩이 입력으로 주어지면 Transformer Encoder layer(Trm)들을 통과하여 결과값을 도출하게 된다. BERT는 <br>
모델의 크기에 따라서 BERT Base, BERT Large 두 가지를 만들었는데, Base 버전은 12개, Large버전은 24개의 encoder를 쌓았다. <br>
BERT base 모델은 GPT와 같은 크기의 파라미터들을 가진다. 직전 SoTA 모델이 GPT여서 그와 비교하려고 그렇게 설정한 듯하다.<br>
GPT와 같은 수의 파라미터를 가지더라도 더 좋은 성능을 얻어낼 수 있음을 강조하기 위해 그런 것 같다. 그리고 Large는 BERT의 최대<br>
성능을 끌어낼 수 있는 파라미터 수를 설정했다고 생각하면 될 것 같다. 이제 대략적으로 어떤 모델인지에 대해 살펴봤으니 하나씩<br>
뜯어보자.<br><br>

## 3) Model Specifications & Experiments
---
<br>

### 1. Input & Output Representations
NLP에는 여러가지 종류의 task가 존재하는데, 각각의 task들은 여러가지 특징들을 갖고, 그 특징에 맞는 별도의 입출력이 존재한다. <br>
이러한 여러가지 down-stream tasks에 BERT로 다루려면 각 task별로 알맞은 입출력이 설정되어야한다. 입력은 하나의 문장이 될 <br>
수도 있고, 한 쌍의 문장 또는 다른 형식이 될 수도 있다. BERT에서는 이 문제를 해결하기 위해 전부다 그냥 하나의 sequence로 <br>
취급해서 입력으로 사용했다. 가장 단순하면서도 확실한 방법인거 같긴하다. <br>
여기까지 정리하고보니 약간 의문이 드는게 이렇게 시퀀스 하나로 처리하면 그 시퀀스가 시퀀스의 최대길이를 넘기면 어떻게<br>
처리하나 싶다. 시퀀스 두개로 입력하게 되려나. <br><br>

BERT에는 WordPiece embedding이라는 30,000개의 token들로 이루어진 vocabulary를 사용해서 시퀀스의 token embedding을 <br>
표현했다. BERT에 입력되는 시퀀스들은 항상 **[CLS]**라는 토큰으로 시작한다(Special Classification token). 이 토큰은 classification <br>
task가 아니라면 무시해도 된다. 그리고 시퀀스에 여러개의 문장이 존재하면 두가지 방법을 통해 각 문장들을 구분한다. 첫번쨰로 <br>
시퀀스내에서 각 문장들은 **[SEP]**라는 다른 토큰으로 구분한다. 그리고 두번째 방법으로는 **Segment Embedding**이라는 별도의 <br>
임베딩을 이용해서 어느 문장에 속하는지 판별해 준다. Segment Embedding은 고정된 값으로 문장을 구분짓는 값이 된다.  이렇게 <br>
WordPiece embedding과 [CLS], [SEP] token들을 이용해서 시퀀스를 표현한다. 하지만 Transformer 구조를 사용한 BERT의 특성상 <br>
위치 정보 전달을 위한 Positional Embedding 또한 더해져야한다. 그리고 앞서 말했던 Segment Embedding을 통해 문장간 구분을<br>
지어준다. 결국 BERT의 입력은 다음 그림과 같게 된다.
<p align="center">
  <img src="/assets/img/paper/BERT/input_representation_img.PNG">
</p>
Token Embedding, Segment Embedding과 Position Embedding 셋을 더해서 BERT의 최종 입력으로 사용한다. 입력된 시퀀스의 [CLS]<br>
토큰은 일련의 계산과정을 거친 후 마지막에 출력되는 hidden vector에서 $$C$$로 표기되고 추후 말할 pre-training에서 사용된다.<br>
이렇게 만들어진 BERT의 입력은 모델에 입력되어 학습에 사용된다. BERT 학습의 첫번째 단계인 Pre-training은 어떻게 이루어질까?<br><br>

### 2. Pre-training BERT
우선 BERT는 pre-training시 연속된 긴 시퀀스 추출을 위해 BooksCorpus와 English Wikipedia 두 개의 corpus를 사용한다. 
BERT의 pre-training은 두 가지의 unsupervised task로 진행된다. 기존의 language model들은 단방향의 정보만을 이용해서 학습하는데 그에 <br>
반해 BERT는 양방향 정보를 활용해 모델을 학습시킨다. BERT의 두가지 pre-training 방식은 **Masked LM (MLM) & Next Sentence**<br> 
**Prediction(NSP)**이다. <br><br>

#### Masked Language Model (MLM)
MLM은 간단히 말해 입력 토큰들의 일부를 [MASK] 토큰으로 바꾼 후 마스킹된 토큰들의 원래 토큰을 예측하며 학습하는 방식이다. <br>
이러한 방식을 채택한 이유는 Bidirectional 구조에 있다. 이전의 LM들이 Unidirectional 구조를 채택한 이유는 양방향 구조를 사용할 <br>
경우 각 단어들이 자기 자신을 간접적으로 불 수도 있기 때문이라고 한다. 이러한 현상을 방지하기 위해 BERT는 양방향 구조를 <br>
사용하며 MLM 방식을 채택하였다. [MASK]토큰은 입력되는 WordPiece Embedding 토큰들의 15%를 랜덤하게 가리고, 가려진 토큰들
만 예측하여 원래의 문장으로 복구하는 방식으로 학습을 진행한다. <br>
MLM은 Bidirectional pre-trained model을 얻어낼 수 있는 장점을 가지고 있으나 단점도 존재한다. [MASK]토큰은 실제로는 전혀 <br>
사용하지 않는 단어이다. 그러므로 fine-tuning 시에 등장하지않는데, 이로 인해 pre-training과 fine-tuning 간의 괴리가 발생하는 <br>
문제가 생긴다. BERT 저자는 이를 완화하기 위해 또다른 기법을 추가한다. 마스킹할 단어들을 [MASK]토큰으로만 바꾸는 것이 아닌<br>
일정 비율로 다른 토큰으로 바꾼다는 것이다. 한 토큰을 마스킹한다고 가정했을 때 아래의 비율로 마스킹을 실시한다. 
* 80% : [MASK] token으로
* 10% : Random token으로
* 10% : 그대로
<br>

#### Next Sentence Prediction (NSP)
Next Sentence Prediction은 두 문장이 있을 때 두 문장이 실제로 연결되는 문장인지 아닌지를 판별하는 binary classification <br>
문제이다(IsNext / NotNext). Pre-training시 문장 A, B가 있다고 할 때 시행 횟수 중 50%는 실제로 A 다음에 위치한 문장을 B로 <br>
설정하고, 나머지 50%는 corpus내의 랜덤한 문장을 B로 설정한다. <br>
Language model에서는 포착할 수 없는 두 문장간의 관계는 Question Answering, Natural Language Inference 등 여러가지 주요 <br>
NLP task에 중요하게 쓰인다. NSP task는 꽤 단순한 방법이지만 두 문장간의 연관성을 이용하는 task들의 성능에 큰 이득을 준다.<br><br>

### 3. Fine-tuning BERT
Pre-training 이후에는 fine-tuning 단계가 이어지는데 각각의 NLP task에 맞는 데이터를 이용해 파라미터들을 다시 조정해주는 <br>
단계이다. Pre-training으로 학습된 language model을 전이 학습(transfer learning)으로 task별 성능을 끌어올리게 된다. BERT의<br>
fine-tuning은 Transformer의 self-attention 매커니즘 덕분에 꽤 단순해진다. 또한 pre-training에 비해 계산 비용도 저렴하다. <br>
각 task별로 fine-tuning이 어떻게 진행되는지 살펴보자. <br>
<p align="center">
  <img src="/assets/img/paper/BERT/fine_tuning_img.PNG">
</p>
<br>
위의 그림에서는 우선 4가지 NLP task들의 fine-tuning을 설명하고 있는데, 순서대로 Sentence Pair Classification, Single Sentence<br>
Classification, Question Answering, Single Sentence Tagging task이다. a, b를 sequence-level task(sequence-> one label), c, d를<br>
token-level task(sequence -> label for each tokens)라고 한다. <br>
BERT의 fine-tuning은 여러가지 데이터셋을 이용해서 실험되었다. 우선 첫번째로 GLUE다.<br><br>

#### GLUE (General Language Understanding Evaluation)
GLUE는 다양한 NLU task들을 모아놓은 데이터셋이다. 실험 결과는 아래의 표와 같다. <br>
<p align="center">
  <img src="/assets/img/paper/BERT/GLUE_result_img.PNG">
</p>
결과를 보면 BERT base와 large 모두 이전 모델들의 성능을 크게 능가했다. 또한 주목해야할 점은 BERT large 모델이 BERT base <br>
모델의 성능을 모두 능가했다는 것인데, 이는 훈련 데이터 양이 적은 task에서도 같은 결과를 보인다. 모델 크기와 성능의 관계는 <br>
ablation study 부분을 볼 때 다루겠다. <br><br>

#### SQuAD v1.1 & v2.0 (Stanford Question Answering Dataset)
Stanford 대학에서 만든 QA task 데이터셋이다. 질문과 지문이 주어지면 지문에서 질문에 걸맞는 답을 예측하는 방식으로 수행된다. <br>
QA task에서도 BERT는 새로운 SoTA를 갱신한다. <br>
<p align ="center">
  <img src="/assets/img/paper/BERT/SQuAD_result_img.PNG" width="450" height="400">
</p>
위의 결과 표를 보면 TriviaQA라는 것이 있는데 이는 SQuAD 데이터셋을 학습시키기 전에 먼저 공개 데이터인 TriviaQA로 학습한 <br>
것이다. 하지만 이러한 추가적인 학습 데이터가 없어도 이미 BERT는 기존 모델의 결과를 크게 능가한다. <br><br>
SQuAD v2.0 또한 v1.1과 같이 BERT가 이전의 모델들 보다 더 좋은 성능을 보인다.<br>
<p align="center">
<img src="/assets/img/paper/BERT/SQuAD2_result_img.PNG" width="480" height="400">
</p>

<br><br>

#### SWAG (Situation With Adversarial Generations)
SWAG은 grounded common-sense inference라고 하는 task의 데이터셋이다. 이는 간단하게 말하자면 문장 A 다음에 올 문장 B를 <br>
4개의 후보 문장 중에서 가장 그럴듯한 것으로 선택하는 task다. SWAG으로 fine-tuning한 결과 마찬가지로 SoTA값을 달성했다. <br>
<p align="center">
  <img src="/assets/img/paper/BERT/SWAG_result_img.PNG">
</p>
<br><br>

### 4. Ablation Studies
BERT의 저자는 여러 부분에 대해서 ablation study를 진행했다. <br><br>

__4-1) Effect of Pre-training tasks(NSP, MLM)__

<p align="center">
  <img src="/assets/img/paper/BERT/ablation_study_img.PNG">
</p>

저자는 우선 pre-training 단계에서 학습을 위해 사용되는 task들이 성능에 어느정도 영향을 미치는지 실험했다. 실험 결과 <br>
NSP와 MLM 둘 중 하나라도 빠지게 되면 꽤나 성능이 하락하는 것으로 나타난다. 여기서 LTR은 Left to Right로 MLM을 제거한 <br>
경우의 학습 방식이다. LTR 모델을 보조하기 위해 BiLSTM layer를 모델 마지막에 추가하여도 성능의 개선은 이루어지지 않았다. <br><br>

__4-3) Effect of Model size__

<p align="center">
  <img src="/assets/img/paper/BERT/ablation_model_size_img.PNG">
</p>

또한 모델의 크기가 성능에 어느정도 영향을 미치는지에 대해서도 실험했다. 표를 보면 모델의 크기가 커지면 커질수록 모델의 성능 <br>
또한 좋아지는 것을 알 수 있다. 실험으로 얻어지는 결과를 보면 BERT large 모델의 성능이 base 모델보다 좋은 것을 볼 수 있는데 이 <br>
실험으로 모델의 크기가 모델의 성능을 좌우함을 이해할 수 있다. <br>
#L : number of layers <br>
#H : hidden size<br>
#A : number of attention heads<br><br>

__4-4) Feature-based Approach with BERT__

<p align="center">
  <img src="/assets/img/paper/BERT/ablation_feature_based_img.PNG" height="500" width="550">
</p>

BERT는 앞서 설명했듯 fine-tuning 방식으로 downstream task들을 수행한다. 이번 실험에서는 fine-tuning이 아닌 feature-based <br>
approach를 이용해서 성능을 측정한다. Feature-base는 사전에 학습된 모델의 고정된 language representation을 이용해 학습을 <br>
진행하게 된다. Feature-base 방식은 task-specific하기 때문에 대부분의 task에서 괜찮은 성능을 얻어낼 수 있다. 그리고 사전에 <br>
미리 계산된 representation을 사용해서 fine-tuning approach에 비해 계산비용이 적게 든다. BERT에 feature-based approach를 <br>
적용시켜 실험했을 때 위 표에서 볼 수 있듯이 기존 fine-tuning approach BERT에 비해 크게 성능 하락이 관찰되지 않았다. 이로써<br>
BERT는 fine-tuning 방식을 사용했을 때만이 아니라 feature-based approach에서도 좋은 성능을 얻을 수 있음을 알 수 있다. <br><br>

__4-5) Effect of Number of Training Steps__
<p align="center">
  <img src="/assets/img/paper/BERT/ablation_num_training_img.PNG" height="500" width="650">
</p>

BERT는 pre-training 횟수에 따라 성능이 차이가 난다. 위의 표는 서로 다른 pre-training 횟수에 따라 MNLI 데이터셋의 성능 차를 <br>
나타낸 표이다. 표에서 알 수 있듯이 pre-training 횟수가 늘어날 수록 성능이 좋아진다. Pre-training 단계가 BERT의 성능에 영향을<br>
끼친다는 것을 알 수 있다. <br><br>

__4-6) Ablation of Different Masking Procedure__
<p align="center">
  <img src="/assets/img/paper/BERT/ablation_masking_img.PNG">
</p>

앞에서 BERT pre-training task중 MLM을 설명할 때 pre-training과 fine-tuning간의 괴리감을 줄이기 위해 일정 비율로 마스킹을 <br>
[MASK]토큰, 랜덤 토큰, 그대로 유지 중 선택해서 진행한다고 설명했다. 이 비율은  실험으로 얻어진 결과인데 실험은 위의 표와 같이<br>
진행되었다. 실험은 MNLI 데이터셋과 NER에서 진행되었다. 실험 결과 두 데이터셋에서의 성능이 모두 좋았던 것은 [MASK] 80%, Random과<br>
그대로 유지하는 것을 각 10% 씩 했을 때였다. 또 알 수 있는 점은 fine-tuning은 어떠한 마스킹 방식을 쓰더라도 준수한 성능을 보인다는<br>
것이다. 그에 반해 feature-based는 전부 다 [MASK]토큰으로 마스킹 시 성능이 눈에 띄게 떨어진다는 것을 확인할 수 있다. <br><br>

## 4) Conclusion
---
이 논문 이전의 단방향 표현을 이용한 Pre-training + Fine-tuning LM은 많은 양의 unsupervised pre-training이 성능 개선의 핵심임을<br>
밝혀냈다. BERT는 그러한 발견을 이제 uni-directional representation이 아닌 bi-directional representation에서도 적용시키며, <br>
bi-directional representation을 이용한 LM도 많은 양의 unsupervised pre-training이 모델 성능 개선의 핵심임을 입증했다.
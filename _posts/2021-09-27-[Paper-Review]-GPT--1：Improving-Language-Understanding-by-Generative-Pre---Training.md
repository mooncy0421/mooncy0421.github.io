---
layout: post
title: "[Paper Review] GPT-1：Improving Language Understanding by Generative Pre-Training"
categories: [Paper]
---

---

이번에 정리할 논문은 [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)이다. 이 논문은 GPT를 만든 회사로 잘 알려진 <br>
OpenAI에서 만든 논문이다. 이 논문으로 GPT의 시작을 알렸으며 현재까지 GPT-3이 나와있다. 이 논문은 Pre-training + Fine-tuning<br>
으로 multi-tasking model을 학습시키자 하였고, GPT-2부터는 fine-tuning과정 없이 pre-training만으로 학습을 시키려했다. 이 논문은<br>
이전에 [집현전 2기](https://github.com/jiphyeonjeon)에서 팀으로 발표했고 발표영상은 해당 깃허브내 링크로 올라와있다.<br><br><br>


# 1) INTRODUCTION
---
GPT-1의 핵심은 논문의 Abstract 부분에서 나온다. *Generative pre-training of a language model on a diverse corpus of unlabedled text,*<br>
*followed by discriminative fine-tuning on each specific task.* 저자의 말에 따르면 GPT-1의 학습은 우선 대용량의 unlabeled corpus로<br>
pre-training 후 task에 맞는 fine-tuning을 통해 모델의 파라미터들을 조정하는 것으로 이루어져 있다.<br>
이러한 방식으로 보았을 때 GPT-1은 크게 보면 semi-supervised learning 방식의 모델이라고 할 수 있다. 이전의 semi-supervised <br>
learning 모델들은 학습을 통해 word-level information 이상의 phrase-level 또는 sentence-level information을 얻어내어 다양한 task에 <br>
맞게 encoding하여 사용하고자 했다. 하지만 unlabeled text에서 word-level 이상의 정보를 얻어내는 것은 두 가지의 문제 때문에 <br>
어려웠다. <br>
우선 무엇이 task에 맞게 transfer하는 데에 유용한 representation 학습에 효과적인 optimization objective인지 모르며,<br>
학습된 representation을 효과적으로 transfer하는 방법 또한 모른다는 것이 문제였다.<br><br>

저자 또한 word-level 이상의 정보를 이용하여 NLP task들을 풀고자 하였고 위의 두 문제를 2단계의 학습 방식으로 해결하려 하였다.<br>
1. 모델의 초기 파라미터 학습을 위한 objective로 language modeling을 선택하여 unlabeled data로 학습시킨다.
2. 학습시킨 파라미터를 task별 supervised objective로 해당하는 task에 적용시킨다.

이것이 우리가 익히 알고 있는 generative pre-training + supervised fine-tuning 방식이다. 저자는 학습한 모델로 여러 NLU task를<br>
실험했고, 그 결과 대부분의 task에서 당시의 SoTA값을 달성할 수 있었다.<br><br><br>

# 2) MODEL ARCHITECTURE
---
GPT-1의 모델 구조 특징은 다음과 같다. <br>
+ Transformer Decoder
+ Language Model
+ Unsupervised pre-training
+ Supervised fine-tuning
+ Task-specific input representation
<br>

## Transformer Decoder
GPT-1 모델에는 transformer의 decoder를 여러층 쌓아 사용했다. Transformer 사용 시 RNN등 다른 구조를 사용할 때 보다 long-term <br>
dependency를 다루는데 좋고 그 결과 transfer시 여러 task에 robust한 성능을 얻어낼 수 있기에 transformer 구조를 선택했다.<br><br>

## Language Model & Unsupervised pre-training
GPT-1의 학습 첫 단계는 앞서 말했듯이 language modeling으로 unlabeled data를 학습하여 모델의 초기 파라미터를 얻어내는 것이다.<br>
GPT는 기본적인 language modeling objective를 사용하는데 다음과 같다.<br>

$$ L_1(\mathcal{U})=\sum_{i}\log P(u_i|u_{i-k},...,u_{i-1};\mathsf{\Theta}) $$

$$ where, \mathcal{U}={u_1,...,u_n} $$

Unsupervised pre-training은 SGD를 통해 파라미터 $$\mathsf{\Theta}$$를 학습하며 $$L_1$$값을 최대화하는 방향으로 진행된다. 위 식에서 $$\mathcal{U}$$는 unlabeled <br>
corpus의 토큰들이고 k는 context window size, P는 $$\mathsf{\Theta}$$로 모델링된 conditional probability이다. GPT 모델은 이 language model에 <br>
multi-layer Transformer decoder를 사용했고 그 구조는 원래 Transformer 구조와 같다. 우선 input tokens에 대해 multi-headed self-<br>
attention후 position-wise feed-forward, softmax 연산을 수행한다. <br>

$$ h_0 = UW_e + W_p \\
h_l = \mathrm{transformer\_ block}(h_{l-1})\forall i \in [1,n] \\
P(u) = \mathrm{softmax}(h_nW_e^T) $$

여기서 $$n, W_e, W_p$$는 각각 layer 수, token embedding matrix, position embedding matrix를 나타낸다. <br><br>

## Supervised fine-tuning
Unsupervised pre-training의 objective $$L_1$$으로 학습된 파라미터는 target task에 맞는 supervised learning 과정을 거친다. 식은<br>
다음과 같다.<br>

$$ L_2(\mathcal{C}) = \sum_{(x,y)}\log P(y|x^1,...x^m)\\
   P(y|x^1,...x^m) = \mathrm{softmax}(h_l^mW_y) $$

여기서 $$\mathcal{C}$$는 labeled dataset을 나타내고 이는 input token들의 sequence $$x^1,...,x^m$$과 그에 따른 label $$y$$로 구성되어 있다.<br>
Supervised fine-tuning은 pre-training과 비슷하게 objective $$L_2$$값을 최대화하는 방향으로 진행된다. <br><br>

저자는 여기에 추가로 auxiliary objective를 추가하여 fine-tuning시 모델의 convergence를 돕고 generalization을 개선하고자 했다.<br>

$$ L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda * L_1(\mathcal{C}) $$

식을 보면 auxiliary objective로 pre-training시의 language model을 사용하며 해당 objective에 weight $$\lambda$$를 곱해 사용한다. 전반적으로 <br>
보면 Fine-tuning 단계에서 추가로 필요한 파라미터는 $$W_y$$와 delimeter tokens를 위한 embedding 밖에 없다. delimeter token은 input <br>
transformation에서 설명할 것이다.<br><br>

## Task-specific input transformation

<p align="center">
  <img src="/assets/img/paper/GPT1/input_form.PNG">
  < Input Transformation >
</p>

위 사진은 GPT-1의 Transformer(왼쪽)와 fine-tuning시 각 task별 input transformation을 나타낸 것(오른쪽)이다. Text classification같은<br>
경우에는 별도의 모델이나 input 형식 수정없이 fine-tuning이 가능한 반면 Question Answering, Textual Entailment의 경우, 별도의<br>
input 형식을 갖고 있어 조치가 필요하다. 이전의 모델들은 task-specific architecture를 사용하여 문제를 해결하였으나 task가 바뀔 때 <br>
마다 새로운 architecture를 요구한다는 단점이 있다. GPT-1은 이를 막고자 각 task별 input transformation을 만들어 모델에 넣어줬다. <br>
그로 인해 architecture의 변화는 최소화하면서 여러 task를 수행할 수 있는 모델을 만들게 되었다. 두 종류 이상의 sequence가 필요한 <br>
task의 경우에 각 sequence 사이에 delimeter token을 추가하여 sequence간 구분이 가능하도록 만들었다. 위 그림에서 similarity task<br>
의 경우에는 GPT-1 모델이 uni-directional information만을 취급하기 때문에 Text1 -> Text2, Text2 -> Text1 각각의 similarity를 얻어내야 <br>
하기 때문에 같은 sequence의 순서만 바꿔 두번 입력한다.<br><br><br>

# 3) MODEL SPECIFICATIONS
---
GPT-1 모델의 세부 정보를 요약하자면 다음과 같다.<br>

+ 12-layered decoder-only Transformer
+ 12 attention heads / 768 dimensional states
+ Position-wise feed-forward / 3072 dimensional inner states
+ Adam Optimizer (Max learning rate : 2.5e-4)
+ Mini-batch size : 64
+ Sequence size : 512
+ Byte Pair Encoding
+ Dropout rate : 0.1
+ L2 Regularization
+ Activation function : GELU (Gaussian Error Linear Unit)
+ Pre-training : 100 epochs
+ Fine-tuning : 3 epochs
+ Fine-tuning learning rate : 6.25e-5
+ Fine-tuning batch size : 32
<br><br>


# 4) EXPERIMENTS
---
## Datasets
GPT-1의 pre-training에는 BooksCorpus dataset이 사용되었다. BooksCorpus는 긴 길이의 텍스트가 포함되어있어 long-range <br>
information 학습에 용이하다. <br>

<p align="center">
  <img src="/assets/img/paper/GPT1/finetuning_dataset.png">
  < Fine-tuning Dataset >
</p>

Fine-tuning시에는 각 task별로 다른 dataset을 이용하여 학습했다. <br><br>

## Results

<p align="center">
  <img src="/assets/img/paper/GPT1/NLI_result.PNG">
  < NLI task results >
</p>

Inference task에서는 RTE dataset을 제외한 나머지에서 모두 GPT가 SoTA를 달성했다.<br>

<p align="center">
  <img src="/assets/img/paper/GPT1/QA_Commonsense_result.PNG">
  < Question Answering & Commonsense Reasoning results >
</p>

QA와 Commonsense reasoning에서는 보두 SoTA를 달성했다. <br>

<p align="center">
  <img src="/assets/img/paper/GPT1/classification_semantic_similarity_result.PNG">
  < Classification & Semantic Similarity results >
</p>

Text classification과 Semantic similarity에서는 각각 1개의 dataset을 제외한 나머지 모두에서 SoTA를 달성했다. <br>
GPT-1은 실험한 dataset 12개 중 9개에서 SoTA를 달성하며 좋은 성능을 나타냈다. <br><br>

## Pre-training experiments
저자는 또한 pre-training에 관해서도 실험을 진행했다. fine-tuning시 pre-train된 layer를 transfer한 수에 따른 성능 변화와 pre-training<br>
양에 따른 성능 변화에 대해 실험했다. 

<p align="center">
  <img src="/assets/img/paper/GPT1/num_transfer.PNG"> <br>
  < Effect of transferred layers >
</p>

위 표는 pre-train된 layer들을 fine-tuning시 얼마나 transfer하느냐에 따른 accuracy 변화이다. 표를 보면 알 수 있듯이 더 많이 transfer<br>
할수록 성능은 더 좋아진다. 이는 곧 pre-trained layer에 downstream task에 유용한 정보가 학습됨을 나타낸다. <br><br>

<p align="center">
  <img src="/assets/img/paper/GPT1/num_pre_training.PNG"> <br>
  < Zero-shot performance of different pre-training updates >
</p>

위 표는 여러 task에서의 Transformer와 LSTM의 zero-shot 성능을 나타낸 것이다. 표를 보면 Transformer구조가 LSTM에 비해 더 좋은 <br>
성능을 나타내며 pre-training 횟수에 따른 성능 증가가 꾸준히 이루어짐을 볼 수 있다. 이로 보았을 떄 LSTM보다 Transformer 구조가 <br>
더 안정적으로 성능이 개선되며 더 좋은 성능을 이끌어냄을 알 수 있고, pre-training 횟수의 증가에 따라 zero-shot 성능이 좋아짐을 <br>
알 수 있다. <br><br>

## Ablation study

<p align="center">
  <img src="/assets/img/paper/GPT1/ablation_study.PNG">
  < Ablation Study >
</p>

저자는 ablation study를 3 종류 진행했다. Auxiliary objective, Transformer, pre-training 각각에 대해 진행한 결과를 정리하면<br>
다음과 같다. <br>
+ Auxiliary Objective : 큰 dataset에는 성능 개선 / 작은 dataset에서는 딱히 없음
+ Transformer Architecture : LSTM에 비해 확실한 성능 개선
+ Pre-training : 성능 개선
<br><br><br>


# CONCLUSION
---
GPT-1은 Transformer 구조를 활용한 pre-training + fine-tuning 접근 방식 모델의 시작을 알린 모델이다. 뒤를 이어 BERT, XLNET등<br>
여러가지 pre-training 모델이 대거 등장했으며 GPT또한 계속해서 발전을 거듭하며 새로운 버전이 나오고 있다. <br>
저자는 이후 GPT-2 논문에서 auxiliary objective를 제거하고 아예 fine-tuning 과정을 없애고 pre-training만으로 multi-tasking을<br>
하고자 시도한다. 
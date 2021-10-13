---
layout: post
title: "개인 프로젝트[사전공부] Text Summarization"
categories: [Project]
---

---

개인 프로젝트를 하나 진행해보려 한다. 뭔가 만들만한 실력은 없는 것 같지만 언제까지고 실력이 부족하다며 아무것도 안하는 것은 <br>
더더욱 아니라고 생각해서 목표를 정해놓고 조금씩 해보려한다. 무엇을 만들어 볼지 찾아보다 꽂힌게 text summarization 기술과 <br>
machine translation 기술을 합쳐서 해외 뉴스를 한글로 요약하는 모델을 만들어 보고자 한다. 아직 둘중 하나도 제대로 모르는 상황<br>
이라 하나씩 공부해볼 것이다. 우선 Text summarization에 대해 지식이 없기 때문에 첫 포스팅은 고려대학교 DSBA 연구실에서 올린<br>
[유튜브 영상](https://www.youtube.com/watch?v=25TEdaQPqQY)을 정리해보려한다.<br><br>

# Text Summarization
우선 Text Summarization이 시작이다. Text summarization은 텍스트가 주어졌을 때 중요한 정보만을 정제하여 요약된 버전을 만드는<br>
task이다. 문서 요약 task는 요약 문장을 얻는 방식에 따라 크게 두 종류로 나눌 수 있다.<br>
__1. Extractive Text Summarization__ <br>
원문에서 중요도가 높다고 판단되는 문장들을 모아 요약문으로 제공하는 방식이다. 원문에 있던 문장을 그대로 가져오는 방식이라<br>
각 문장은 자연스러운 표현을 가지나 필요없는 표현이 덜 요약되거나 표현이 제한적일 수 있다는 단점이 존재한다.<br>
__2. Abstractive Text Summarization__ <br>
Extractive에 비해 좀 더 발전된 방식이다. 원문에서 중요한 내용을 담은 부분을 인식하고, 해당 부분을 해석하여 새로운 문장을 <br>
만들어 요약문을 생성하는 방식이다. Extractive에 비해 좀 더 다양하고 유연한 표현을 가지나 말이 안되는 부분이 존재할 수 있다.<br>
과거 문서 요약 task는 주로 Extractive 방식을 쓴 반면 최근 연구는 Abstractive가 주축을 이루고 있다.<br>

그 외에도 다른 기준으로 요약 task를 나눌 수 있는데, Input type으로 구분지을 경우 text가 하나이면 Single-Document, 여러개면<br>
Multi-Document라 하고, 요약 목적에 따라 Generic, Domain-specific, Query-based 방식으로 나눌 수 있다.<br>

+ Based on input type
  + Single-Document
  + Multi-Document
+ Based on output type
  + Extractive
  + Abstractive
+ Based on the purpose
  + Generic
  + Domain-specific
  + Query-based
<br><br>

## Supervised vs. Unsupervised
Text summarization은 학습 방식에 따라 두 종류로 볼 수 있다. Supervised와 unsupervised 둘 다 익숙할 것인데, 우선 supervised는<br>
original document와 사람이 직접 만든(Human-generated) summary를 같이 제공한다. 다시말해 문서 내의 문장이 가질 수 있는 <br>
features를 정의하고 각 feature를 바탕으로 summary에 문장이 포함될지 안될지 여부를 학습하는 과정을 supervised summarization<br>
이라고 한다. 대표적인 supervised dataset으로는 CNN, Daily Mail Newsroom dataset이 있다. <br>
Supervised 방식이 잘 작동하기 위해서는 ground truth summary(human-generated)의 양이 많아야 하며 그 퀄리티도 좋아야한다.<br>
하지만 데이터셋의 ground truth summary는 사람이 직접 만들기 때문에 그 양이 한정적이고 비용이 비싸다는 것이 단점이다. 또한 <br>
사람이 만든 요약문은 abstractive summarization 방식에 가까워 extractive summarization 학습 시에는 좋은 데이터셋이 아닐 수 있다.<br><br>

반면에 unsupervised 방식은 주어진 input document에서 중요한 부분을 파악한 다음 significance score(문서에 기여하는 중요도)를<br>
산출한다. 이를 위해 보통은 문서를 그래프로 나타내고, 문서 내의 문장을 그래프의 노드, 문장간 관계를 엣지의 강도로 나타낸다.<br>
그렇게 만든 그래프에서 가장 중요한 노드가 무엇인지를 판별하는 방식으로 요약 작업을 진행한다. 대표적인 알고리즘으로는 <br>
TextRank가 있다. <br><br>

## Evaluation
모델을 통해 요약문을 생성했으면 만들어진 요약문이 얼마나 좋은 결과물인지를 평가하는 방법이 있어야 한다. 현재 가장 널리 <br>
사용되는 방식으로는 ROUGE(Recall-Oriented Understudy Gisting Evaluation)가 있다. ROUGE score는 ROUGE-N으로 사용되는데<br>
이는 정답 summary와 모델이 만든 summary사이의 n-gram에 대한 recall을 뜻한다. 식은 다음과 같다.<br>

$$ \mathsf {ROUGE\!\!-\!\!N} = {\sum\nolimits_{S \in \{ Reference Summaries \} } \sum\nolimits_{gram_n \in S} Count_{match}(gram_n) \over \sum\nolimits_{S \in \{ Reference Summaries \} } \sum\nolimits_{gram_n \in S} Count(gram_n)} $$

위 식에서 분모는 정답 summary (ground truth)의 총 n-gram 수를 의미하고, 분자는 정답 summary와 모델 생성 summary 사이에 <br>
일치하는 n-gram의 수를 나타낸다. ROUGE-1은 unigram, ROUGE-2는 bigram, ROUGE-L은 정답과 모델 생성 summary 사이의 공통된<br>
부분 중 가장 긴 길이가 어느정도의 비율을 차지하는지를 계산하는 것이다. ROUGE-L은 간단하게 설명하자면 sequence의 순서를 <br>
유지하는 상태에서 일치하는 토큰의 갯수가 몇개인지를 계산하는 방식이다. 이 ROUGE score는 논문에서 말하는 원래의 ROUGE인데<br>
최근에는 위의 recall 버전이 아닌 precision 버전을 함께 사용하는 ROUGE F1 score를 자주 사용한다. <br>

$$ \mathsf {ROUGE\!\!-\!\!N_{recall}} = {number\; of\; n\!\!-\!\!gram\; in\; match \over number\; of\; n\!\!-\!\!gram\; in\; ground\; truth} $$

$$ \mathsf {ROUGE\!\!-\!\!N_{precision}} = {number\; of\; n\!\!-\!\!gram\; in\; match \over number\; of\; n\!\!-\!\!gram\; in\; model\; summary} $$

$$ \mathsf {ROUGE\!\!-\!\!N_{F1}} = {\mathsf {2ROUGE\!\!-\!\!N_{rec}ROUGE\!\!-\!\!N_{pre}} \over \mathsf {ROUGE\!\!-\!\!N_{rec}+ROUGE\!\!-\!\!N_{pre}}   } $$

F1 score는 다음과 같이 ROUGE recall과 precision 버전의 합과 곱으로 나타낼 수 있다.<br>
ROUGE-N 방식은 recall 버전만을 사용할 경우 완벽한 평가 방법을 제공하지는 못한다. 일치하는 n-gram의 존재 여부만 보기 때문에 <br>
n-gram의 순서가 뒤섞여 말이 되지 않더라도 좋은 요약문이라고 판단할 수도 있기 때문이다. ROUGE-1, ROUGE-2, ROUGE-N 중 어떤 <br>
것을 가장 많이 쓰는가에 대한 의문은 무의미하다고 한다. 셋 중 하나만을 평가 지표로 삼게 되면 제대로된 평가가 불가능하여 세 개<br>
모두를 사용하여 평가한다. ROUGE 평가 방식이 부족한 점이 많으나 현재까지 나온 평가 방식 중 사람이 요약한 결과물과 비교했을<br>
때 ROUGE score가 가장 비슷한 결과를 냈기에 ROUGE score가 높은 요약문을 좋은 요약문이라고 판단하는데에는 무리가 없다고 <br>
한다. <br><br>

하지만 ROUGE score에도 단점이 존재하는데 우선 첫번째로 ROUGE score는 평가 시 겹치는 n-gram만을 평가하기 때문에 문법적 <br>
오류, 자연스러움, 문맥 일치에 관련된 평가는 진행하지 않는다. 그로 인해 어색한 문장에도 높은 점수를 부여할 수도 있게 된다. <br>
그리고 겹치는 n-gram의 수를 이용해서 평가하기에 abstractive summarization 방식을 사용할 경우 동의어로 인해 잘 요약된 문장<br>
이더라도 낮은 점수를 받을 수 있게 된다. <br>
이러한 단점들을 극복하고자 이후 ROUGE 2.0이 발표되는데 문제점 극복을 위해 동의어에 대한 정보를 담고자 하였고, 원 문서의 <br>
주제에 대한 정보 또한 커버하고자 하였다. 하지만 이 또한 충분히 만족스럽지는 못했는데 우선 동의어 개념을 담고자 자체적 동의<br>
어 처리가 아닌 동의어 사전을 채용했다는 점, 그리고 토픽에 대한 정보를 담기 위해 문서마다 특정 품사의 단어를 한정하여 주제를 <br>
예측한 점으로 약간 불만족스러운 개선이라고 볼 수 있다.<br><br>

여기까지가 영상에서 다루어진 내용이다. 이 다음 포스팅은 우선 문서 요약 task의 전반적인 흐름에 대해서 알아야 한다 생각하기에 <br>
서베이 논문을 읽어보려한다. 논문은 올해 나온 [A Survey of the State-of-the-Art Models in Neural Abstractive Text Summariztion](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9328413)이다.
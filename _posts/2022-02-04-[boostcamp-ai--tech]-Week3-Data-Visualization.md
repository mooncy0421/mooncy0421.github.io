---
layout : post
title : '[boostcamp ai-tech] Week3 Data Visualization'
categories : boostcamp
---

---

# Data Visualization
---
데이터 시각화는 데이터를 그래픽적 요소로 매핑해서 시각적으로 보기 좋게 나타내는 것을 말한다. 사실 사람마다 보기좋다거나 알기 쉬운 시각화의 기준이 다양해서 이렇다할 정답은 없지만 데이터 시각화의 목적과 시각화하려는 대상, 보여주려는 독자 등 여러 요소에 의해 각 상황별로 모범적인 좋은 시각화들이 있긴하다. <br>
이번 주는 좋은 시각화를 만들기 위한 라이브러리 사용법과 몇 가지 팁에 관한 내용에 대한 강의가 있었다. <br><br>

## Dataset 종류
데이터를 시각화 할 때는 어떤 관점에 따라 시각화를 진행할지 선택해야한다. <br>
데이터셋 전체의 분포, 구성 등을 시각화하는 관점을 **데이터셋 관점(global)**<br>
데이터셋내의 개별 데이터 각각을 시각화하는 관점을 **개별 데이터 관점(local)**<br>
이라고 한다. <br>
그리고 데이터 시각화는 데이터셋의 종류에 따라 가독성을 좋게할 방법이 존재한다. 데이터셋의 종류는 매우 많은데 이를 크게 몇 가지 종류로 구분지어보면 다음과 같다. <br>

|데이터셋 종류|특징|
|:---|:---:|
|**정형 데이터**|테이블 형태로 제공<br>(csv, tsv)<br>가장 쉽게 시각화 가능<br>(ex: 데이터간 관계 및 비교, 통계적 특성과 feature간 관계)|
|**시계열 데이터**|시간 흐름에 따른 데이터<br>(기온, 주가, 음성, 비디오)<br>시간에 따른 추세, 계절성, 주기성 시각화|
|**지리 데이터**|지도 + 보려는 정보 또는 지도 자체를 단순화<br>거리, 경로, 분포 등 실사용처 다양함|
|**관계형(네트워크) 데이터**|객체 간의 관계 시각화(Graph/Network)<br>크기, 색, 수 등으로 객체와 관계간 가중치 휴리스틱하게 표현|
|**계층적 데이터**|데이터간 포함관계가 분명한 것들<br>(Tree, Treemap, Sunburst등)|
|**비정형 데이터**| 종류 매우 다양함 <br>(사진, 비디오, 음성, 텍스트 등)|

<br><br>
그리고 데이터셋의 데이터는 4가지로 분류가 가능하다. <br>

> + **수치형(Numerical)**
>   + 연속형(Continuous) : 연속된 실수형 데이터 (키, 온도)
>   + 이산형(Discrete) : 연속되지않은 값 가지는 데이터 (주사위 눈금, 인구수)
> + **범주형(Categorical)**
>   + 명목형(Norminal) : 순서 필요없는 범주형 데이터 (혈액형, 인종)
>   + 순서형(ordinal) : 순서 필요한 범주형 데이터 (학년, 별점)

<br><br><br>

## 시각화의 요소
데이터 시각화에서는 데이터를 **점, 선, 면**으로 나타낸다. 이를 데이터 시각화에서 **mark**라고 부른다. <br>

<p align="center">
  <img src="/assets/img/boostcamp/DataViz/intro_mark_img.png">
  <br> < Mark Example >
</p>

그리고 각 mark를 변경하는 요소들을 **channel**이라고 한다. <br>

<p align="center">
  <img src="/assets/img/boostcamp/DataViz/intro_channel_img.png">
  <br> < Channel Example >
</p>

Channel은 mark의 위치를 바꾸거나 색, 모양, 기울기, 길이 등을 바꾸어 데이터간의 구분을 준다. <br><br>

그리고 전주의적 속성(Pre-attentive Attribute)라는 것이 있는데 강의의 설명을 따르면 주의를 주지않아도 인지하게 되는 요소라고 한다. 처음들을 때는 뭔말일까 했는데, 단어만 어렵지 내용은 별로 어렵지않았다. <br>
그냥 쉽게 말하자면 데이터 표현시 일부 값들만 눈에 띄도록 표현(색, 크기등으로)해서 잘 보이도록 만들어주는 것이다. <br>
하지만 너무 많은 channel들을 사용해 표현하게 되면 오히려 더 인지하기 힘들어지고 복잡해지는 역효과를 만들어낼 수도 있다.<br>
그리고 데이터 시각화에는 근본적 원칙이 몇가지 있는데, 그 중 하나가 **Principle of Proportion Ink**라고 하는 잉크양 비례 법칙이다. 이는 실제 값과 그래픽으로 표현할 때 사용되는 잉크의 양은 비례되어야 한다는 말이다. 더 큰 값은 크기, 길이 혹은 색을 진하게 하거나 하는 식으로 비례하게 해주어야한다. <br><br>



## Matplotlib
**Matplotlib**은 Python에서 사용가능한 데이터 시각화 라이브러리이다. 현재 머신러닝/딥러닝 코딩은 Python을 이용해 이루어져 호환성이 좋은 Matplotlib을 많이 사용한다. <br>
Matplotlib에서는 막대그래프, 선그래프, 산점도 등 여러가지 시각화 방법론을 제공한다. <br><br>

#### 기본 사용법

```python
# matplotlib libray와 pyplot module 불러옴
import matplotlib as mpl
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,7))    # figsize(inch단위) 크기의 figure 할당
ax = fig.add_subplot()  # subplot추가. subplot : 그래프 그리는 공간 1개
plt.show()  # figure에 추가된 subplot 출력
```

각 subplot의 그래프에는 색, 선의 형태등 여러가지 channel 요소를 적용시킬 수 있다. <br>
> + ax.plot(color) : 그래프별 색 지정
> + ax.plot(label) : 그래프별 라벨 지정 (legend로 출력)
> + ax.set_title() : title 지정 
> + ax.set_xticks() : 해당 축에 적히는 수 위치 지정(yticks도 있음)
> + ax.set_xticklabels() : 축에 적히는 텍스트 수정
> + ax.text() / ax.annotate() : subplot에 text 추가

<br><br>

### Bar Plot
**Bar Plot**은 우리가 흔히 아는 막대 그래프를 말한다. Bar plot은 주로 카테고리에 따른 수치 값 비교에 자주 사용된다. <br>
Bar plot의 기본형은 막대가 수직으로 있는 형태이나 카테고리가 많을 경우 수평하게(Horizontal) 그래프를 그려준다(막대 눕힘). 

```python
ax1.bar(x, y)   # Vertical
ax2.barh(x,y)   # Horizontal
```

<br>

#### Bar plot 종류
여러 그룹에 대한 bar plot을 그리는 것을 Multiple Bar Plot이라 한다. Multiple bar plot에서는 그룹이 너무 많아지면 가독성이 떨어질 수 있어 5~7개 정도일 때 가장 효과적으로 사용할 수 있다.<br>
**Stacked Bar Plot** <br>
2개 이상 그룹 순서 유지하며 쌓아서 표현. <br>
가장 아래의 bar 분포 제외하고는 파악 힘듦<br>
응용으로 전체의 비율 나타내는 **Percentage Bar Chart** 있다.<br>
**Overlapped Bar Plot**
2개 이상 그룹 겹쳐서 표현<br>
3개 넘어가면 색 겹치기 때문에 파악 어려워짐<br>
```alpha```값으로 투명도 조절해서 그림<br>
**Grouped Bar Plot**
그룹별 범주에 따라 bar 이웃되게 배치<br>
Matplotlib으로는 구현 어렵다(Seaborn사용하기)<br><br>

+ 시각화 시에 데이터 종류에 따라 정렬해서 시각화(더 정확한 정보)
+ 여백, 공간 조절해서 가독성 높히기
    + set_xlim()
    + .spines[spine].set_visible()
    + width
    + .legend()
    + .margins()
+ 필요없는 복잡함은 없애기
    + .grid()
    + .set_ticklabels() (major/minor)
    + .text() / .annotate()
+ errorbar로 uncertainty정보 추가
<br><br>

### Line Plot
연속적으로 변하는 값 순서대로 점으로 나타낸것(선으로 연결)<br>
시계열 데이터의 분석에 특화됨<br>
.plot()으로 그림<br>
선 사용은 5개 이하로(넘어가면 가독성 떨어진다)<br>
선 구별 요소 : color, marker(marker, markersize), line(line style, line width)<br>
수시로 변하는 데이터는 Noise많아 smoothing으로 인지 방해 줄인다.<br>
그래프 그릴 때에는 xtick, ytick 간격을 규칙적으로해서 오해를 없애야 한다.<br>
legend로 각 그래프 label 나타내는 것보다는 선 끝에 label 추가하는 것이 더 보기 좋다. <br>
Min, max, mean 등의 정보는 annotation을 이용해서 점 추가해두면 분석에 도움된다. <br><br>

### Scatter Plot
각 데이터를 점으로 표현하는 그래프. 점으로 데이터의 두 feature간 관계 알고자할 때 사용한다. 보통 scatter plot은 데이터들의 군집, 값 사이의 차이, outlier 파악을 위해 사용된다. Scatter plot에서는 인과 관계와 상관 관계를 헷갈리기 쉬움에 주의해야한다. 만약 인과 관계를 나타내려면 반드시 사전정보를 가정으로 제시해야 한다.<br>
```.scatter()```로 사용<br>
점의 색, 모양, 크기를 변형시키며 데이터 그룹간 구분 준다. 단, 너무 많은 그룹 보이려하면 구분 힘들어질 수도 있다. <br>
또한 모양만 변형시키면 구분이 힘들어져 색도 함께 변화시키는 것이 좋다.<br>
Scatter plot은 grid와 꽤 상성이 좋지않아 굳이 같이 사용하지 않는다. 꼭 써야된다면 무채색의 grid를 사용하는 것이 좋다. <br>
Scatter plot에서 점이 너무 많아지면 분포 파악이 힘듦 <br>
+ 투명도 조정
+ Jittering : 점 위치 변경
+ 2D histogram : heatmap으로 시각화
+ Contour plot : 등고선으로 분포 표현
<br>위와 같은 방식으로 점의 분포 파악을 도울 수 있다.
<br><br>

### Text
Text를 사용하면 그래픽만으로 줄 수 없는 설명들을 많이 추가할 수 있고, 그림만 썼을 때 생길 수 있는 오해를 막을 수 있다. 그렇다고 너무 많이 사용하면 가독성 떨어지는 역효과 불러올 수 있다.<br>
Text는 .text()나 .annotate()로 원하는 위치에 추가하거나 title, label, tick label, legend등 여러 위치에 설정해 줄 수 있다. <br>
```.annotate()``` 사용 시에는 기리키고자 하는 위치에 화살표를 추가하고 축에 평행한 선을 그려 강조할 수 있다. <br>
+ .text(bbox) : text에 테두리 만들어 사용 가능
+ .text(va)/.text(ha) : text 세로, 가로 정렬 가능
+ .text(rotation) : 회전 각 조절 가능
<br>
이 외에도 color, linespacing, alpha, zorder등으로 text의 여러가지 property 설정 가능하다.<br><br>

### Color
데이터 시각화에서 가장 큰 부분을 차지한다고 할 수 있다. 사람이 볼 때 가장 구별하기 쉽고 눈에 잘들어오는 것이 색이다. 위치나 모양 같은 경우에는 색에 비하면 좀 떨어지긴한다. 하지만 또 그렇다고 색을 너무 휘황찬란하게 쓰면 오히려 더 보기 힘들어질 수도 있어서 특정 값을 강조하거나, 그룹을 나누어 구분짓는 등 목적에 맞게 적절한 색을 사용해야 효과가 극대화 된다. <br>
또한 색은 보편적으로 사용되는 용도에 맞게 사용해주는 것이 좋다. 예를 들면 높은 온도는 빨간색, 낮은 온도는 파란색과 같이 기존 정보의 이미지와 맞춰 사용해줘야 인지에 도움을 준다. (색깔 맞춰쓰는데는 이유가 있다)<br>
색의 사용에 있어서는 사람들이 미리 만들어둔 color palette를 사용하는 것이 좋다. <br>
+ 범주형 : 카테고리형 변수에 사용, 최대 10개까지, 색의 차이로 구분지어짐
+ 연속형 : 순서, 연속형 변수에 적합. 단일 색조의 연속적인 변화로 값 표현
+ 발산형 : 연속형과 유사하나 중안 기준으로 발산. 상반된 값이나 서로다른 2개 표현에 적합<br>

색을 이용해서 데이터를 강조하려면 명도, 색상, 채도, 보색 대비를 이용해서 highlighiting해 보인다. <br><br>

Matplotlib과 같은 라이브러리에서 색을 사용할 때에는 RGB로 색을 보는 것보다는 HSL(Hue, Saturate, Lightness)로 색을 보는 것이 좋다. <br>
Color palette는 R color palette와 같이 잘 만들어진 colormap을 쓰는 것을 추천한다. <br><br>

### Facet
화면 분할을 의미한다. 사실상 subplot 여러개 그리는 것이라고 보면된다. <br>
화면 분할을 통해 같은 방식으로 여러개의 feature를 보거나 각각의 feature들간 관계등 데이터의 여러 부분집합을 쉽게 보기 위해 사용한다. Matplotlib에서 사용하는 방법에는 여러가지가 있다.<br>
+ plt.subplot()
+ plt.figure() + fig.add_subplot()
+ plt.subplots()
+ fig.add_grid_spec() + array slicing
+ ax.inset_axes() : plot 내부에 미니맵처럼 그림. 메인 시각화 해치지않는 선에서만 사용하기
+ make_axes_locatable(ax) : subplot의 옆에 추가
<br>

또한 subplot들의 parameter를 조정해서 시각화 효과를 더 줄 수 있다.<br>
+ figuresize : plot 크기 조절
+ dpi : 해상도 조절
+ sharex, sharey : x, y축 공유
+ squeeze : False 설정시 항상 2차원 배열로 받을 수 있음
+ aspect : 가로, 세로 비율 조절

<br><br>

### Grid
기본적인 grid는 바둑판 모양의 격자로 정보를 보조적으로 제공한다. color, zorder, major/minor, axis와 같은 parameter로 grid의 출력을 변경할 수 있다. <br>
그리고 grid의 모양 자체를 바꿀 수도 있다. <br>
+ x+y=c : 두 변수의 합이 중요할 경우
+ y=cx : 비율이 중요할 경우
+ xy=c : 두 변수 곱이 중요할 경우
+ $ (x-x')^2 + (y-y')^2=c $ : 특정 데이터 중심으로 보고싶을 경우
<br>

<p align="center">
  <img src="/assets/img/boostcamp/DataViz/grid_example_img.png">
</p>

<br><br>

### Line & Span
Annotation과 함께 축에 평행한 선을 사용하면 plot의 상한, 하한, 평균과 같은 여러가지 정보를 보기 쉽게 표현할 수 있다. 또한 특정 면적을 서로 다른 색으로 칠하여 그룹에 대한 표시를 보다 쉽게 나타낼 수도 있다. <br><br>

### Theme
Matplotlib에는 여러가지 테마를 설정하거나 기본 테마를 커스터마이징하여서 좀 더 보기좋게 시각화할 수 있다. (ggplot, fivethirtyeight 등 여러가지 있음)<br><br>

<br>

## 3주차를 마치며
사실 나는 데이터 시각화에 대해서는 아는 것도 없었고 관심도 없었다. 논문 읽을 때 한번씩 보이던 시각화된 데이터들도 어떻게 만들어졌는지 관심이 없었다. 이번주 강의를 듣고 생각보다 시각화 방법의 종류가 많음에 꽤 놀랐다. 이제껏 NLP 논문에서 주로 보이던 시각화된 데이터들은 Scatter plot, line graph, heatmap정도 밖에 기억이 안나 종류가 이렇게 다양할 줄은 몰랐다. 슬슬 준비하기 시작한 대회참가나 얼마 뒤에 있을 부스트캠프 내부 대회에서 아마 이 시각화 강의가 꽤 용이하게 사용될 듯하다. 데이터의 분포나 편향, 학습 결과등을 표현하고자 할 때 많이 쓰이지 싶다. 
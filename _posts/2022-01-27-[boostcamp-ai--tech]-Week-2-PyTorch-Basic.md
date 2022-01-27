---
layout : post
title : '[boostcamp ai-tech] Week 2 PyTorch Basic'
categories : [boostcamp]
---

---

이제 부스트캠프 2주차에 접어들었다. 1주차에 비해 강의양이 줄어 1주차보다는 체감상 좀 더 수월하게 강의를 들을 수 있었다. 
이번 주 강의는 **PyTorch**에 관한 내용이였다. Python은 코딩테스트를 준비하느라 조금 했었는데, PyTorch는 많이 접해보지않아 꽤 유익한 강의가 되었다.
사실상 코드를 전부 다 정리하고 쓰는 것보다는 공부하면서 생소했던 부분이나 쓰기 어려웠던 부분들 위주로 정리하려한다.
<br><br>

# PyTorch Basic
---

## PyTorch 장점
딥러닝 코드를 처음부터 전부 다 짜는 건 꽤 고된 일이다. 그래서 요즘 거의 모든 딥러닝 코드들은 딥러닝 프레임워크를 이용해서 만든다. 프레임워크 종류는 여러가지가 있는데 그중에서 가장 많이 쓰이고 유명한 것은 PyTorch와 TensorFlow이다. 그 중 나는 파이토치를 공부하게 되었고, 사실 이전부터 텐서플로보다는 파이토치를 공부하고 싶긴했다. 왜냐면 날이 갈수록 파이토치 사용이 텐서플로보다 많아지는 추세이고, 여러 장점들이 존재하기 때문이다.<br><br>

**Dynamic Computation Graph(DCG)** <br>
간단하게 말하면, 매 반복(iteration)마다 computational graph를 새로 정의해서 실행하는 것이다. 다른 말로는 Define by Run이라고도 한다. <br>
이 방식의 장점은 코드가 깔끔하고 쉬워져 가독성이 높아진다는 것이다(Pythonic). 그리고 GPU 관련 지원, API와 Community가 잘 되어있어 많이들 쓴다. 사실 쓰기 편해서 쓰는게 주 이유라고 보면된다. <br><br>

## PyTorch 구조
딥러닝 논문 모델의 구조를 보게되면 모델들은 많은 layer들을 쌓아올려 만든 것임을 알 수 있다. 각 layer들은 PyTorch나 다른 프레임워크에 잘 구현되어 있어 쌓아올리기만 잘하면된다. <br>
### torch.nn.Module
딥러닝 모델을 구성하는 layer들의 base가 되는 class이다. Input, output, backward(autograd 이용)가 정의되어 있으며 학습의 대상이 되는 parameter들을 정의한다. <br><br>

### torch.nn.Parameter
Tensor의 subclass이다. Module의 parameter들을 ```nn.Parameter```로 저장하며 모듈내에서 ```requires_grad=True```로 지정되어 학습대상이 된다. <br>
모델을 만들 때 직접 지정할 일은 없다고 보면 된다. PyTorch에서 기본적으로 제공해주는 layer들이 알아서 weight나 bias등을 지정하기 때문에 그냥 맞춰서 쓰면된다. <br><br>

### backward
각 Layer에 있는 parameter들의 미분을 수행한다. model prediction과 label간의 차이(loss)에 대해 미분하여, 그 값으로 ```optimizer.step()```에서 parameter를 업데이트한다. <br><br>

## PyTorch Dataset
딥러닝 모델에 데이터를 넣어주려면 다음과 같은 단계를 거치게 된다.\
1. Data Collecting, Cleaning, Pre-processing
2. Data augmentation, ToTensor 등 Data transform
3. Dataset class로 변환 (```__getitem__()```으로 데이터 불러올 때 반환 방식 정의, map-style이라함)
4. DataLoader로 dataset의 data 묶어서 모델에다가 data feeding (batch, shuffle등 추가)

<br><br>

### Dataset class
Dataset class는 data의 입력 형태를 정의해주는 class이다. 이를 통해 데이터 입력 방식을 표준화시켜 모든 데이터에 적용한다. 이미지, 텍스트, 오디오 등 파일 형태에 따라서 입력을 다르게 정의해준다. 사용할 때 주의할 점이 있는데,
1. 각 데이터 형태별 특성에 맞게 함수를 다르게 정의해주어야 한다.
2. 데이터 생성 시점에 모든 처리를 할 필요는 없다. (image->Tensor는 학습 필요시점에 해도됨)

<br><br>

### DataLoader class
Dataset class로 data의 입력을 정의해주었으면 DataLoader를 이용해 data batch를 만들어준다(generator). 정의된 데이터 입력방식에 따라 data를 batch로 묶어 모델에 입력해 준다. 학습 시 GPU에 데이터가 올라가기 전 마지막 데이터 변환을 수행하며 메인 역할은 Tensor변환과 batch 처리이다. DataLoader는 iterable한 객체로 next(iter(DataLoader)) 같은 방식으로 데이터 추출이 가능하다. <br>
DataLoader에는 collate_fn이라는 데이터셋의 sample된 list를 batch단위로 바꾸기 위해서 필요한 기능이다. 원래 데이터-라벨로 묶인 데이터셋의 각 sample들을 데이터는 데이터끼리, 라벨은 라벨끼리 묶어주는 것으로 이해하고 있다. <br>
이는 가변길이 데이터에 zero-padding과 같이 data의 크기를 맞춰주기 위해 주로 사용된다고 한다. <br><br>


## Model 불러오기
최근 딥러닝은 백본 모델을 불러와 데이터에 맞게 다시 학습하는 **pre-training + fine-tuning** 기법(Transfer learning)이 발달되어있다. 이는 가진 데이터가 부족하거나 성능의 향상을 위해 잘 사용되기도 한다. 학습된 백본 모델을 불러오기 때문에 학습 결과를 저장해두는 것이 중요한데, 이는 백본 모델 사용 시에도 중요할 뿐만 아니라 학습이 오래걸리는 특성상 학습 결과를 중간중간 저장하여 학습 내용이 날아가는 불상사를 방지하기 위해서라도 반드시 필요하게 된다. <br><br>

### 학습 결과 저장
학습 결과는 model.save()를 이용해 모델의 architecture와 parameter를 함께 저장할 수 있다. 또는 state_dict로 저장된 모델의 parameters만을 저장하는 방식도 존재한다. 저장시에는 .pt 파일(pickle)로 저장하고 불러오면 된다. 보통은 모델 architecture없이 parameter로 저장 및 공유를 한다. <br><br>

### Transfer learning
Transfer learning은 다른 데이터셋으로 학습시켜 놓은 모델에다가 현재 데이터를 이용해 한번 더 학습시키는 방법이다. 이 때 하고자하는 task에 맞춰서 모델 일부를 변경하거나 몇개의 layer를 추가하기도 한다. <br>
이는 보통 딥러닝 모델의 성능은 사용된 데이터셋의 크기가 좌우하기에 대용량 데이터셋을 학습시킬 여건이 안되는 상황이나 시간이 안되는 상황에서 사용하기 좋으며 성능 또한 보장할 수 있다. 여러 장점 덕분에 현재 딥러닝에서 가장 일반적으로 사용되는 학습방법이 되었다. <br>
NLP에서는 HuggingFace가 이러한 transfer learning에 사용되는 백본 모델들의 표준이라고 볼 수 있다. <br>
미리 학습시켜 놓은 모델의 파라미터는 사용자가 가진 데이터셋으로 재학습할 때 일부를 freezing해주어서 학습을 막고 새로 추가하는 layer나 일부분만을 재학습시켜서 사용하게 된다. <br><br>


## GPU
최근 딥러닝 모델들은 GPU를 사용해 학습을 진행한다. CPU만을 사용해 학습을 진행하게 되면 엄청나게 긴 시간동안 학습을 해야하기에 시간 단축을 위해 GPU를 사용하곤 한다. <br><br>

### Multi-GPU
돈만 많다면 여러대의 GPU와 컴퓨터를 사용해 학습하는 것이 좋다. 이를 Multi-GPU 학습이라고 하며, Multi-GPU 학습을 Single Node Multi GPU, Multi Node Multi GPU 둘로 나눌 수 있다. 각각은 한 대의 컴퓨터(Single Node)에 여러 개의 GPU(Multi GPU), 여러 대의 컴퓨터(Multi Node)에 여러 개의 GPU(Multi GPU)를 사용하는 방식을 말한다. <br>
Multi GPU 학습 방법은 모델을 병렬화하거나 데이터를 병렬화하여 진행한다. 모델 병렬화는 예전 AlexNet부터 사용되었으나 모델 병목현상이나 파이프라인의 어려움등으로 인해서 꽤나 까다로운 연구분야로 자리잡혔다. 반면에 데이터 병렬화 같은 경우는 비교적 단순한 편인데, 이는 데이터를 나누어 GPU에 할당하고, 모델에 나누어진 데이터를 병렬적으로 입력하여 loss나 accuracy 등의 결과값을 평균내어 모델의 성능으로 사용하여 최적화하는 방식이다. <br>
PyTorch에서는 **DataParallel**과 **DistributedDataParallel** 두 가지 방식을 제공한다. <br>
DataParallel 방식은 단순히 데이터를 분배해서 GPU에 할당한 후 평균을 내는 방식인데, 이 때 한 GPU가 데이터 분배, 취합 및 여러가지 일을 맡아 하게 되어 GPU 사용의 불균형을 초래할 수 있게 된다. 이는 GPU의 병목현상을 발생시킬 수 있으며 data의 batch size를 줄일 수 밖에 없게 만든다. <br>
DistributedDataParallel은 각 CPU마다 process를 생성해서 개별 GPU에 할당시켜 연산을 평균내는 방식이다. 기본적으로 DataParallel과 유사한 방식이지만 DataParallel에서 발생할 수 있는 GPU 병목현상을 방지해줄 수 있다. <br><br>


## HyperParameter Tuning
모델에는 학습을 통해 값이 조정되는 parameter와 직접 지정하게 되는 Learning rate, Batch size, Epoch수와 같은 Hyperparameter가 있다. Hyperparameter를 어떻게 설정하느냐에 따라서 모델의 성능이 달라지게 되는데 사실 최근 모델에서는 다른 요소들에 비해 성능 향상폭이 적은 편이라 당장은 심각하게 많은 시간을 쏟아가며 고뇌할 필요까지는 없어보인다(그렇다고 안중요하지는 않고). <br>
Hyperparameter를 찾는 방법에는 **grid search**와 **random search** 두 가지가 있는데, grid search는 hyperparameter를 일정한 간격으로 변경시켜가며 튜닝해가는 방식이다. Random search는 초반에는 hyperparameter를 무작위로 변경시켜가며 튜닝해나가다가 성능이 좋아지는 범위를 찾게되면 그 이후부터는 해당 범위에서 grid search를 해나가며 튜닝해나가게 된다(옛날 방식). <br>
추가로 요즘에는 베이지안 튜닝 기법이 인기를 얻고 있다고 한다. <br><br>

### Ray
Multi node Multi processing을 지원하는 모듈이다. 딥러닝 병렬 처리를 위해서 개발된 모듈이며, 현재는 분산병렬 ML/DL 모듈의 표준이 되었다. 뿐만아니라 hyperparameter search를 위한 다양한 모듈을 제공하여 hyperparameter tuning시에도 사용된다. <br><br>

## TroubleShooting
딥러닝 모델을 학습시키다보면 여러가지 에러가 발생하게 되는데, 주로 OOM(Out Of Memory) 문제가 많다. OOM 문제는 발생 원인이나 발생한 위치를 찾기가 어려운데 이는 이전의 메모리 상태를 파악하기가 어려워 정확한 원인은 분석하기가 쉽지않다. 그래서 보통 이런 문제가 생기게 되면 iteration 부분에서 발생하게 되어, Batch size를 줄이고 GPU clean을 실행한다. 코드가 제대로 작성되었다면 이 절차를 거치면 문제없이 잘 작동하며 만약 해결되지않는다면 코드를 다시 검토해봐야한다. <br><br>

### GPUtil
nvidia-smi와 같이 GPU의 상태를 보여줄 수 있는 모듈이다. nvidia-smi와는 다르게 실시간 GPU 상태를 모니터링할 수 있으며, colab에서 모니터링이 편하다는 장점이 있다. <br><br>

### torch.cuda.empty_cache()
학습을 하다보면 메모리 사용량이 누적되는 경우가 있다. 이는 변수를 del을 이용해 free해주더라도 해당 메모리 주소의 메모리는 <br>
여전히 사용되고 있어 발생하는 문제이다. 그래서 메모리 공간을 확보하려면 사용해 주는 것이 좋으며 loop가 시작하기 전에 한번 <br>
메모리를 비워주고 가는 것을 추천한다. <br><br>

### 그 외
이외에도 에러 발생을 방지하기 위한 여러가지 팁이 있는데, 
1. 1D tensor는 python 기본 객체(int, float등)로 변환해서 처리해주는 것이 좋다. Tensor는 GPU상의 메모리를 쓰기에 GPU 메모리를 <br> 절약하려면 python 기본 객체로 바꿔주자. <br>
2. 더 이상 필요없는 변수는 del 명령어로 삭제하자.
3. 가능한 batch size들을 실험해보자
4. Inference 시점에는 gradient가 필요없게 되니 with torch.no_grad()를 사용하자.
5. CUDNN_STATUS_NOT_INIT : 보통 GPU를 잘못 설치했거나 제대로 설치하지 않은 경우 발생한다.
6. device-side-assert : OOM의 일종으로 생각하면 된다.
7. colab에서는 LSTM같은 메모리 많이 차지하는 layer나 너무 큰 사이즈는 실행하지말자.
8. CNN에서 발생하는 에러는 대부분 크기가 안맞아서 발생하니 torch.summary로 크기를 맞춰 실행하면 된다. 특히 CNN 마지막단의 fc layer를 조심



<br><br><br>

# PyTorch Functions
---
### torch.gather(input,dim, index, *, sparse_grad=False, out=None) -> Tensor

PyTorch 공식 문서의 설명은 다음과 같다.<br>
*Gathers values along an axis specified by dim.*<br>
dim으로 입력된 axis를 따라서 값을 모은다는 말로 해석되는데 사실 이것만봐서는 이해가 잘 안되었다. 3D 텐서에서의 예시는 다음과 같이 나타난다고 한다.<br>

```python
out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
```

여기서 제한사항이 있는데, input과 index는 같은 dimension을 가져야된다. <br>
나는 공식 문서의 설명으로는 제대로 이해가 되지 않아서 직접 그려가며 이해하려고 했다. 그 결과 이해한 것이 우선 Tensor의 indexing을 보면 ```Tensor[i,j,k]```와 같이 사용하게 된다. 여기서 i,j,k 순서대로 각각 axis 0,1,2를 나타낸다고 하자. 그리고 gather 함수의 dim은 axis를 지정한다고 했는데 해당 axis의 index를 gather에 들어오는 index로 교체해준다고 이해했다. <br>
예를 들자면 아래는 텐서에서 대각선에 해당하는 값들, ```1, 4, 5, 8```만을 뽑아내는 함수다. <br>

```python
T = torch.Tensor([[[1, 2],
                   [3, 4]],
                  [[5, 6],
                   [7, 8]]])
diag = torch.gather(T, 1, [[[0, 1]], [[0, 1]]])
```

T 텐서에서 각 대각요소의 index는 순서대로 ```[0,0,0], [0,1,1], [1,0,0], [1,1,1]```이다. 그리고 dim으로 들어온 1에 해당하는 axis는 두번째 위치, index로 입력된 텐서들의 index는 각각 ```[0,0,0]->0 / [0,0,1]->1 / [1,0,0]->0 / [1,0,1]->1```이다. <br>
여기서 gather함수가 실행되면, index 텐서의 index값 중에 dim axis 위치의 축을 해당 index의 값으로 변경한다. <br>
```python

[0,0,0] --> [0,0,0]  # 두번째 값 0에서 0으로 변경
[0,0,1] --> [0,1,1]  # 두번째 값 0에서 1로 변경
[1,0,0] --> [1,0,0]  # 두번째 값 0에서 0으로 변경
[1,0,1] --> [1,1,1]  # 두번째 값 0에서 1로 변경
```
위와 같이 변경된다. <br>
여기까지가 내가 이해한 gather의 동작방식이다. 이게 아닐 수도 있지만 우선은... 나는 이렇게 이해하는게 잘 이해된 것 같았다. <br><br>

### torch.scatter(input, dim, index, src) -> Tensor
함수를 풀어서 말하면, dim axis에 따라서 index에 해당하는 src Tensor 값을 input Tensor에 scatter하는 것이다. <br>

```python
SRC = torch.Tensor([[[ 1.,  2.,  3.],
                     [ 4.,  5.,  6.],
                     [ 7.,  8.,  9.]],
                    [[ 0., -1., -2.],
                     [-3., -4., -5.],
                     [-6., -7., -8.]]])
INPUT = torch.zeros(2,3,5)
torch.scatter(INPUT, 1, torch.tensor([[[0,1,2],[1,2,0]],[[0,1,2],[1,2,0]]]), SRC)
'''
tensor([[[ 1.,  0.,  6.,  0.,  0.],
         [ 4.,  2.,  0.,  0.,  0.],
         [ 0.,  5.,  3.,  0.,  0.]],

        [[ 0.,  0., -5.,  0.,  0.],
         [-3., -1.,  0.,  0.,  0.],
         [ 0., -4., -2.,  0.,  0.]]])
'''
```

torch.scatter()도 gather처럼 조건이 있다. scatter는 input, index, src가 dimension이 같아야한다. 그리고 index Tensor의 원소 값은 src Tensor의 각 dim axis 보다 작아야한다. <br>
scatter도 내가 이해한대로 말해보자면, 우선 index Tensor의 각 index별 값을 ```index[i,j,k] = V```이고, ```dim=1```이라 하자. 그렇게 되면 SRC Tensor의 원소 순서대로 ```index[i,V,k]``` 위치에 하나씩 scatter해준다. 순서대로 모두 적용한 결과가 위 코드의 tensor다. 
scatter() 함수는 one-hot encoding에서 유용하게 쓰인다고한다. 1을 각각의 원하는 index 위치에 넣을 수 있으니 잘 쓰일 것 같다. <br><br>

### torch.nn.Linear()  &  torch.nn.LazyLinear()
nn.Linear()와 nn.LazyLinear()는 둘 다 선형 변환 계층으로 하는 동작은 똑같다. 하지만 입력받는 parameter가 다르다. <br>
```python
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
torch.nn.LazyLinear(out_features, bias=True, device=None, dtype=None)
```
넘겨받는 parameter를 보면 LazyLinear에서는 ```in_features```가 없다. 이는 두 함수가 약간 다르기 때문인데, nn.Linear는 in_features가 정해져있어서 입력되는 텐서가 N x in_features 형태로 들어와야한다. 그에 반해 LazyLinear는 in_features가 따로 정해져있지 않아서 입력 텐서가 N x M (이 때 M은 아무 값이나 상관없음) 형태로 들어온다. 둘 다 출력되는 텐서는 N x out_features로 동일하긴하다. <br><br>

### Python List & nn.ModuleList
PyTorch에는 모델에서 사용되는 모듈을 list 형태로 순서대로 모아 indexing 기능을 제공해주는 ModuleList라는 것이 있다(dict 형태로 모아주는 ModuleDict도 있긴하다). 물론 Python에도 list가 존재하긴한다. 하지만 submodule들의 parameter들을 model.parameters()같은 함수를 통해 전달받으려면 python list로는 불가능하게 된다. 그래서 submodule들의 parameter를 저장해주는 ModuleList를 사용해야한다. <br>
[Difference in usage between nn.ModuleList and python list](https://discuss.pytorch.org/t/the-difference-in-usage-between-nn-modulelist-and-python-list/7744) <br><br>

### Tensor & nn.Parameter
PyTorch에는 Parameter라는 클래스가 있다. Parameter 클래스는 결국 tensor와 하는 역할이 같다고 볼 수도 있는데 왜 굳이 Parameter라는 클래스를 따로 만들어 사용할까? 모델의 파라미터들을 nn.Parameter가 아닌 텐서로 만들어준다면 모델 최적화시에 문제가 생긴다. 텐서로 만들어진 모델의 파라미터들은 model.parameters()에 등록이 되지않아 torch.optim을 이용한 최적화 단계에서 최적화할 parameter를 찾을 수 없게되어 최적화가 진행되지않게 된다. 그래서 nn.Parameter class로 모델의 parameter들을 등록하여 최적화 시 적절하게 역전파가 일어날 수 있게 해준다. <br>
[Difference between Parameter vs. Tensor in PyTorch](https://stackoverflow.com/questions/56708367/difference-between-parameter-vs-tensor-in-pytorch)
<br><br>

### buffers
사실 buffer도 parameter처럼 텐서와 같다고 볼 수 있다. 하지만 이 또한 굳이 사용하는 이유는 state_dict로의 등록 여부에 관련이 있다. buffer는 nn.Parameter처럼 gradient를 갖진 않지만 일부 layer들에서는 모델에 저장된 값들을 불러와 계산에 사용하여야 하기 때문에 buffer로 굳이 등록해주어야할 필요가 있다. <br>
[What PyTorch means by buffers?](https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266)
<br><br>

### hook
Hook은 패키지화된 코드에 custom 코드를 중간에 실행시킬 수 있도록 만들어둔 인터페이스이다. 주로 프로그램 실행 로직 분석이나 추가 기능 제공을 위해 만들어 사용한다. PyTorch의 hook은 Tensor에 적용시키거나 Module에 적용시키는 hook 두가지로 구분지을 수 있다. ```register_hook()```함수를 이용해 모델에 hook을 넣을 수 있으며, ```forward_hook(), forward_pre_hook(), backward_hook``` 총 세가지가 있다. 여기서 ```backward_hook```만 Tensor에 적용할 수 있으며 Module에는 세가지 모두 적용할 수 있다. 각각의 hook을 모델에 등록하는 방법은 다음과 같다. <br>

```python
register_forward_hook(hook)  # hook(module, input, output) -> None or modifed output
register_forward_pre_hook(hook)  # hook(module, input) -> None or modified input
register_full_backward_hook(hook)  # hook(module, grad_input, grad_output) -> tuple(Tensor) or None

# For Tensor backward hook
register_hook(hook)  # hook(grad) -> Tensor or None
```

backward_hook 등록 함수는 원래 ```register_backward_hook()```도 존재했는데 더이상 사용되지않고 ```register_full_backward_hook()```으로만 사용된다. <br><br>


### apply
```model.apply(fn)```는 입력된 함수를 모델의 모든 submodule에 recursive하게 적용시켜주는 함수다. 여기서 fn 함수는 입력을 각 submodule로 받아 각각의 submodule에 함수를 적용시켜준다. 일반적으로 apply 함수는 모델의 가중치 초기화에 주로 사용된다. <br>
apply 함수를 사용하여 모델의 repr 출력을 수정할 수 있다. (apply에 ```model.extra_repr```수정 코드 넣어 사용)
<br><br>

### collate_fn
collate_fn은 앞서 설명했듯이 data의 길이를 맞춰주는 데에 주로 사용된다. 적용 방식은 DataLoader에 설정된 batch_size 크기만큼의 샘플들에 collate_fn으로 설정된 함수를 적용시킨 후 batch단위로 바꾸어 출력하는 형태이다. <br><br>


### drop_last
batch단위로 데이터를 불러오면 총 sample 수가 batch size에 나누어 떨어지지 않는다면 마지막 batch는 다른 batch들에 비해 크기가 작게 된다. 이 때 마지막 batch를 사용하지 않도록 해주는 parameter이다. 쉽게 말하면 batch 단위로 data를 묶고 마지막 꼬다리를 버리는 것이라 보면 된다. <br><br><br>


### 이번주를 마치며...
예전에 PyTorch를 공부하려 공식문서에서 제공되는 예제 코드를 보며 공부한 적이 있었는데, 이번 주 강의를 들으면서 새롭게 알게된 내용과 함수가 꽤 많았다. 특히 생각해보지도 않았던 Tensor와 nn.Parameter의 차이점, python list와 nn.ModuleList의 차이점에 대해 생각해볼 수 있었던 시간이였고, 실제 딥러닝 모델을 만들 때 유용하게 쓰일 수 있는 함수들에 대해 알게 되었다. 그리고 모델 생성, 학습에 관해서는 어느정도는 알고 있다고 생각했는데, 데이터를 가져와 모델에 넣기위한 형태로 바꾸는 단계와 transfer learning의 방법, data padding 방식과 같은 내용들은 상당히 생소했고, 제대로 공부하지않았던 예전의 내가 조금은 부끄러워졌기도 했다. <br>
앞으로 남은 기간 여러 프로젝트와 대회도 진행해가며 차근차근 실력을 쌓아 쓸만한 NLP 엔지니어가 되고 싶다. 
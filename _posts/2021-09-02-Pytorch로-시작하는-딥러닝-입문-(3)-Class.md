---
layout: post
title: PyTorch로 시작하는 딥러닝 입문 (3) Class
categories: [PyTorch]
---

이번 포스팅에서는 class에 대해서 알아볼 것이다. PyTorch에서 구현한 모델들의 대부분은 class로 구현되어 있다. <br>
딥러닝 모델 뿐만 아니라 다양한 기능들 또한 대부분 class로 구현되어 있다. 그만큼 class는 python에서 상당히 중요한 부분이다.<br>

# 1. Class vs. Function

Class는 함수와 자주 비교되고는 하는데, 둘은 명백히 차이가 있다. 우선 함수와 클래스 모두 반복되는 특정 기능을 <br>
하나의 클래스 혹은 함수로 정의하여 불필요한 반복을 피하고 코드의 가독성을 높여준다. <br>
<br>
둘의 큰 차이는 클래스는 서로 영향을 주지 않는 독립적인 객체를 만들 수 있는 반면 함수는 그러지 못한다는 점이다. <br>
간단한 예시로 전역변수를 선언하고 덧셈 기능을 하는 함수와 클래스를 각각 만들어 보겠다.<br>
각 함수와 클래스는 두개의 독립적인 연산을 하도록 만들 것이다. <br>

```python
result1 = 0
result2 = 0

def add1(num):
    global result1
    result1 += num
    return result1

def add2(num):
    global result2
    result2 += num
    return result2

class Calcul:
    def __init__(self):
        self.result = 0
    
    def add(self, num):
        self.result += num
        return self.result
```

위 처럼 선언한 각각의 함수들과 클래스에 다른 두개의 연산을 수행한다면 아래와 같이 된다.

```python
print(add1(3))  # Result : 3
print(add1(4))  # Result : 7
print(add2(3))  # Result : 3
print(add2(7))  # Result : 10

cal1 = Calcul()  # Generate Calcul object
cal2 = Calcul()  # Generate Calcul object

print(cal1.add(3))  # Result : 3
print(cal1.add(4))  # Result : 7
print(cal2.add(3))  # Result : 3
print(cal2.add(7))  # Result : 10
```

보다시피 함수를 이용해서 개별적인 두개의 같은 연산을 수행하려면 함수가 두개 선언되어야 하지만 클래스를 활용한다면 <br>
클래스를 한번만 생성하면 Generator로 서로 영향을 끼치지 않는 객체를 여러개 만들 수 있다. <br>
이 때 Generator는 위 코드에서 ```= Calcul()``` 부분 처럼 객체를 생성하는 것을 말한다.<br><br><br>

# 2. Class

이제 클래스에 대해 좀 더 자세히 알아보려한다. [wikidocs](https://wikidocs.net/60034)에서는 자세한 설명이 생략되어 있어 <br>
다른 사이트들을 검색하며 좀 더 자세히 공부했다. <br><br>

## 1) Class 선언
---

Class의 선언은 함수와는 달리 앞에 ```def```를 붙이지 않아도 된다. ```class Clss_Name()```과 같이 생성하고, 그 후 내부에 <br>
클래스 내부 함수인 클래스 메소드를 생성한다. 메소드 뿐만 아니라 클래스 내부에는 변수 또한 생성이 가능하다. 

```python
class Class_Name():
    def __init__(self, var1, var2, var3):
        super().__init__()
        self.class_var1 = var1
        self.class_var2 = var2
        class_var3 = var3

    def class_method(params):
        ...
```

위의 예시에서 불 수 있듯이 클래스의 선언 시에는 ```__init__```을 통해 클래스를 초기화해주어야 한다. 클래스를 이용해 객체를 <br>
선언하게 되면 클래스 내부에서는 ```__init__``` 초기화 함수를 이용해서 객체가 사용하게 될 클래스 내의 메소드, 변수와 같은 속성들을 <br>
초기화 해준다. 클래스 메소드들은 객체를 생성한 후 각 객체끼리 서로 영향을 주지 않고 독립적으로 사용된다. <br>
그리고 ```__init__``` 초기화 함수를 보면 ```self.class_var```와 ```class_var3```을 볼 수가 있는데, 이는 각각 인스턴스 속성(instance attribute), <br>
클래스 속성(class attribute)이라고 부른다. 인스턴스 속성은 각 객체별로 따로 갖게 되지만 클래스 속성은 동일한 클래스로 선언된 <br>
객체들은 모두 동일한 속성을 갖게 된다.

```python
cls1 = Class_Name()  # Generate object cls1
cls2 = Class_Name()  # Generate object cls2

cls1.class_method(params)  # cls1만의 메소드
cls2.class_method(params)  # cls2만의 메소드 (cls1.class_method와 서로 영향 없음)
```

위와 같이 객체를 선언할 수 있으며 클래스 메소드는 각 객체별로 따로 동작한다. <br><br><br>

## 2) 상속 (Inheritance)
---

클래스에는 상속이라는 개념도 있는데, 이는 물려주는 클래스(Super class, Parent class)의 내용을 물려받는 클래스<br>
(Sub class, Child class)가 갖게 되는 것이다. 딥러닝 모델을 만들 때에는 보통은 PyTorch의 ```nn.Module``` 클래스를 상속받아 모델을 만든다. <br>
```nn.Module``` 클래스를 물려받으며 해당 클래스의 여러가지 유용한 기능들을 상속받을 수 있으며, <br>
그로 인해 좀 더 쉽게 딥러닝 모델을 만들 수 있게 된다. 상속받은 Sub class는 별도의 메소드 정의 없이 <br> 
Super class의 메소드를 사용할 수 있게 되며 속성 또한 사용할 수 있게 된다. <br>

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Conv2d(1, 20, 5)
        self.layer2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return F.relu(self.layer2(x))
```

위의 코드는 ```nn.Module``` 클래스를 상속받는 딥러닝 모델의 간단한 예시이다. 코드는 [PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=nn%20module#torch.nn.Module)에서 볼 수 있다. <br>
```nn.Module```을 상속받는 모델들은 항상 ```__init__```과 ```forward``` 메소드를 선언해 주어야한다. <br>
```nn.Module```을 상속받은 모델은 따로 ```Conv2d``` 메소드를 선언하지 않았음에도 Conv2d 메소드를 사용할 수 있게되고, <br>
다른 유용한 기능들 또한 사용 가능해진다.<br><br><br><br>

여기까지가 첫번째 챕터 파이토치 기초 부분에 관한 내용이며, 다음 장 부터는 선형 회귀 부분에 대해서 정리할 것이다.
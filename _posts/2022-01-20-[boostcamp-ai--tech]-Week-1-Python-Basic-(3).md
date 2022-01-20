---
layout: post
title: '[boostcamp ai-tech] Week1 Python Basic (3)'
categories: [boostcamp]
---

---

# Class
---
이번 강의는 Python의 **Class**와 **Module**, **Project**에 관련한 내용이였다. <br>
우선 Class에 들어가기 앞서, 객체 지향 프로그래밍에 대해서 간단하게 설명하는데, 객체 지향 프로그래밍이란 컴퓨터 프로그램을 여러 가지 객체들의 모임으로 파악하고, Attribute와 Action을 가진 각각의 객체들이 서로 작용하여 데이터를 처리하는 것이다. <br> 여기서 Attribute는 변수로, Action은 Method로 표현할 수 있다. Python 또한 대표적인 객체 지향 언어 중 하나이다. <br>
이러한 객체 지향 프로그램은 설계도로 볼 수 있는 **Class**와 실 구현체인 **Instance**로 나눌 수 있다.<br><br>

Python에서 class 구현은 다음과 같이 이루어진다. <br>

```python
class ClassName():  # () : 상속받는 객체명 넣어줌
    def __init__(self, variable1, variable2, variable3):  # __init__ : 객체 초기화 예약 함수(magic method)
        self.var1 = variable1
        self.var2 = variable2
        self.var3 = variable3

    def class_method1(self, arg1):
        ...
```
<br>
여기서 각 부분에 대해 좀 더 자세히 알아보면 <br>

> + \_\_init\_\_() : 생성자, class에 attribute를 추가해준다. Argument는 항상 self 들어가야됨.
> + clas_method() : 기존 함수와 같은 방식이다. 단, self를 함수 인자로 넘겨야 class 함수로 인정된다. 

<br>

Instance의 생성은 ``` instance_name = ClassName(value1, value2, value3) ``` 와 같이 instance의 이름을 선언하며 클래스 초기값을 <br>
입력하면 된다. <br><br>

**+Tip)** Python Naming Rule <br>
- 함수/변수 : snake_case, 띄어쓰기 부분에 _ 추가해줌
- class : CamelCase, 띄어쓰기 부분에 대문자
<br><br>


## Class의 특징
Class는 3가지 특징을 갖고 있다. **Inheritance, Polymorphism, Visibility**<br>
셋 다 중요한 개념이다. <br><br>

### Inheritance (상속)
> + 부모 클래스의 속성과 메소드를 자식클래스가 물려받는 것
>
> ```python
> class Person():
>   def __init__(self, name, age):
>     self.name = name
>     self.age = age
>
> class Korean(Person):
>   pass
>
> korean_1 = Korean("Gil-dong", 99)
> ```
>
> 위 코드는 Korean이라는 class가 Person class를 상속받게 된다.
> Korean class에 아무것도 선언하지 않더라도 korean_1 instance는 ```name```과 ```age```라는 ```attribute```를 가질 수 있다.

<br><br>

### Polymorphism (다형성)
> + 같은 이름을 가지는 method의 내부 로직을 다르게 작성하는 것
> + Dynamic Typing 특성에 의해 파이썬에서는 같은 부모 클래스의 상속에서 주로 발생
> + 중요한 개념이긴 하나 너무 깊게 알 필요는 없다...
>
> ```python
> class Shape():
>   ...
>   def draw():
>     ...
>
> class Circle(Shape):
>   ...
>   def draw():
>     ...
>
> class Square(Shape):
>   ...
>   def draw():
>     ...
> ```
> Square class의 instance를 생성하고 draw를 실행하면 Square class의 draw()가 실행된다.

<br><br>

### Visibility (가시성)
> + Encapsulation(캡슐화)이라고도 한다.
> + Object의 정보 접근 레벨 조절하는 것
> + 사용자의 임시 수정, 필요 없는 정보로의 접근, 소스의 보호 등을 위해 접근 막는다.
> + 또한 class 설계 시 class간 간섭, 정보 공유의 최소화를 위해 사용한다. 
> + 변수 선언 시 앞에 \_\_붙여 Private 변수로 선언해준다. 
>
> ```python
> class NewClass():
>   def __init__(self):
>     self.__priv_val = []  # __priv_val : class 외부에서 접근 불가
>
>   def add_priv_val(self, variable):  # add_priv_val 함수로만 priv_val에 접근 가능
>     self.__priv_val.append(variable)
> ```
> 
> + **property decorator**를 이용하면 private 변수에 접근 가능하게 해줄 수 있다.
>
> ```python
> @property
> def priv_val(self):
>   return self.__priv_val
> ```
> 
> 위와 같이 property decorator를 이용한 함수를 class에 만들어주면 ```priv_val``` 함수를 변수처럼 호출할 수 있게 된다. (내부값 수정 방지 위해 copy로 전달 권장)

<br><br><br>

# Module and Projects
---
### Module
> + Python의 모듈은 **.py** 파일들을 말한다.
> + 만들어진 모듈을 사용하기위해서는 저장된 .py 파일의 이름을 import 문으로 호출해주면 된다. 
>
> ```python
> import my_module  # my_module.py 파일로 저장된 모듈 호출
> ```
>
> + 위의 방식으로 모듈을 import하면 모듈의 코드가 메모리에 로드되어 파일 내의 함수가 사용가능해진다. 
> + 모듈 로드 시에 뒤에 ```.py```나 ()는 없이 호출하면 된다.
> + 호출 시 전체 코드가 아닌 코드의 일부 범위만 호출할 수도, 로드한 모듈을 별칭으로 사용할 수도 있다.
>
> ```python
> import my_module as mm  # my_module을 mm이라는 alias(별칭)으로 사용
> from my_module import my_class  # my_module에서 my_class만 로드
> from my_module import *  # my_module 전부다 로드
> ```
> 
> + Python에는 기본 제공 라이브러리인 **Built-in Module**이 다양하게 제공된다.
> + Built-in module은 설치할 필요 없이 import로 바로 활용 가능하다.
>   + (종류가 워낙 다양해 필요시마다 검색해서 알아보자)

<br><br>

### Package
> + 하나의 대형 프로젝트를 만드는 코드의 묶음이다.
> + 다양한 모듈들의 합을 말하며, 패키지는 폴더로 연결된다. 
> + 세부적인 기능별로 폴더를 구성해야하며 각 폴더에 필요한 모듈들을 구현하면 된다.
> + ```__init__ ```, ```__main__```과 같은 키워드 파일명이 사용된다. (3.3 버전 이후부터는 ```__init__``` 없어도 상관없으나 왠만하면 만드는 편이다.)
>
> ```python
> # __init__.py
> __all__ = ["module1", "module2"]  # 사용할 모듈
> from . import module1  # 현재 폴더에서 module1 import
> from . import module2  # 현재 폴더에서 module2 import
>
> # __main__.py
> if __name__ == "__main__":
>   ...
> ```
>
> 위와 같이 ```__init__.py```와 ```__main__.py```를 작성하면 된다. 
> + Package 내에서 다른 폴더의 모듈을 호출하려면 상대참조로 호출한다.
>
> ```python
> from .folder1 import module  # 현재 폴더안의 folder1에서 module import
> from ..folder2 import module2  # 현재 위치의 부모 디렉토리 안의 module2 import
> ```

<br><br>

### Open Source Library (with Virtual environment)
> + Python은 다양한 오픈소스 라이브러리가 존재한다. 
> + 오픈소스 라이브러리를 사용할 때에는 오픈소스 간의 충돌을 방지 하기위해 가상 환경을 설정해야한다.
> + 대표적으로는 virtualenv와 conda가 있는데 나는 주로 conda를 사용할 예정이다. 
> 
> + ```conda create -n project_name python=3.8```로 project_name인 가상환경 생성
> + ```conda ativate project_name```으로 가상환경 실행
> + ```conda deactivate```으로 가상환경 종료
> + ```conda install package_name```으로 필요한 패키지 설치

> **Tip)**
> + 오픈소스 라이브러리 중 matplotlib과 tqdm은 상당히 유용하니 설치하자.
> + tqdm : 코드 내의 loop가 얼마나 남았는지를 쉽게 보여준다.
> + matplotlib : 데이터 분석에 유용한 다양한 기능들을 제공한다.

<br><br>

이전에 Transformer 구현 코드를 읽고 한번 구현해 보려고 했던 적이 있었다. 하지만 당시 ```__init__```, ```__main__``` 함수가 뭘 말하는지를 몰라 상당히 난항을 겪었던 적이 있었는데, 오늘 강의를 들음으로써 드디어 이해할 수 있게 되었다. <br>
앞으로 다양한 프로젝트를 제작하게 될 것인데, 오늘 학습한 내용이 상당히 유용하게 사용될 것이라 예상되며 decorator의 경우는 좀 더 추가로 학습해야 될 것 같다.<br>
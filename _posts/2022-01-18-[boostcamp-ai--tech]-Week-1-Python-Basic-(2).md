---
layout: post
title: '[boostcamp ai-tech] Week1 Python Basic (2)'
categories: [boostcamp]
---

---

오늘도 파이썬 기초 강의를 들었다. 오늘 들은 강의는 대략 정리해보자면, <br>
+ **파이썬의 여러가지 데이터 타입과 자료구조**
+ **Pythonic Coding 방법**<br><br>

이렇게 두가지로 요약할 수 있다.<br><br>

# 1. Data types and Data Structures
---
Python에는 int, float, string, boolean 등 여러가지 data type이 있다. <br>
Python은 Dynamic typing이라는 것을 지원해서 변수를 선언할 때 data type을 지정해주지 않아도 interpreter가 알아서 data type을 <br>
해석해서 지정해준다. <br>
이는 사용자에게 편의성을 가져다주지만 compiler 언어들에 비해 속도가 비교적 떨어진다. <br><br>
**Dynamic Typing** : 변수의 타입 interpreter가 알아서 지정 (but, 속도 떨어짐)

```python
integer_val = 10
str_val = 'abc'
print(type(integer_val))    # <class, 'int'>
print(type(str_val))        # <class, 'str'>
```
<br>

### List
+ **List** : sequence 자료형, 여러 데이터들을 가지는 집합.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 각 데이터들은 주소값(offset)을 가진다. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; List는 여러 연산이 가능한데, 아래의 기능들을 많이 사용하게 된다. <br>
>    + Indexing : 주소값 사용해서 할당값을 호출하는 것 (index는 0부터 시작)
>    + Slicing : list의 값들을 index로 잘라서 사용하는 것
>    + concatenate : list와 list 합치는 것
>    + list.append() : list에 새 값을 추가하는 것
>    + list.extend(list) : list의 뒤에 list를 하나 concat하는 것 (concatenate와 같음)

다른 언어와 달리 Python은 한 list안에 다양한 종류의 자료형을 저장할 수 있다.

```python
arr = [1, 2, 3, 'abc', 2.4]
print(arr[1])   # 2
print(arr[1:4]) # [2, 3, 'abc']
print(arr+arr)  # [1, 2, 3, 'abc', 2.4, 1, 2, 3, 'abc', 2.4]
arr.append(9)
print(arr)      # [1, 2, 3, 'abc', 2.4, 9]
arr.extend(arr)
print(arr)      # [1, 2, 3, 'abc', 2.4, 9, 1, 2, 3, 'abc', 2.4, 9]
```
<br>

List를 복사할 때는 주의해야할 점이 있다.<br>
arr1을 arr2에 그대로 복사하고자 할 때 ```arr2 = arr1```과 같이 작성하게 되면 arr1의 주소값이 arr2에 그대로 들어가기에 값만 복사하기 위해서는 ```arr2 = arr[:]```과 같이 작성해야한다.<br>
2차원 list일 경우<br>

```python
import copy
arr2 = copy.deepcopy(arr1)
```

<br><br>

### Stack
+ **Stack** : LIFO(Last In First Out) 구조 : 마지막에 넣은 데이터부터 반환<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Data 입력 = Push, Stack의 마지막에 데이터 집어넣음<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Data 출력 = Pop, Stack의 마지막에 있는 데이터 뽑아냄<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; List로 Stack 구현이 가능하다. (Push => list.append(),  Pop => list.pop())<br><br>

### Queue
+ **Queue** : FIFO(First In First Out) 구조 : 먼저 넣은 데이터부터 반환<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Data 입력 = Put, Queue의 마지막에 데이터 집어넣음<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Data 출력 = Get, Queue의 맨 앞의 데이터 뽑아냄<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Stack과 마찬가지로 List로 구현이 가능하다. (Put => list.append(),  Get => list.pop(0))<br><br>

### Tuple
+ **Tuple** : 값 변경이 불가능한 list<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 선언할 때 ()사용<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; list의 연산, indexing, slicing 등을 동일하게 사용가능함<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 사용자의 실수에 의한 에러를 미리 방지하기 위해서 사용된다. (프로그램 동작하는 동안 값이 바뀌지않는 데이터를 저장)<br><br>

### Set
+ **Set** : 중복을 불허하며 값을 순서없이 저장하는 sequence data<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *set()* 객체를 선언하여 생성<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *set.update(list) / set.add()* : 원소 추가<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *set.remove() / set.discard()* : 원소 삭제<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *set.clear()* : 모든 원소 삭제<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |, &, - 로 집합연산 가능 (순서대로 union, intersection, difference)<br><br>

### Dict
+ **Dict** : 데이터를 Key:Value 형태로 저장하는 sequence data<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Key를 이용해 Value를 관리, 검색한다. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *dict.items()* : Value만 tuple 형태로 출력<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *dict.keys()* : Key만 출력 <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; in을 이용해서 특정 key/value가 dict안에 존재하는지 확인 가능<br><br>

### Collections module
- List, Tuple, Dict에 대한 python built-in 확장 자료 구조이다. 
- 부스트캠프 코딩테스트를 준비하면서 상당히 자주보고 자주썼던 모듈이다. 상당히 유용하니 잘 기억해두는게 좋을 것 같다.
- deque, Counter, OrderedDict, defaultdict, namedtuple이 있다. (OrderedDict와 namedtuple은 잘 쓰이지않아 정리에서 제외한다)<br><br>

#### deque
> + Stack과 Queue를 지원하는 모듈
> + List로 stack/queue 만들 때보다 효율적인 저장 방식을 지원한다. (list보다 빠르다)
> + rotate, reverse등 linked list의 특성을 지원한다. 
> + list의 기능(append, extend)에 추가적으로 leftappend, leftextend를 지원하며 다른 list의 기능 또한 지원된다.
> + 

<br><br>

#### defaultdict
> + dict type 자료형의 기본 Value를 지정해준 형태
> + 생성시 Key에 Value 정해주지 않아도 기본값을 넣어준다. 
>```python
>defaultdict(lambda : defaultvalue) # 기본값은 함수의 형태로
>```
> + Text-mining 접근법 - Vector Space Model에 사용가능

<br><br>

#### Counter
> + Sequence type의 data element들 갯수를 dict 형태로 반환한다.
> + 다시말해 sequence data에서 각각의 element가 몇번씩 나타났는지를 세주는 자료형
> + 

<br><br>

# 2. Pythonic Coding
---
Pythonic code란 파이썬 스타일의 코딩 기법을 말한다.<br>
파이썬은 여러 언어들의 장점을 채용해 효율적으로 코드를 표현할 수 있게 한다.<br><br>
코드를 굳이 왜 pythonic하게 짜야되는가 라고 생각할 수도 있을텐데, <br>
이미 많은 수의 개발자들이 pythonic하게 코딩하고 있기에 **협업**에 있어서 코딩 스타일은 꽤나 중요한 요소가 된다. 그리고 pythonic code는 코드의 길이를 줄일 수 있을 뿐만 아니라 **효율성** 측면에서도 꽤 좋다.<br>
파이썬스러움은 아래의 기법들이 아주 잘 나타낸다.<br>
- split & join
- List comprehension
- enumerate & zip
- lambda & map & reduce
- generator
- asterisk
<br><br>

### split & join
> #### split
> + string type의 값을 기준값을 이용해서 쪼개 list로 만든다.
> + 만약 기준값 없이 사용하게 되면 공백을 기준으로 string을 나눈다.
> ```python
> split_str = 'first second third'.split(' ')   # 공백 기준으로 쪼개기
> print(split_str)
> >>> ['first', 'second', 'third']
> ```

> #### join
> + 기준값을 이용해서 sequence data를 하나의 string으로 합친다. 
> ```python
> # split의 코드에 이어서
> '='.join(split_str)
> >>> 'first=second=third'
> ```

<br><br>

### list comprehension
> + 기존의 list에 기반한 새로운 list를 만들기 위한 문법적 구조
> + Python에서 가장 많이 사용되는 문법 중 하나
> + ```for + append```같은 방식보다 더 빠른 속도를 가진다.
> + 여러 줄의 list 생성 코드를 한 줄로 간단히 표현할 수 있다. <br>
> *list 생성 코드* <br>
> 
> ```python
> arr1 = []
> for i in range(10):
>     arr.append(i**2)   # arr1 = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
> 
> arr2 = []
> for i in range(3):
>         arr2.append([])
>     for j in range(4):
>         arr2[-1].append(i+j)  # arr2 = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
> 
> arr3 = []
> for i in range(1,4):
>     for j in range(1,4):
>         arr3.append(i*j)      # arr3 = [1, 2, 3, 2, 4, 6, 3, 6, 9]
> 
> arr4 = []
> for i in range(4):
>     for j in range(4):
>         if i!=0 and j!=0:
>             arr4.append(i+j)  # arr4 = [2, 3, 4, 3, 4, 5, 4, 5, 6]
> ```
>
> *List comprehension* <br>
>
> ```python
> arr1 = [i**2 for i in range(10)]   # arr1 = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
> 
> arr2 = [[i+j for j in range(4)] for i in range(3)]    # arr2 = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
> 
> arr3 = [i*j for i in range(1,4) for j in range(1,4)]  # arr3 = [1, 2, 3, 2, 4, 6, 3, 6, 9]
> 
> arr4 = [i+j for i in range(4) for j in range(4) if i!=0 and j!=0] # arr4 = [2, 3, 4, 3, 4, 5, 4, 5, 6]
> ```

<br><br>

### enumerate
> + list의 element를 추출할 때 번호를 붙여서 뽑아내는 방법
> + ```list(enumerate(list))```하면 ```(list_index, list[list_index])```형태의 tuple로 반환된다.
> + 보통 dict type으로 선언하여 사용한다. (list(list_index):list_index 의 순서로 보통 사용)

<br><br>

### zip
> + list 여러개를 묶어 병렬적으로 추출

<br><br>

### lambda
> + 함수명없이 사용할 수 있는 익명함수
> + ``` lambda variables : method_contents ```의 형태처럼 사용한다
> Python3부터는 lambda의 사용을 권장하지 않는다고 한다.<br>
> 문법이 어려운데다가, 코드의 테스드와 해석도 쉽지 않은데다가, docstring 지원이 안돼서 사용을 지양하자는 의견이 많이 나오는 추세이다. <br>
> 하지만 여전히 많이 쓰이긴한다...

<br><br>

### map
> + Sequence형 데이터에 함수를 mapping해준다.
> + return은 generator로 나와서 만약 list형태로 보고싶다면 ```list(map(...))```으로 사용해야한다. 
> + map은 실행되는 시점에 값을 생성하여 메모리 효율적인 방식이라 한다. 
> + 여러개의 sequence 데이터에도 적용이 가능하다. 
> lambda와 마찬가지로 어려워서 map function 보다는 list comprehension을 권장하는 추세이다. <br>
> 하지만 마찬가지로 여전히 많이 사용된다...

<br><br>

### reduce
> + map function과 같이 sequence형 데이터에 함수를 적용시킨다. 다른 점은 reduce function은 같은 함수를 적용시켜 sequence형 데이터를 통합해나가는 함수이다. 
> + 대용량 데이터를 다룰 때 map과 함께 유용하게 사용된다.

<br><br>

위의 함수들은 간단한 코드로도 매우 다양한 기능을 제공한다. <br>
하지만 lambda, map, reduce의 경우 코드의 직관성이 떨어져 python3에서는 권장하지 않고 있다. <br>
그러나 머신러닝 코드나 오래된 library에서는 여전히 많이 사용하기 때문에 사용법을 숙지할 필요가 있다. <br><br>

### iterable object
> + Sequence형 자료형에서 데이터를 순서대로 추출하는 object
> + 내부적으로 구현된 매직메소드 ```__iter__```와 ```__next__```가 사용됨
> + ```iter()```함수로 iterator 만들 수 있음
> 
> ```python
> arr = [1,2,3,4]
> arr_iter = iter(arr)  # 또는
> arr_iter2 = arr.__iter__()  # type(arr_iter) => <class 'list_iterator'>
> ```
> 
> + a_iter는 메모리 주소값을 가지고 있다.
> 
> ```
> >>> a_iter
> <list_iterator object at 0x0000018B8C6EF340>
> ```
>
> +  ```next()``` 함수로 iterator의 값을 하나씩 뽑아낼 수 있다.
>
> ```
> >>> next(a_iter)
> 1
> >>> next(a_iter)
> 2
> >>> next(a_iter)
> 3
> >>> next(a_iter)
> StopIteration
> ```
>
> + iterator의 끝까지 가게 되면 StopIteration 예외가 발생한다.

<br><br>

### generator
> + iterable object를 특수한 형태로 사용해주는 함수다.
> + Python wiki에 따르면 generator 함수는 iteraotr 처럼 동작하는 함수를 선언해준다고 한다. (아직은 잘 이해가 되지는 않는다...)
> + 선언 시에는 list comprehension과 비슷한 방식으로 선언이 가능한데, [,] 대신 (,)를 사용해서 선언한다. 
> + generator는 yield라는 것을 사용해서 한번에 하나의 element만 반환한다.
> + for loop으로 호출 시 마다 값을 하나씩 출력하게 되는데 다음과 같은 결과가 나온다.
> 
> ```python
> gen_list = (i*i for i in range(1,4))
> for gen_val in gen_list:
>   print(gen_val)
> # 1 
> # 4 
> # 9
> ```
> + list type의 데이터를 만들 때에는 generator를 이용해서 만드는 것이 메모리를 효율적으로 사용할 수 있다. 
> + 특히 대용량 데이터나 파일 데이터를 처리할 때에는 generator가 매우 유용하게 쓰인다. 

<br><br>

## Function arguments
함수에 입력되는 arguments는 다음과 같이 다양한 형태를 띈다. 
1. Keyword arguments
2. Default arguments
3. Variable-length arguments
<br><br>

### Keyword arguments
> + 함수에 입력되는 parameter의 변수명을 사용해서 arguments를 넘겨주는 방식
>
> ```python
> def sample_func(arg1, arg2):
>   ...
> sample_func(arg1=value1, arg2=value2)
> ```
> 
> + 위와 같이 각 argument에 맞춰서 parameter값을 넘겨주는 방법이다. 
> + keyword arguments로 함수에 parameter값을 전해줄 시 arguments의 순서가 바뀌더라도 알맞은 parameter가 각 argument에 전달된다.
> + 쉽게 말해 순서 신경안써도 된다.

<br><br>

### Default arguments
> + 함수의 arguments에 기본값을 지정해주는 방식
> + 함수 사용 시 parameter를 전달해주지 않아도 기본값으로 자동 설정된다.
> ```python
> def sample_func(arg1, arg2 = default_val):
>   ...
> sample_func(val1)  # arg1에 val1 전달되고 arg2는 기본값인 default_val이 된다.
> ```

<br><br>

### Variable-length arguments (가변 인자)
> + 갯수가 정해지지 않은 변수를 함수의 parameter로 사용할 때 쓰이는 방법이다.
> + Asterisk 기호(*)를 사용해 함수의 parameter를 표시한다.
> + 일반적으로 가변 인자는 ```*args```를 변수명으로 쓴다.
> + 입력된 가변 인자는 tuple type으로 사용된다. 
> + 주의해야될 점이 있는데, 가변 인자는 함수에 **단 하나만** 사용가능하며, parameter의 제일 **마지막 순서**에서 사용 가능하다.
> 
> ```python
> def sample_func(arg1, arg2, *args):
>    ...
> sample_func(val1, val2, val3, val4, val5)
> # val1 -> arg1, val2 -> arg2, (val3, val4, val5) -> *args
> ```

<br><br>

### Keyword variable-length arguments (키워드 가변인자)
> + parameter의 이름을 따로 지정하지않고 입력하는 방법이다. 
> + Asterisk 두개 사용해 함수의 parameter를 표시한다. ```**kwargs```
> + 입력된 값은 dict type으로 사용할 수 있다.
> + 키워드 가변인자는 가변인자와 마찬가지로 함수마다 **하나만** 사용가능하며, 순서는 **기존 가변인자 다음**에서 사용 가능하다.
> ```python
> def sample_func(arg1, arg2, *args, **kwargs):
>   ...
> sample_func(val1, vla2, val3, val4, val5, kwarg1=val6, kwarg2=val7, kwarg3=val8)
> # val1 -> arg1, val2 -> arg2, (val3, val4, val5) -> *args, 
> # {'kwarg1':val6, 'kwarg2':val7, 'kwarg3':val8} -> **kwargs
> ```

가변 인자와 키워드 가변 인자는 머신러닝 알고리즘에 아주 많이 사용된다. <br>
추가로 더 공부해보고 적재적소에 잘 쓸 수 있도록 자주 사용해야겠다.

<br><br>

### Asterisk (*)
> + Asterisk는 곱셈, 제곱, 가변 인자 활용 등 다양하게 사용된다. 
> + tuple, dict, list 등의 container를 unpacking할 때 사용된다.
> + dict type 데이터를 unpacking할 때에는 Asterisk 두개 사용 (Keyword unpacking이라고 한다)
>
> ```python
> test_list = [1, 2, 3, 4]
> print(*test_list)  # 1 2 3 4
> test_tuple = (3, 4, 5, 6)
> print(*test_tuple)  # 3 4 5 6
> 
> def func(arg1, *args): 
>   ...
> func(1, (2,3,4,5))  # arg1 : 1, *args : ((2,3,4,5))
> func(1, *(2,3,4,5)) # arg1 : 1, *args : (2,3,4,5)
> # (2,3,4,5) tuple이 unpacking되어 들어감
> ```

<br>

---
### f-string
f-string은 console에 print함수로 출력할 때 사용되는 형식인데, %-format이나 format함수로도 출력 형식을 지정해줄 수 있으나, 요즘에는 f-string이 대세라고 한다. <br>
사용하는 방법은 다음과 같다. <br>

```python
print(f"~~~{variable1}~~~{variable2}")
```

<br><br>

### function type hint
> + 함수의 interface를 알기 쉽도록 함수의 define 부분에 변수의 타입을 명시해줄 수 있다. 
> 
> ```python
> def func(variable_name: variable_type) -> return_type:
>   ...
> ```
>
> + type hint를 사용하면 함수의 정보를 명확히 알 수 있다.
> + 발생 가능한 오류를 사전에 확인할 수 있으며, 시스템의 전체적인 안정성을 확보할 수 있다.

<br><br>

### function docstring
> + Python method의 상세 스펙을 사전에 작성하여 함수에 대한 이해도를 올려준다.
> + 따옴표 세개를 이용해 docstring 영역을 함수명 바로 아래에 표시한다.
> ```python
> """
>   Purpose of Method
>   Parameters:
>     ...
>   Returns:
>     ...
> """
> ```
> Tip) VScode에서는 docstring generator를 깔고 쓰면 편하게 docstring 만들 수 있다.<br>
> (Ctrl + Shift + P => generate docstring 으로 docstring 자동 생성)

<br><br>

### Method 작성 가이드라인
1. 함수는 가능한 줄을 **짧게**
2. 함수명에 역할, 의도 **명확하게**
3. 한 함수에는 **유사한** 역할하는 코드만
4. 전달받은 parameter 자체를 변경하지 말기
5. **여러번** 쓰이는 코드는 함수로
6. **복잡한** 수식, 조건은 함수로

<br><br>

### Coding Convention (코딩 규칙)
실무에서 코딩은 팀 단위로 이루어지기 때문에 규칙 지키는 것이 상당히 중요하다.<br>
주요 룰은 다음과 같다. <br><br>

1. 들여쓰기는 **space 4칸** 권장
2. 코드 한줄은 최대 **79자**
3. 불필요한 공백은 없애기
4. 연산자는 **한칸씩만** 띄우기
5. 소문자 L, 대문자 O, 대문자 i **금지**
6. 함수명은 **소문자로만**, **snake_case** 사용
7. 클래스명은 **CamelCase** 사용
<br><br>

Tip1.) flask8로 coding convention 준수하는지 확인이 가능하다.<br>
Tip2.) black으로 자동으로 coding convention 준수하도록 코드 교정이 가능하다.
<br><br>

Python data structure에 대해서는 나름 잘 알고 있다고 생각했는데, iterator나 generator에 관한 내용은 꽤나 낯설었다. 많이 유용하고 자주 쓰이는 함수인 만큼 이후에 더 알아본 후 내용을 추가해둬야겠다. <br>
그리고 Pythonic Coding 방법이 상당히 흥미로웠다. 취업 후 원할한 팀플레이를 위해 잘 숙지해야겠다.
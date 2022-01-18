---
layout: post
title: '[boostcamp ai-tech] Week1 Python Basic'
categories: [boostcamp]
---

---

이번 주 부터 부스트캠프가 시작되었다. 한주간 강의를 듣고 복습겸 공부한 내용에 대해 정리해보려한다. <br>
첫 주차에는 기초적인 파이썬에 대한 강의가 있었다. <br>
(유용하거나 중요할 듯한 내용만 정리할 것이라 약간 듬성듬성 써질 듯 하다.)<br>

## DAY 1
---

파일의 경로에는 절대경로와 상대경로가 있다. <br>

>절대 경로 : root 디렉토리에서부터 현재 파일이 있는 위치까지의 전체 경로를 나타낸다. <br>
> __ex) C:\Users\...\Desktop\...\git\github_blog\mooncy0421.github.io__

> 상대 경로 : 현재 위치한 디렉토리를 기준으로 타겟이 되는 파일까지의 경로를 나타낸다.<br>
> __ex) ..\..\examplefile.txt__
<br><br>

### CLI 명령어
앞으로 터미널에서 상당히 많은 작업을 할 것이 예상되는데 여러 명령어들에 대해서는 정리하고 가는게 좋을 것 같아 자주 쓰일 것 같은 명령어들을 정리하려한다.<br>
각 명령어들은 옵션을 붙여서 디테일한 동작을 하도록 지시할 수 있는데 옵션까지 정리하기에는 너무 많은 내용을 넣어야 할 것 같아 각 옵션을 사용할 때마다 정리할 것이다.<br>

|설명|Shell|윈도우 CMD|
|----|:-----:|:---------:|
|현재 디렉토리 이름 보여주기/디렉토리 이동|cd/pwd|cd|
|디렉토리 내 목록 보기|ls|dir|
|디렉토리 생성|mkdir|mkdir/md|
|디렉토리 삭제|rm -r|rmdir|
|파일 생성|touch/vi|copy con (txt파일)|
|파일 복사|cp|copy|
|파일 이동|mv|move|
|파일, 디렉토리 이름 변경|mv|move|
|파일 삭제|rm|del|
|파일 내용 보기|cat|type(txt파일)|
|파일 내용 검색|grep|find|
|현재 터미널 화면 지우기|clear|cls|

[Shell 명령어 참고](https://zetawiki.com/wiki/%EC%9C%88%EB%8F%84%EC%9A%B0_CMD_%EB%AA%85%EB%A0%B9%EC%96%B4_%EB%AA%A9%EB%A1%9D) <br>
[윈도우 CMD 참고](https://velog.io/@jewon119/01.%EB%A6%AC%EB%88%85%EC%8A%A4-%EA%B8%B0%EC%B4%88-CLI-%EA%B8%B0%EB%B3%B8-%EB%AA%85%EB%A0%B9%EC%96%B4)

현재 중요하다 생각되는 명령어들은 위 표에 정리해두었는데, 추후 공부하며 다른 중요하다 생각되는 명령어들이 있으면 추가하려 한다. <br><br><br>

### Jupyter / Colab 단축키
그리고 개인 앞으로 Jupyter나 google Colab을 자주 이용하게 될 것인데, 여기에도 유용한 단축키들이 있다. <br>
Jupyter Notebook과 Colab은 같은 동작을 하더라도 다른 단축키를 사용하는 경우도 많다고 해서 여기에 정리해두고 헷갈릴 때 마다 보려고 한다.<br>

|동작|Jupyter Notebook|Colab|
|----|:-----:|:------:|
|Edit Mode 진입|Enter|Enter|
|CMD Mode 진입|ESC|ESC|
|현재 셀 실행(Edit)|Ctrl + Enter|Ctrl+Enter|
|현재 셀 실행 + 아래에 셀 삽입(Edit)|Alt + Enter|Alt + Enter|
|현재 셀 실행 + 아래 셀 이동(Edit)|Shift + Enter|Shift + Enter|
|아래 셀과 합치기(CMD)|Shift + M|개인 설정|
|선택 셀 삭제(CMD)|D D|Ctrl + M D|
|셀 잘라내기(CMD)|X|개인 설정|
|셀 복사(CMD)|C|개인 설정|
|셀 붙여넣기(CMD)|V|개인 설정|
|삭제 셀 되돌리기(CMD)|V|개인 설정|
|현재 셀 위에 셀 추가(CMD)|A|A|
|현재 셀 아래에 셀 추가(CMD)|B|B|
|현재 커서 기준으로 셀 분할|Ctrl + Shift + -|Ctrl + M + -|

단축키들은 Jupyter Notebook과 colab에 들어가면 모두 볼 수 있다. 우선 자주 쓰이는 단축키들은 위와 같다고 생각된다. 확실히 <br>
Jupyter Notebook 단축키가 더 쉽기도하고 편하게 쓸 수 있어 보인다. colab은 개인별로 설정이 가능하긴하니 나중에 비슷하게 설정하던가 해야겠다.<br>
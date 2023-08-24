---
layout: post
title: "Attention과 Transformer 이해하기"
date: 2023-08-24 14:15:56 +0900
categories: NLP
---

Transformer는 2017년 Google이 발표한 *Attention is All You Need*라는 논문을 통해 제안된 architecture입니다. BERT, GPT 등 NLP를 공부해본 사람이라면 다 알 법한 모델들이 바로 이 transformer를 기반으로 설계되었습니다. 워낙 유명하다보니 transformer를 설명한 글이 많은데요, 이 글은 transformer가 나오게 된 배경부터 시작해 세세한 모델 디테일까지, 직관적인 이해를 돕기 위해 작성되었습니다. 처음 transformer를 공부하는 입장이라면 조금 어려울 수도 있지만, 천천히 끝까지 이해하다 보면 앞으로 NLP를 공부하는 것이 훨씬 수월해지리라 생각합니다.

# 제안 배경

Sequential data인 $$\mathbf x = (x_1, \dots, x_n)$$가 주어졌다고 해봅시다. Sequential data라고 하는 것은 기본적으로 $$x_1, \dots, x_n$$이 서로 연관되어있음을 의미합니다. 각각 독립적으로 생성된 data가 아니기 때문에, sequential data를 다루는 모델들은 데이터 간의 연관성, 동향을 잘 파악해내는게 중요합니다. 예로, token이 모여져 만들어진 text는 sequential data이며, 따라서 언어 모델은 token 사이의 연관성, 즉 문맥을 잘 파악해내야 합니다. 해당 문서에서는 text 데이터를 기준으로 설명하겠습니다.

2017년 transformer이 세상에 나오기 전까지는 RNN과 LSTM이 sequential model을 처리하는 대표적 모델이었습니다. 두 모델 모두 $$x_1$$부터 $$x_n$$까지 순차적으로 입력을 받는데요, 모델에 $$x*i$$를 입력할 때 $$x_i$$ 뿐만 아니라 $$x_{i-1}$$을 입력해 얻어낸 출력 $$y_{i-1}$$을 같이 입력하는 방식입니다. "Recurrent"라고 불리는 이유이죠.

$$
\begin{aligned}
y_1 &= \operatorname{CNN}(x_{1}, c) \\
y_2 &= \operatorname{CNN}(x_2, y_1)= \operatorname{CNN}(x_{2}, \operatorname{CNN}(x_{1}, c)) \\
y_3 &= \operatorname{CNN}(x_{3}, y_{2})  = \operatorname{CNN}(x_{3}, \operatorname{CNN}(x_{2}, y_{1})) = \operatorname{CNN}(x_{3}, \operatorname{CNN}(x_{2}, \operatorname{CNN}(x_{1}, c))) \\
& \vdots \\
y_n &= \operatorname{CNN}(x_n, \operatorname{CNN}(x_{n-1}, \dots, \operatorname{CNN}(x_1, c)\dots))
\end{aligned}
$$

따라서 모든 $$y_{i}$$이 $$x_1, \dots, x_i$$을 고려해 출력될 수 있습니다.

그러나 문제가 있었는데요, sequence를 순차적으로 받아들여 순차적, 혹은 직렬적으로 처리하다보니 시간도 오래걸리고 가면 갈수록 초반에 입력받았던 값에 대한 정보를 잃어버리게 됩니다. (더 궁금하다면 CNN과 LSTM대해 알아보세요!)

Transformer는 이런 문제를 해결하기 위해 제안되었습니다. Transformer는 input $$\mathbf x = (x_1, \dots, x_n)$$을 한번에 받아 병렬적으로 처리하기 때문에 처리 속도도 빠를 뿐만 아니라 앞 전의 문맥 정보를 잃어버릴 가능성이 훨씬 적습니다. 그렇다면 $$x_1, \dots, x_n$$ 데이터 간 문맥을 어떻게 파악해낼까요? Transformer는 self-attention을 사용합니다.

# Self-Attention

사용자가 검색창을 통해 어떤 query를 입력하면 key를 통해 원하는 데이터를 찾은 후 그 value를 반환하는 key-value database를 생각해봅시다.

| Key (Title)      | Value (URL)                     |
| ---------------- | ------------------------------- |
| My dog is happy! | www.youtube.com/my_dog_is_happy |
| VLOG in Zürich   | www.youtube.com/VLOG_in_zurich  |
| ...              | ...                             |

유저로부터 query로 "dog videos"가 주어진다면, 첫 번째 key인 "My dog is happy!"가 두 번째 key인 "VLOG in Zürich"보다 더 유사하기 때문에, 첫 번째 key에 대항하는 동영상을 유저에게 제공해주는 것이 적절할 것입니다.

Self-attention은 데이터베이스의 retrieval system에서 영감을 받았습니다. 위에서 언급한 3가지(query, key, value)가 self-attention의 구성 요소입니다.

- Query $$\mathbf Q$$
- Key $$\mathbf K$$
- Value $$\mathbf V$$

Sequential data $$\mathbf x = (x_1, \dots, x_n)$$가 주어졌을 때, self-attention은 다음과 같은 식으로 서로의 연관성을 파악합니다.

```python
score = [[0] * n for _ in range(n)]

for i in range(n):
	# x_i가 query일 때,
	for j in range(n):
    	# x_j가 key일 때,
    	score[i][j] = score(x[i], x[j])
```

$$x_i$$가 query고 $$x_j$$가 key일 때, 둘이 얼마나 관련있는가를 함수 $$\operatorname{score}(x_i, x_j)$$를 통해 알아내는 것이죠.

<p align="center">
<img src="https://velog.velcdn.com/images/kaist19/post/c96eaa78-9d7b-4b25-9825-9ee08dd96f74/image.png" width="300">
</p>

위의 그림은 학습한 transformer의 $$\operatorname{score}(x_i, x_j)$$를 나타낸 것입니다. 왼쪽의 token이 query로 주어졌을 때, 관련성이 높은 token이 높은 $$\operatorname{score}(x_i, x_j)$$을 알 수 있습니다.

그렇다면, $$\operatorname{score}(x_i, x_j)$$는 어떻게 계산할까요?
NLP를 공부하셨다면, 내적으로 어떤 유사성을 계산해내는 것에 익숙하실 것입니다. 여기서도 마찬가지인데요, 각 token $$x_i, x_j$$에서 query $$q_i$$와 key $$k_j$$를 먼저 계산합니다.

$$
\begin{aligned}
q_i &= x_i W^Q \\
k_j &= x_j W^K
\end{aligned}
$$

$$W^Q$$와 $$W^K$$는 $$x_i$$에서 각각 $$q_i$$와 $$k_i$$를 얻을 수 있게 해주는 선형 연산자, 즉 행렬입니다.

이때 $$q_i$$와 $$k_j$$의 내적 $$q_i \cdot k_j$$이 $$\operatorname{score}(x_i, x_j)$$가 됩니다.
따라서,

$$
\operatorname{score}(x_i, x_j) = q_i \cdot k_j = q_i k_j^\intercal
$$

가 되며, 모든 query $$q_1, \dots, q_n$$을 묶어 행렬 $$Q = (q_1, \dots, q_n)$$을 만들고, 모든 key $$k_1, \cdots,k_n$$을 묶어 $$K = (k_1, \dots, k_n)$$을 만들면,

$$
\begin{aligned}
\operatorname{score}(x_i, x_j) &= row_i(Q) \cdot row_j(K) \\
&= row_i (Q) column_j(K^\intercal) \\
&= (QK^\intercal)_{ij}
\end{aligned}
$$

이 됩니다. 즉, $$\mathbf x$$로부터 계산된 $$QK^\intercal$$은 $$x_i$$가 query, $$x_j$$가 key일 때 연관성을 의미합니다.

여기서 token이 존재하는 embedding 공간의 dimensionality $$d_k$$가 커짐에 따라 $$QK^\intercal$$이 무한대로 커지게 되면 컴퓨터 연산 도중 문제가 발생할 수 있으므로 $$\sqrt{d_k}$$로 나눠줍니다.

$$
QK^\intercal \rightarrow \frac{QK^\intercal}{\sqrt{d_k}}
$$

마지막으로, "query $$q_i$$가 주어졌을 때 각 key $$k_j$$가 얼마나 연관성이 있는지"로 해석하기 위해서는 이 값을 그대로 쓰는 것 보다 $$\operatorname{softmax}$$를 통해 normalize해주는 것이 타당할 것입니다.

$$
QK^\intercal \rightarrow \frac{QK^\intercal}{\sqrt{d_k}} \rightarrow \operatorname{softmax}(\frac{QK^\intercal}{\sqrt{d_k}})
$$

행렬에 $$\operatorname{softmax}$$을 취하는 것이 조금 의아할 수도 있는데, 위에서 설명한 대로 어떤 query $$q_i$$가 주어졌을 때 key $$k_j$$들과의 연관성을 normalize하는 것이 맞으므로, 행렬 $$\operatorname{softmax}(\frac{QK^\intercal}{\sqrt{d_k}})$$의 row에 각각 $$\operatorname{softmax}$$을 취해주면 됩니다.

마지막으로 해당 값에 value $$V = (v_1, \dots, v_n)$$를 곱해주면 됩니다. $$v_i$$ 또한 query, key와 마찬가지로 $$v_i = x_i W^V$$를 통해 얻어지며, 따라서

$$
\operatorname{SelfAttention}(\mathbf x) = \operatorname{softmax}(\frac{QK^\intercal}{\sqrt{d_k}})V
$$

로 정리할 수 있습니다. _Attention is All You Need_ 논문에서는 이를 **Scaled Dot-Product Attention**라고 부릅니다.

# Multi-Head Attention

위에서 살펴본바와 같이 self-attention은 어떤 input $$\mathbf x$$에 대해 연산자 $$W^Q, W^K, W^V$$을 곱해 query, key, value를 각각 얻어내는 방식을 사용했습니다. 그러나, 어떤 sequential data들은 한 가지로만 연결되어 있는 것이 아니라, 여러 factor에 의해 연결되어 있는 경우가 많습니다. 예로 text의 경우, 동의어거나 품사가 같다거나 등등의 연관성이 있을 수 있겠죠.

Multi-head attention은 이러한 다양한 연관성을 잡아내기 위해 같은 sequential data $$\mathbf x$$에 대해 각각 다른 $$(W^Q, W^K, W^V)$$을 가진 $$\operatorname{SelfAttention}$$을 사용합니다. $$i$$로 인덱싱 하면,

$$
\operatorname{SelfAttention}_i(\mathbf x) = \operatorname{softmax}(\frac{QK^\intercal}{\sqrt{d_k}})V
$$

가 되며, 이 $$\operatorname{SelfAttention}_i$$에 해당하는 $$(Q, K, V)$$를 얻기 위해 해당하는 $$W_i^Q, W_i^K, W_i^V$$도 존재할 것입니다.

Multi-head attention은 이 여러 $$\operatorname{SelfAttention}_i$$를 사용해 input $$\mathbf x$$로부터 얻어낸 결과를 concatenate, 즉 이어붙인 후 input과 차원을 다시 맞춰주기 위해 연산자 $$W^O$$를 통해 project하는 방식을 사용합니다. Multi-head attention에서는 $$\operatorname{SelfAttention}_i$$를 $$\operatorname{head}_i$$라고 대신 칭하며, 정리하면

$$
\operatorname{MultiHeadAttnention}(\mathbf x) = \operatorname{Concat}(\operatorname{head}_1(\mathbf x), \dots , \operatorname{head}_h(\mathbf x)) W^O
$$

가 됩니다.

---

이제 transformer를 알아보기 위한 준비가 끝났습니다!

> 아래부터 미완성 글입니다! 계속 읽어도 무방하지만, 아직 설명이 빈약하며 추후에 positional encoding같은 내용도 추가할 예정입니다.

# Transformer

![](https://velog.velcdn.com/images/kaist19/post/7a4f155d-1923-400a-b710-b4d38d92553c/image.png)

Transformer의 기본 architecture입니다. 왼쪽 파트를 encoder, 오른쪽 파트가 decoder입니다. 그림에 보이는 $$N \times$$는 해당 구조가 $$N$$번 쌓아져 있다는 뜻인데요, transformer block라고 불리는 구조이며 encoder와 decoder의 transformer block은 구성이 살짝 다릅니다. 논문에서는 encoder와 decoder 모두 6번을 쌓았습니다. ($$N = 6$$)
Encoder는 말 그대로 text를 encode하는 부분이며, 이와 비슷한 기능을 하는 언어 모델 BERT가 이 encoder만을 가지고 설계되었습니다. 오른쪽 decoder는 text decode, 즉 생성에 관여하는 부분으로서 언어 생성 모델인 GPT가 decoder만을 가지고 만들어졌습니다.

# Transformer Block

![](https://velog.velcdn.com/images/kaist19/post/8278dec4-7113-4d8b-b93b-801722fca415/image.png)

## 1. Input $$\mathbf x$$

$$\mathbf x$$를 입력 받습니다.

## 2. Multi-Head Attention, Add & Norm

받은 입력 $$\mathbf x$$에

1. multi-head attention을 적용,
2. residual connection을 적용,
3. Layer Normalization을 적용합니다.

$$
\mathbf z = \operatorname{LayerNorm}(\mathbf x + \operatorname{MultiHeadAttention}(\mathbf x))
$$

### Multi-Head Attention

$$
\operatorname{MultiHeadAttnention}(\mathbf x) = \operatorname{Concat}(\operatorname{head}_1(\mathbf x), \dots , \operatorname{head}_h(\mathbf x)) \mathbf W^O
$$

$$
\operatorname{head}_i (\mathbf x) = \operatorname{SelfAttention}(\mathbf Q, \mathbf K, \mathbf V) = \operatorname{softmax}\left(\frac{\mathbf Q \mathbf K^\intercal}{\sqrt{d_k}} \right) \mathbf V
$$

where

- Query $$\mathbf Q = \mathbf x \mathbf W_i^Q$$
- Key $$\mathbf K = \mathbf x \mathbf W_i^K$$
- Value $$\mathbf V = \mathbf x \mathbf W_i^V$$
- $$d_k = dim(\mathbf V)$$

### Residual Connection

$$\mathbf x + \operatorname{MultiHeadAttention}(\mathbf x)$$

계산한 $$\operatorname{MultiHeadAttention}(\mathbf x)$$에 다시 $$\mathbf x$$를 더해주는 과정입니다. RNN에서 forward propagation & backward propagation을 할 때 발생했던 문제들을 해결해주기 위해 cell을 추가해 준 것과 비슷합니다.

### Layer Normalization

$$\mathbf z = \operatorname{LayerNorm}(\mathbf x + \operatorname{MultiHeadAttention}(\mathbf x))$$\*\*

$$\operatorname{LayerNorm}(\mathbf a) = \left( \frac{\mathbf a - \mathbb E[\mathbf a]}{\sqrt{\operatorname{Var}[\mathbf a] + \epsilon}} \right) * \gamma + \beta.$$
$$\gamma$$ and $$\beta$$ are learnable parameters, representing **gain** and **offset**, respectively.

## 3. Feed Forward & Add & Norm

$$\mathbf y = \operatorname{LayerNorm}(\mathbf z + \operatorname{FFN}(\mathbf z))$$

## 4. Output $$\mathbf y$$

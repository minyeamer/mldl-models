# BERT Background

## Index
- [Introduction](#introduction)
- [BERT Summary](#bert-summary)
- [BERT Paper](#bert-paper)

---

## Introduction
- 자연어 처리에서 가장 유명한 언어 모델 중 하나인 BERT에 대해 다뤄보자는 취지로 해당 분석을 진행합니다.
- BERT의 논문을 분석할 계획이며, BERT의 근간이 되는 Transformer에 대한 분석은 [해당 링크](../transformer/background.md)를 참고해주시기 바랍니다.
- BERT에 대해 한눈에 이해할 수 있도록 필요한 내용을 정리하여 요약합니다.
- 별도의 노트북 파일에서 BERT에 대한 코드 분석을 수행합니다.
- 해당 문서는 필요에 의하여 지속적으로 업데이트 됩니다.

---

## BERT Summary
- BERT는 인코더-디코더 구조의 트랜스포머 모델에서 인코더 부분만 사용합니다.
- 인코더 레이어 L, 어텐션 헤드 A, 은닉 유닛 H에 대해,   
  BERT_base는 (L=12, A=12, H=768), BERT_large는 (L=24, A=16, H=1024) 크기를 가집니다.
- 거대한 말뭉치에 대해 MLM과 NSP 태스크로 사전 학습한 가중치를 새로운 태스크에 적용할 수 있습니다.

### BERT Embedding
- BERT의 입력 데이터는 토큰 임베딩, 세그먼트 임베딩, 위치 임베딩의 합으로 표현됩니다.
- **Token Embedding**은 첫 번째 문장의 시작 부분에 [CLS] 토큰을, 모든 문장의 끝에 [SEP] 토큰을 추가합니다.
- **Segment Embedding**은 [SEP] 토큰과 별도로 두 문장을 구별하기 위해 입력 토큰 $E_A, E_B$를 제공합니다.
- **Position Embedding**은 트랜스포머에서 사용된 것과 같이 단어의 위치에 대한 정보를 제공합니다.

### BERT Tokenizer
- BERT는 **WordPiece Tokenizer**를 사용하는데, 해당 토크나이저는 BPE 알고리즘에 기반합니다.
- **Byte Pair Encoding**은 모든 단어를 문자로 나누고 고유 문자를 어휘 사전에 추가하는 알고리즘이며,   
  어휘 사전 크기에 도달할 때까지 가장 빈도수가 큰 기호 쌍을 반복적으로 병합해 어휘 사전에 추가합니다.
- **Byte-Level Byte Pair Encoding**은 문자 대신 바이트 단위로 시퀀스를 구성하며,   
  유니코드 문자에 대해 바이트로 변환된 쌍에 대한 어휘 사전을 구축해 다국어 설정에서 유용합니다.
- **WordPiece**의 경우 BPE와 유사하지만, 빈도수 대신 likelihood를 기준으로 기호 쌍을 병합합니다.
- BPE와 같은 방식은 어휘 사전에 없는 단어를 하위 단어로 분할하기 때문에, OOV 처리에 효과적입니다.

### Pre-training
- **언어 모델링**은 일반적으로 임의의 문장이 주어지고 단어를 순서대로 보면서 다음 단어를 예측하도록 학습시키는데,   
  공백 문자에 대해 전방(왼쪽에서 오른쪽)과 후방(오른쪽에서 왼쪽)으로 예측하는 **자동 회귀 언어 모델링**과,   
  예측을 하면서 양방향으로 문장을 읽는 **자동 인코딩 언어 모델링**이 있습니다.
- BERT는 주어진 입력 문장에서 전체 단어의 15%를 무작으로 마스킹하고 마스킹된 단어를 예측하는,   
  **Masked Language Modeling**을 사용해 학습합니다.
- [MASK] 토큰을 사전 학습시킬 경우 fine-tuning 시 입력에 [MASK] 토큰이 없어 불일치가 발생할 수 있기 때문에,   
  15% 토큰 중에서 80%에 대해서만 [MASK] 토큰으로 교체하고, 10%는 임의의 토큰으로, 10%는 변경하지 않는 전략을 취합니다.
- **Whole Word Masking** 기법을 통해 하위 단어가 마스킹되면 해당 단어와 관련된 모든 단어를 마스킹하며,   
  해당하는 마스킹 비율이 15%를 초과하면 다른 단어의 마스킹을 무시합니다.
- **Next Sentence Prediction**은 두 문장에 대해 어느 것이 다음 문장인지 예측하는 것으로,   
  B 문장이 A에 이어질 경우 `isNext`를 반환하고, 그렇지 않으면 `notNext`를 반환합니다.
- NSP는 [CLS] 토큰 표현에 소프트맥스 함수와 FFN을 거쳐 두 클래스에 대한 확률값을 반환하는 방식으로 처리되는데,   
  [CLS] 토큰이 모든 토큰의 집계 표현을 보유하고 있기 때문에 문장 전체에 대한 표현을 담고 있다고 판단하는 것입니다.
- 사전 학습은 warm-up으로 1만 스텝을 학습하며, dropout 0.1, GELU 활성화 함수를 사용합니다.

### References
- [구글 BERT의 정석, Sudharsan Ravichandiran](https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=281761569&start=slayer)

---

## BERT Paper

> **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**   
> 2018 · Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

<a href="https://arxiv.org/pdf/1810.04805.pdf"><button type="button" class="btn btn-primary">Paper Link</button></a>

### 1. Introduction

- 언어 모델 사전학습은 문장 간의 관계를 예측하는 것을 목적으로 다양한 자연어 처리 작업에 효과적으로 기여했습니다.
- 사전학습된 언어 표현을 적용하는데 있어서 두 가지 전략(*feature-based*, *fine-tuning*)이 존재합니다.
- **Feature-based approach(ELMo)**는 사전학습된 표현의 특징을 포함하는 task-specific 아키텍처를 사용합니다.
- **Fine-tuning approach(OpenAI GPT)**는 모든 사전학습된 파라미터를 fine-tuning합니다.
- 두 접근 방식은 모두 단방향으로 언어 표현을 학습하는 동일한 목적 함수를 이용하는데,   
이는 문장 수준의 작업에만 최적화되는 제한을 걸어 토큰 수준의 접근 방식에 적용하기 어려울 수 있습니다.
- 해당 논문에서는 Masked Language Model(MLM)을 사용하는 양방향 학습 기법 **BERT**를 소개합니다.
- **MLM**은 문장 내 일부 토큰을 랜덤하게 가려서 원본 단어를 예측하게 하는 목적 함수입니다.

### 2. Related Work

#### 2.1. Unsupervised Feature-based Approaches

- Word embedding 및 문맥상 잘못된 단어를 올바르게 식별하기 위해 left-to-right 언어 모델링을 적용했습니다.
- ELMo는 LSTM을 통해 left-to-right 및 right-to-left 표현에서 문맥적 특징을 파악하여,   
question answering, sentiment analysis, named entity recognition 등에서 성능 향상을 일으켰습니다.

#### 2.2. Unsupervised Fine-tuning Approaches

- 분류되지 않은 텍스트로부터 word embedding 파라미터를 사전학습 하였습니다.
- 처음부터 소수의 파라미터만 요구된다는 장점을 가져 문장 수준 작업에서 성능 향상을 일으켰습니다.

#### 2.3. Transfer Learning from Supervised Data

- 방대한 데이터셋을 가지고 지도 학습을 거친 작업에 대한 전이 학습을 거치는 연구도 존재합니다.
- Computer Vision 분야에서 ImageNet과 같이 사전 학습된 모델에 대한 전이 학습의 중요성을 알렸습니다.

### 3. BERT

<img src="../.media/bert/figure_1.png">

- 분류되지 않은 데이터를 가지고 사전 학습을 진행한 후,   
분류된 데이터를 활용해 모든 파라미터를 fine-tuning 하는 작업을 거쳤습니다.

#### Model Architecture

- [Vaswani 등이 발표](https://arxiv.org/abs/1706.03762)한 구조를 기반으로 다층 양방향 변환 인코더를 적용시킨 아키텍처입니다.
- 레이어의 개수에 따라 $BERT_{BASE}$와 $BERT_{LARGE}$로 구분했습니다.
- $BERT_{BASE}$는 OpenAI GPT와 같은 사이즈를 가지고 있지만 양방향 방식을 사용했습니다.

#### Input/Output Representations

- 단일 문장과 질문/답변 등 짝지어진 문장을 확실하게 표현할 수 있는 데이터를 다루게 했습니다.
- 30000개의 토큰 단어를 가지고 있는 WordPiece embedding을 사용했습니다.
- 모든 문장의 첫번째 토큰은 분류 토큰 `[CLS]`, 마지막 토큰은 A 또는 B 문장에 속하는지 알리는 토큰 `[SEP]`입니다.
- 대응되는 토큰, segment와 position embedding을 합산해 입력 데이터의 표현을 구성했습니다.

#### Task #1: Masked LM

- 일반적인 언어 모델에서 양방향 조건을 적용하면 각 단어의 방향성을 상실하여 예측 성능이 떨어지기 때문에,   
일정 비율의 토큰을 랜덤하게 숨겨서 이를 예측하도록 할 필요가 있습니다.
- WordPiece의 각 문장에서 15%의 토큰을 가리고 해당 단어들만 예측하도록 설정했습니다.
- Fine-tuning에서 [MASK] 토큰이 나타나지 않아 사전 학습된 것과 맞지 않는 문제가 생길 수 있기 때문에,   
[MASK] 토큰을 가지고 있는 단어 중에서 일부만 가리도록 적용시켜서 cross entropy loss를 파악했습니다.

#### Task #2: Next Sentence Prediction (NSP)

- 대다수의 NLP 과제는 두 문장 간의 관계를 파악하는데 집중하기 때문에,   
어떠한 단일 언어 말뭉치로부터 다음 문장을 예측할수 있도록 사전 학습을 진행했습니다.

#### Pre-training data

- 실존하는 문학적 표현을 학습시키기 위해 BooksCorpus 및 Wikipedia의 단어를 사용했습니다.
- Wikipedia로부터는 리스트나 테이블 등은 무시한채 구절만 추출했습니다.

#### Fine-tuning BERT

- 일반적으로는 양방향 관계를 확인하기 전에 텍스트 짝을 각각 독립적으로 인코딩하지만,   
BERT는 두 텍스트를 결합하여 인코딩을 진행했습니다.
- Pre-training과 비교해서, fine-tuning은 상대적으로 시간 등의 자원을 덜 소비합니다.

### 4. Experiments

#### 4.1. GLUE

- GLUE benchmark는 다양한 자연어를 얼마나 잘 이해하는지 평가하는 기준입니다.
- Batch size 32, 3 epochs를 적용해 GLUE 작업을 진행했을 때,   
$BERT_{LARGE}$의 fine-tuning이 불안정적인 것을 확인하고 데이터를 shuffle하여 랜덤하게 재시작했습니다.
- $BERT_{BASE}$와 $BERT_{LARGE}$ 모두 기존의 SOTA와 비교했을 때,   
4.5%에서 7.0% 정도의 평균 정확도 상승을 확인했습니다.
- 매우 작은 학습 데이터를 가지고도 $BERT_{LARGE}$가 $BERT_{BASE}$보다 매우 높은 성능을 보였습니다.

    <img src="../.media/attention/table_1.png" width="80%">

#### 4.2. SQuAD v1.1

- SQuAD v1.1은 대중으로부터 수집한 100k 개의 질문/답변 모음으로, 답변 텍스트를 예측하기 위한 목적을 가집니다.
- 목적 함수는 문장의 시작부터 끝 부분의 정확성에 대한 로그 log-likelihoods 합을 사용하며,   
3 epochs, learning rate 5e-5, batch size 32를 가지고 fine-tuning을 진행했습니다.
- Ensemble에서는 +1.5 F1, 단일 BERT에서는 +1.3 F1의 상승을 가져왔으며,   
TriviaQA fine-tunning 데이터를 제외한 경우에서는 0.1-0.4 F1의 손실만 발생시켰습니다.

#### 4.3. SQuAD v2.0

- SQuAD 1.1에 주어진 문장에서 짧은 답변이 아닐 수도 있는 가능성을 부여한 확장된 작업입니다.
- TriviaQA를 사용하지 않고, 2 epochs, learning rate 5e-5, batch size 48을 적용하여 fine-tuning을 수행했습니다.
- 기존의 결과와 비교했을 때 +5.1 F1의 상승을 확인했습니다.

   <img src="../.media/attention/table_2.png" width="80%">

   <img src="../.media/attention/table_3.png" width="80%">

#### 4.4. SWAG

- SWAG 데이터셋은 113k 개의 완성된 문장이 짝지어진 것으로,   
네 가지 선택지 중에서 가장 그럴듯한 다음 문장을 찾기 위한 목적을 가집니다.
- 3 epochs, learning rate 2e-5, batch size 16을 가지고 fine-tuning 했을 때,   
$BERT_{LARGE}$가 ESIM+ELMo보다 +27.1%, OpenAI GPT보다는 +8.3%의 향상을 보여주었습니다.

   <img src="../.media/attention/table_4.png" width="80%">

### 5. Ablation Studies

#### 5.1. Effect of Pre-training Tasks

- SQuAD에서 LTR 모델은 토큰 예측 능력이 떨어지기 때문에,   
랜덤하게 초기화된 BiLSTM을 상단에 적용하여 SQuAD의 결과를 개선시켰습니다.
- ELMo처럼 LTR과 RTL 모델을 각각 학습하고 합치는 것이 가능하다는 것을 인식했지만,   
단일 양방향 모델에 비해 두 배의 비용이 나가고, 모든 레이어에서 좌우 문맥을 사용해 성능도 떨어지는 문제가 있습니다.

#### 5.2. Effect of Model Size

- GLUE 작업에서 확인되는 것과 같이, 작은 데이터셋을 포함하는 모든 경우에 대해   
큰 사이즈의 모델이 높은 정확도 향상을 나타냈습니다.

### 6. Conclusion

- 언어 모델의 전이 학습을 통한 발전은 비지도 사전 학습이 언어를 이해하는 시스템에서 중요하다는 것을 증명했습니다.
- 적은 자원을 활용한 작업에서도 깊은 양방향 아키텍처가 장점을 가진다는 결과를 제시했습니다.
- 사전 훈련된 모델이 광범위한 NLP 작업을 성공적으로 수행하는데 기여했다 판단했습니다.

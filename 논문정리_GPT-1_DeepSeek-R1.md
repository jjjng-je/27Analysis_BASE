# 논문 리뷰: GPT-1 & DeepSeek-R1

> **다루는 논문**
> 1. Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* (OpenAI, GPT-1)
> 2. DeepSeek-AI (2025/2026), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (arXiv:2501.12948v2)

두 논문은 "LLM 학습 패러다임"의 두 변곡점에 해당한다.
GPT-1은 **비지도 사전학습 + 지도 파인튜닝**이라는 표준 레시피를 확립했고,
DeepSeek-R1은 그 다음 단계인 **사후학습(post-training)에서 순수 RL만으로 추론 능력을 유도**할 수 있음을 보였다.

---

## 1. GPT-1 — Improving Language Understanding by Generative Pre-Training

### 1.1 문제의식
- NLP 태스크(함의 추론, QA, 의미 유사도, 분류)마다 **라벨 데이터가 희소**하다.
- 반면 **라벨 없는 텍스트 코퍼스는 대량으로 존재**한다.
- 기존 word embedding은 단어 수준 정보만 전이되어 상위 수준 의미를 못 담는다.
- 두 가지 미해결 이슈:
  1. 전이에 유용한 표현을 배우려면 **어떤 목적함수**가 좋은지 합의가 없음
  2. 학습된 표현을 타깃 태스크로 **어떻게 전이**할지 합의가 없음

### 1.2 방법론: 2단계 학습

**(1) 비지도 사전학습 (Unsupervised Pre-training)**

일반적인 언어모델링 목적함수로 다음 토큰을 예측한다.

$$L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)$$

- 모델: **Transformer decoder 12층** (masked self-attention)
- 은닉차원 768, attention head 12개, FFN inner 3072
- 학습 데이터: **BooksCorpus** (미출간 도서 7,000권 이상)
  - 핵심 이유: **긴 연속 텍스트**가 있어 장거리 의존성 학습이 가능
  - 비교 대상인 1B Word Benchmark는 문장 단위로 셔플되어 장거리 구조가 파괴됨

**(2) 지도 파인튜닝 (Supervised Fine-tuning)**

마지막 블록 활성값 $h_l^m$ 위에 선형층 $W_y$만 추가:

$$P(y \mid x^1,\dots,x^m) = \text{softmax}(h_l^m W_y)$$

보조 LM 목적함수를 함께 최적화 (가중치 $\lambda = 0.5$):

$$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C})$$

→ 일반화 성능 향상 + 수렴 가속 효과.

**(3) Task-specific Input Transformation** ★ 이 논문의 핵심 아이디어

태스크마다 아키텍처를 새로 설계하지 않고, **구조화된 입력을 하나의 연속 토큰 시퀀스로 변환**한다.

| 태스크 | 입력 변환 |
|---|---|
| 분류 | `[start] text [extract]` |
| 함의(Entailment) | premise + `$`(구분자) + hypothesis |
| 유사도(Similarity) | 두 순서를 모두 만들어 독립 처리 후 표현을 **element-wise 합산** |
| QA / 상식추론 | `[z; q; $; a_k]` 형태로 답안별 시퀀스를 만들고 softmax로 정규화 |

→ 추가되는 파라미터는 사실상 $W_y$와 구분자 임베딩뿐.

### 1.3 주요 결과
- **12개 데이터셋 중 9개에서 SOTA 달성** (앙상블 모델도 다수 능가)
- Story Cloze +8.9%p, RACE +5.7%p, MultiNLI +1.5%p, GLUE 68.9 → **72.8**
- CoLA 35.0 → **45.4** (문법성 판단에서 큰 폭 향상 = 내재적 언어 편향 학습)
- 약점: RTE(2,490개, 소규모 데이터셋)에서는 56.0%로 멀티태스크 BiLSTM(61.7%)보다 낮음

### 1.4 분석 (Analysis) — 시험에 잘 나올 파트
- **전이 레이어 수 실험**: 전이하는 층이 많을수록 성능이 단조 증가, MultiNLI 기준 최대 9%까지 향상
  → 각 층이 타깃 태스크에 유용한 기능을 담고 있다는 의미
- **Zero-shot 실험**: 파인튜닝 없이 휴리스틱만으로 태스크 수행 시, 사전학습 업데이트가 늘수록 성능이 꾸준히 상승
  → 생성 사전학습이 태스크 관련 기능을 자연스럽게 습득함을 시사
  → LSTM은 분산(variance)이 커서 Transformer의 inductive bias가 전이에 유리함을 보임
- **Ablation (Table 5)**
  | 조건 | 평균 점수 |
  |---|---|
  | Transformer + aux LM (full) | 74.7 |
  | Transformer w/o aux LM | 75.0 |
  | LSTM + aux LM | 69.1 |
  | **사전학습 없음** | **59.9** |
  - 사전학습 제거 시 14.8% 하락 → **사전학습이 가장 결정적**
  - aux LM은 큰 데이터셋에서 유리, 작은 데이터셋에서는 오히려 손해
  - Transformer → LSTM 교체 시 평균 5.6점 하락

---

## 2. DeepSeek-R1 — Incentivizing Reasoning Capability via RL

### 2.1 문제의식
- 기존 추론 능력 강화는 **사람이 작성한 CoT 데이터(annotated reasoning trace)** 에 의존
- 이 방식의 한계:
  1. 확장성 부족 + 사람의 인지 편향 주입
  2. 사람의 사고 과정을 모방하도록 강제하면 **성능 상한이 사람 수준에 묶임**
- 질문: **사람 라벨 없이, 보상만으로 추론 능력을 "유도(incentivize)"할 수 있는가?**

### 2.2 DeepSeek-R1-Zero: 순수 RL

- 베이스: DeepSeek-V3-Base
- **SFT 단계를 완전히 생략**하고 곧바로 RL 적용 (핵심 설계 선택)
- 알고리즘: **GRPO (Group Relative Policy Optimization)**

**GRPO 핵심**: PPO의 value network(critic)를 없애고, 같은 질문에 대해 그룹으로 $G$개 응답을 샘플링한 뒤 **그룹 내 보상의 표준화 값**을 advantage로 사용한다.

$$A_i = \frac{r_i - \text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\})}$$

→ 별도 critic 모델이 필요 없어 자원 소모가 크게 줄어든다.

**보상 설계 (rule-based)**

$$Reward_{rule} = Reward_{acc} + Reward_{format}$$

- **정확도 보상**: 수학은 정답 박스 형식 검증, 코드는 컴파일러 + 테스트케이스로 객관적 검증
- **형식 보상**: 사고 과정을 `<think>...</think>`, 답변을 `<answer>...</answer>` 안에 넣도록 유도
- ⚠️ **신경망 보상 모델(neural RM)을 의도적으로 쓰지 않음** → 대규모 RL에서 **reward hacking**에 취약하기 때문

**주요 하이퍼파라미터**: lr 3e-6, KL 계수 0.001, temperature 1, 질문당 16개 샘플링, batch size 512, 총 10,400 스텝(약 1.6 epoch), 8.2k 스텝 이후 최대 길이 32,768 → 65,536 토큰

### 2.3 창발(emergence) 현상 ★ 이 논문의 하이라이트

- **AIME 2024 pass@1: 15.6% → 77.9%**, self-consistency(cons@16) 적용 시 **86.7%** (인간 참가자 평균 초과)
- **응답 길이가 학습 중 자연스럽게 증가** — 외부 개입 없이 "더 오래 생각하는" 방향으로 스스로 적응
- 반성적 추론(reflection), 검증(verification), 대안 탐색 같은 고급 전략이 **가르치지 않았는데 자발적으로 출현**
- **"aha moment"**: 중간 체크포인트에서 모델이 "Wait, wait. Wait." 하며 스스로 되짚는 순간이 등장, 이 시점에 "wait" 사용 빈도가 급증

> 핵심 메시지: 문제 푸는 법을 **가르치는(teaching)** 게 아니라, 올바른 **인센티브를 주면(incentivizing)** 모델이 스스로 전략을 개발한다.

### 2.4 DeepSeek-R1: 다단계 파이프라인

R1-Zero의 문제 — **가독성 저하**, **언어 혼용**(영어/중국어 섞임), 추론 외 영역(글쓰기·개방형 QA) 성능 부족.
이를 해결하기 위한 4단계 파이프라인:

| 단계 | 내용 | 산출 |
|---|---|---|
| 1 | **Cold-start SFT** — 대화체·인간정렬 long CoT 수천 건으로 SFT | R1 Dev-1 |
| 2 | **1차 RL** — 추론 프롬프트 + rule-based 보상 + **언어 일관성 보상** | R1 Dev-2 |
| 3 | **Rejection sampling + SFT** — 추론/비추론 데이터 모두 투입 | R1 Dev-3 |
| 4 | **2차 RL** — 다양한 프롬프트 + 선호(preference) 보상으로 helpfulness·harmlessness 정렬 | **DeepSeek-R1** |

**언어 일관성 보상**

$$Reward_{language} = \frac{Num(Words_{target})}{Num(Words)}$$

- ablation 결과 성능은 **미세하게 하락**하지만, 사람 선호(가독성)에 부합하므로 채택 → **성능 vs 정렬의 트레이드오프**를 명시적으로 감수한 사례

**보상 모델 (2차 RL)**
- Helpful RM: DeepSeek-V3로 선호쌍 생성, 위치 편향 완화를 위해 4회 질의 후 평균, 길이 편향 완화 위해 응답 길이 맞춤, 총 66,000쌍
- Safety RM: 106,000개 프롬프트를 safe/unsafe로 라벨링, point-wise 학습
- 최종 보상: $Reward = Reward_{reasoning} + Reward_{general} + Reward_{language}$

### 2.5 단계별 성능 (Table 3)

| Benchmark | R1-Zero | Dev-1 | Dev-2 | Dev-3 | **R1** |
|---|---|---|---|---|---|
| AIME 2024 (Pass@1) | 77.9 | 59.0 | 74.0 | 78.1 | **79.8** |
| MATH-500 | 95.9 | 94.2 | 95.9 | 95.4 | **97.3** |
| Codeforces (Rating) | 1444 | 1534 | 1687 | 1746 | **2029** |
| IF-Eval | 46.6 | 71.7 | 72.0 | 78.1 | **83.3** |
| AlpacaEval 2.0 | 24.7 | 50.1 | 55.8 | 62.1 | **87.6** |
| ArenaHard | 53.6 | 77.0 | 73.2 | 75.6 | **92.3** |

**읽는 법**
- Dev-1: 지시 따르기(IF-Eval)는 크게 오르지만 cold-start 데이터가 적어 **AIME는 오히려 하락** (77.9 → 59.0)
- Dev-2: 추론 중심 RL로 수학·코드 회복 및 향상, 일반 태스크는 소폭 변화
- Dev-3: 비추론 데이터 투입으로 일반 생성·코드 엔지니어링 개선
- 최종 R1: 수학·코드는 소폭 상승(이미 앞 단계에서 포화), **AlpacaEval +25%, ArenaHard +17%** — 즉 마지막 RL의 기여는 주로 **사용자 선호 정렬**

### 2.6 증류(Distillation)
- R1이 만든 800k 데이터로 소형 모델(Qwen 1.5B/7B/14B/32B, Llama 8B/70B)을 2~3 epoch 파인튜닝
- 결과: **원본 instruction-tuned 모델을 상회** → 대형 모델의 창발적 추론 패턴을 소형 모델로 이전 가능

### 2.7 한계 (논문이 직접 밝힌 부분)
- **구조화 출력 / 도구 사용**: 검색·계산기 등 툴 사용 불가
- **토큰 효율**: 쉬운 문제에 과도하게 생각하는 overthinking 잔존
- **언어 혼용**: 중/영 최적화 → 제3언어 질의 시 영어로 추론하는 경향
- **프롬프트 민감성**: few-shot이 오히려 성능을 떨어뜨림 → **zero-shot으로 문제와 출력 형식만 명시하는 것을 권장**
- **소프트웨어 공학 태스크**: 평가 시간이 길어 대규모 RL 미적용, V3 대비 개선 폭 작음
- **Reward Hacking**: 글쓰기처럼 신뢰할 만한 규칙 기반 검증이 불가능한 영역에서는 순수 RL 확장이 여전히 미해결 과제
- **안전성**: 자체 안전 수준은 GPT-4o 수준의 "보통", 외부 리스크 관리 시스템과 결합해야 우수 등급

---

## 3. 두 논문 비교

| 구분 | GPT-1 (2018) | DeepSeek-R1 (2025) |
|---|---|---|
| 해결 대상 | 라벨 데이터 부족 | 사람 CoT 라벨 의존 / 추론 능력 한계 |
| 학습 구조 | 비지도 사전학습 → 지도 파인튜닝 | (사전학습 완료 가정) → 순수 RL / 다단계 RL |
| 학습 신호 | 다음 토큰 예측 + 태스크 라벨 | **최종 정답 정확성**(규칙 기반 보상) |
| 인간 개입 | 태스크별 라벨 필요 | 최종 정답만 필요, 추론 과정은 자유 |
| 핵심 기법 | Task-specific input transformation | GRPO + rule-based reward |
| 성능 상한 | 인간 라벨 품질에 종속 | 검증기(verifier)가 있으면 인간 초과 가능 |
| 창발 현상 | zero-shot 능력이 점진적으로 상승 | 응답 길이 증가, self-reflection, "aha moment" |

### 관통하는 흐름
1. **인간 감독의 축소**: GPT-1은 라벨을 줄였고(사전학습), R1은 추론 과정 라벨까지 없앴다.
2. **구조 제약의 최소화**: GPT-1은 태스크별 아키텍처 대신 입력 변환으로, R1은 사고 내용 대신 형식(`<think>` 태그)만 제약했다. 둘 다 **"덜 제약할수록 모델이 더 잘한다"** 는 발견.
3. **검증 가능성이 새로운 병목**: R1의 결론은 "확장의 열쇠는 대규모 라벨링이 아니라 **어려운 문제 + 신뢰할 수 있는 검증기 + 충분한 컴퓨팅**"이다. 즉 앞으로의 한계는 데이터가 아니라 **보상을 신뢰할 수 있게 정의할 수 있는가**에 달려 있다.

---

## 4. 한 줄 요약

- **GPT-1**: 라벨 없는 텍스트로 먼저 언어를 배우게 하고, 입력 형태만 바꿔 태스크에 붙이면 태스크별 특화 모델을 이긴다.
- **DeepSeek-R1**: 정답 여부만 보상으로 주면 모델이 스스로 더 오래 생각하고, 되돌아보고, 검증하는 추론 전략을 만들어낸다. 다만 그 전제는 **신뢰할 수 있는 검증기**다.

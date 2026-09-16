# 논문 리뷰: GAN & DDPM

> **다루는 논문**
> 1. Goodfellow et al. (2014), *Generative Adversarial Nets* (Université de Montréal, arXiv:1406.2661)
> 2. Ho et al. (2020), *Denoising Diffusion Probabilistic Models* (UC Berkeley, NeurIPS 2020, arXiv:2006.11239)

둘 다 **"데이터 분포를 어떻게 학습해서 새 샘플을 생성할 것인가"** 를 다루지만, 해법이 정반대다.
GAN은 **암묵적 모델 + 적대적 게임**, DDPM은 **명시적 잠재변수 모델 + 변분 추론**이다.
GAN 이후 6년 만에 DDPM이 GAN을 이미지 품질에서 따라잡은 흐름을 같이 보면 좋다.

---

## 1. GAN — Generative Adversarial Nets

### 1.1 문제의식
- 딥러닝의 성공은 대부분 **판별 모델(discriminative)** 쪽이었다
- 생성 모델이 뒤처진 이유:
  1. 최대우도 추정에 등장하는 **다루기 힘든 확률 계산**(특히 분배함수 partition function)
  2. 생성 맥락에서 ReLU 같은 **구간선형 유닛의 이점을 살리기 어려움**
- 기존 대안들의 한계:
  - RBM, DBM: 분배함수와 그 기울기가 계산 불가 → MCMC로 근사해야 하는데 **mixing이 문제**
  - Score matching, NCE: 확률밀도를 정규화 상수까지 **해석적으로 명시**해야 함
  - GSN: 샘플링에 **마르코프 체인 필요**

→ GAN은 이 모든 걸 우회한다. **확률밀도를 명시하지 않고, 샘플을 뽑는 기계를 직접 학습**한다.

### 1.2 핵심 아이디어

**위조지폐범 vs 경찰 비유**
- 생성자 $G$: 위조지폐를 만들어 들키지 않으려 함
- 판별자 $D$: 진짜와 가짜를 구분하려 함
- 경쟁이 둘 다를 개선시켜 결국 구분이 불가능해진다

**minimax 게임의 가치함수**

$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

- $G(z;\theta_g)$: 노이즈 $z$ 를 데이터 공간으로 매핑하는 MLP
- $D(x;\theta_d)$: $x$ 가 진짜일 확률을 출력하는 스칼라 함수

**중요한 특징**
- 마르코프 체인 불필요, 추론(inference) 불필요
- 학습은 **역전파만**, 샘플링은 **순전파만**

### 1.3 이론적 결과 ★ 이 논문의 핵심

**Proposition 1 — 최적 판별자**

$G$ 가 고정일 때 최적 $D$ 는

$$D_G^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

(증명: $a\log y + b\log(1-y)$ 는 $y = \frac{a}{a+b}$ 에서 최대)

**Theorem 1 — 전역 최적해**

$D$ 를 최적으로 두고 $V$ 를 정리하면

$$C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{data} \,\|\, p_g)$$

- JSD는 항상 $\geq 0$ 이고 두 분포가 같을 때만 0
- 따라서 **전역 최소값은 $-\log 4$ 이고, 그때 $p_g = p_{data}$**
- 이 시점에 $D(x) = \frac{1}{2}$ — 판별자가 동전 던지기 수준이 됨

> **시험 단골**: "GAN은 어떤 divergence를 최소화하는가?" → **Jensen-Shannon divergence**

### 1.4 실제 학습에서의 트릭

**(1) $k$ 스텝 $D$ / 1 스텝 $G$**
- $D$ 를 매번 완전히 최적화하는 건 계산상 불가능하고 과적합을 부른다
- 실험에서는 $k=1$ (가장 싼 옵션)을 사용

**(2) Non-saturating loss** ★ 실무에서 가장 중요한 부분
- 학습 초기에 $G$ 가 형편없으면 $D$ 가 자신 있게 가짜라고 판정 → $\log(1-D(G(z)))$ 가 **포화(saturate)** 되어 기울기가 거의 안 흐름
- 해결: $\log(1-D(G(z)))$ 를 **최소화**하는 대신 $\log D(G(z))$ 를 **최대화**
- 고정점(fixed point)은 같지만 초기 기울기가 훨씬 강하다

### 1.5 실험
- 데이터셋: MNIST, TFD(Toronto Face Database), CIFAR-10
- 평가: **Parzen window 기반 로그우도 추정** — 명시적 밀도가 없어서 어쩔 수 없이 쓴 방법
- MNIST 225±2, TFD 2057±26 (Deep GSN 214, DBN 138 대비)
- 논문 스스로 "이 평가법은 분산이 크고 고차원에서 잘 안 통하지만 현재로선 최선"이라고 인정
- 생성 모델 평가 방법 자체가 열린 문제라고 제기 → 이후 IS, FID가 등장하는 배경

### 1.6 장단점 (논문이 직접 정리)

**장점**
- 마르코프 체인 불필요, 역전파만으로 기울기 획득
- 학습 중 추론 불필요
- 미분 가능한 함수면 뭐든 넣을 수 있음
- **매우 날카로운(sharp), 심지어 퇴화된 분포까지 표현 가능** — MCMC 기반은 체인이 모드 간 이동하려면 분포가 흐릿해야 함

**단점**
- $p_g(x)$ 의 **명시적 표현이 없다**
- $D$ 와 $G$ 의 **동기화가 까다롭다**
- **"Helvetica scenario"** — $G$ 를 $D$ 갱신 없이 너무 많이 학습시키면 여러 $z$ 를 같은 $x$ 로 붕괴시켜 다양성을 잃음. 이후 **mode collapse**로 널리 불리는 문제

### 1.7 후속 연구 방향 (논문이 제시)
조건부 생성 $p(x|c)$, 학습된 근사 추론, 준지도학습에 판별자 특징 활용 등 — 실제로 cGAN, BiGAN 등으로 이어졌다.

---

## 2. DDPM — Denoising Diffusion Probabilistic Models

### 2.1 문제의식
- Diffusion model 자체는 2015년 Sohl-Dickstein이 제안(비평형 열역학에서 착안)
- 정의도 간단하고 학습도 효율적이지만, **고품질 샘플을 만든 사례가 없었다**
- DDPM의 기여: **실제로 GAN급 이미지를 만들 수 있음을 보이고**, 그 비결인 파라미터화를 제시

### 2.2 구조 — 두 개의 마르코프 체인

**Forward process (diffusion)** — 학습 파라미터 없음, 고정

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I)$$

데이터에 점진적으로 가우시안 노이즈를 더해 신호를 파괴한다.

**핵심 성질**: 임의의 시점 $t$ 를 **닫힌 형태로 바로 샘플링 가능**

$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t)I), \quad \alpha_t = 1-\beta_t,\ \bar\alpha_t = \prod_{s=1}^{t}\alpha_s$$

→ $t$ 를 랜덤하게 뽑아 학습할 수 있어서 **효율적인 SGD 학습이 가능**해진다. 이게 없으면 1000스텝을 다 돌려야 함.

**Reverse process** — 학습 대상

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t),\ \Sigma_\theta(x_t,t))$$

$\beta_t$ 가 충분히 작으면 역과정도 가우시안으로 둘 수 있다는 게 이론적 근거.

### 2.3 학습 목적함수

변분 하한(ELBO)을 세 항으로 분해:

$$L = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t>1} \underbrace{D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} \underbrace{- \log p_\theta(x_0|x_1)}_{L_0}$$

- $L_T$: $\beta_t$ 를 상수로 고정했으므로 **학습 중 상수 → 무시**
- $L_{t-1}$: $q(x_{t-1}|x_t,x_0)$ 가 가우시안이라 **KL을 닫힌 형태로 계산** 가능 (몬테카를로 추정 불필요 → 분산 감소)
- $L_0$: 이산 디코더로 처리 ([0,255] → [-1,1] 스케일링과 연결)

### 2.4 ε-예측 파라미터화 ★ 논문의 핵심 기여

$\mu_\theta$ 를 직접 예측하는 대신, **더해진 노이즈 $\epsilon$ 을 예측**하도록 재파라미터화한다.

$$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

그러면 목적함수가 아름답게 단순해진다:

$$L_{simple}(\theta) = \mathbb{E}_{t,x_0,\epsilon}\left[\left\| \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\ t) \right\|^2\right]$$

**그냥 노이즈에 대한 MSE다.** 이게 전부.

**두 가지 연결고리**
1. 이 식은 **여러 노이즈 레벨에 대한 denoising score matching**과 같은 형태 (Song & Ermon의 NCSN)
2. 샘플링 절차는 **annealed Langevin dynamics**를 닮았다
→ "변분 추론으로 Langevin 유사 샘플러를 직접 학습한다"는 해석이 성립

**가중치를 버린 게 오히려 이득인 이유**
- $L_{simple}$ 은 원래 있던 $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$ 가중치를 버린 형태
- 결과적으로 **작은 $t$(노이즈가 거의 없는 쉬운 구간)의 손실을 down-weight**
- 네트워크가 **큰 $t$ 의 어려운 디노이징에 집중**하게 되어 샘플 품질이 좋아진다

### 2.5 알고리즘 (외워둘 만큼 짧다)

**Training**
1. $x_0 \sim q(x_0)$
2. $t \sim \text{Uniform}(\{1,...,T\})$
3. $\epsilon \sim \mathcal{N}(0,I)$
4. $\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2$ 에 대해 경사하강

**Sampling**
1. $x_T \sim \mathcal{N}(0,I)$
2. $t = T$ 부터 1까지: $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right) + \sigma_t z$

### 2.6 실험 설정
- $T = 1000$, $\beta_t$ 는 $10^{-4}$ 에서 $0.02$ 까지 **선형 증가**
- 네트워크: **U-Net** (PixelCNN++ 백본, group normalization, 16×16 해상도에 self-attention)
- 시점 $t$ 는 **Transformer의 사인파 위치 임베딩**으로 각 residual block에 주입 → 모든 $t$ 가 파라미터를 공유
- CIFAR10 모델 35.7M 파라미터, LSUN/CelebA-HQ 114M

### 2.7 결과

**CIFAR10 (Table 1)**

| Model | IS ↑ | FID ↓ |
|---|---|---|
| NCSN | 8.87 | 25.32 |
| SNGAN | 8.22 | 21.7 |
| BigGAN (conditional) | 9.22 | 14.73 |
| **Ours (L_simple)** | **9.46** | **3.17** |

- 무조건부(unconditional) 모델인데 **조건부 모델까지 능가**
- LSUN 256×256에서 ProgressiveGAN급 품질 (Bedroom FID 4.90, Church 7.89)

**핵심 트레이드오프**: 로그우도(NLL)는 오히려 나쁘다 (≤3.75 bits/dim). 진짜 변분 하한으로 학습하면 코드길이는 좋아지지만 **샘플 품질은 $L_{simple}$ 이 압도적**이다.

**Ablation (Table 2)** ★ 두 선택지가 왜 함께 가야 하는지 보여줌

| 파라미터화 | 목적함수 | FID |
|---|---|---|
| $\tilde\mu$ 예측 | L, 고정 Σ | 13.22 |
| $\tilde\mu$ 예측 | 단순 MSE | 학습 불안정 |
| $\epsilon$ 예측 | L, 고정 Σ | 13.51 |
| **$\epsilon$ 예측** | **$L_{simple}$** | **3.17** |

- $\epsilon$ 예측만으로는 $\tilde\mu$ 예측과 비슷한 수준
- **$\epsilon$ 예측 + 단순 목적함수 조합**일 때만 급격히 좋아진다
- 분산 $\Sigma$ 를 학습시키면 **불안정**해짐 → 고정이 낫다

### 2.8 추가 분석

- **Progressive lossy compression**: 코드길이를 rate(1.78 bits/dim)와 distortion(1.97 bits/dim)으로 분해하면 **절반 이상이 사람 눈에 안 보이는 왜곡을 기술하는 데 쓰인다** → diffusion은 우수한 손실 압축기라는 귀납 편향을 가진다
- **Progressive generation**: 역과정 초반에 **큰 스케일 특징**이 먼저 나타나고 세부가 마지막에 나타난다
- **자기회귀 모델과의 연결**: forward process를 "좌표를 하나씩 마스킹"으로 정의하면 자기회귀 모델이 된다 → Gaussian diffusion은 **일반화된 비트 순서를 가진 자기회귀 모델**로 해석 가능
- **보간(interpolation)**: 잠재공간에서 선형 보간 후 역과정으로 디코딩. $t$ 가 클수록 거칠고 다양한 보간, 작을수록 세부만 유지

---

## 3. 두 논문 비교

| 구분 | GAN (2014) | DDPM (2020) |
|---|---|---|
| 모델 유형 | 암묵적(implicit) | 명시적 잠재변수 모델 |
| 학습 방식 | 두 네트워크의 minimax 게임 | 단일 네트워크의 변분 하한 최적화 |
| 손실함수 | 적대적 손실 (JSD 최소화) | 노이즈에 대한 MSE |
| 학습 안정성 | **불안정** (동기화, mode collapse) | **안정** (지도학습 회귀와 다름없음) |
| 샘플링 비용 | **1회 순전파** | **T=1000회 순전파** |
| 우도 계산 | 불가 | 가능 (변분 하한) |
| 다양성 | mode collapse 위험 | 전 모드 커버에 강함 |
| CIFAR10 FID | SNGAN 21.7 / BigGAN 14.73 | **3.17** |

### 관통하는 흐름
1. **GAN이 남긴 문제를 DDPM이 해결했다.** 학습 불안정성과 mode collapse가 GAN의 고질병이었는데, DDPM은 목적함수가 그냥 MSE라 안정적이고 모드 붕괴도 구조적으로 덜하다.
2. **대신 비용을 옮겼다.** GAN은 학습이 어렵고 샘플링이 싸다. DDPM은 학습이 쉽고 샘플링이 1000배 비싸다. 이후 DDIM, latent diffusion 같은 연구가 전부 **이 샘플링 비용을 줄이는 방향**이다.
3. **평가 지표의 역사도 같이 읽힌다.** GAN 논문은 Parzen window라는 조악한 방법밖에 없다고 한탄했고, DDPM 시점엔 IS와 FID가 표준이 되어 있다.
4. **"우도가 좋은 모델"과 "샘플이 좋은 모델"은 다르다**는 게 DDPM의 흥미로운 발견이다. NLL은 나쁜데 FID는 최고다. 생성 모델 평가가 단일 척도로 환원되지 않는다는 증거다.

---

## 4. 한 줄 요약

- **GAN**: 생성자와 판별자를 경쟁시키면, 확률밀도를 한 번도 명시하지 않고도 데이터 분포를 학습할 수 있다. 이론적으로는 JSD를 최소화하며 $p_g = p_{data}$ 에서 유일한 최적해를 갖는다.
- **DDPM**: 노이즈를 점진적으로 더하는 과정을 되돌리도록 학습하되, 평균이 아니라 **더해진 노이즈를 예측**하게 하고 가중치를 버린 단순 MSE로 학습하면, GAN을 넘는 이미지 품질이 나온다.

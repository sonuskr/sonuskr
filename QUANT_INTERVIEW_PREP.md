# 📈 Quant Interview Prep — Probability & Statistics for Trading
> From Zero → Interview Ready → Real Trading Thinking

---

## 🗺️ Table of Contents

1. [What is a Random Variable?](#1-what-is-a-random-variable)
2. [Expected Value — Your Edge](#2-expected-value--your-edge)
3. [Variance & Standard Deviation — Your Risk](#3-variance--standard-deviation--your-risk)
4. [Key Distributions](#4-key-distributions)
5. [Conditional Probability](#5-conditional-probability)
6. [Bayes' Theorem](#6-bayes-theorem)
7. [Mean vs Median vs Mode](#7-mean-vs-median-vs-mode)
8. [Law of Large Numbers](#8-law-of-large-numbers)
9. [Central Limit Theorem](#9-central-limit-theorem)
10. [Correlation & Covariance](#10-correlation--covariance)
11. [Sharpe Ratio](#11-sharpe-ratio)
12. [Monte Carlo Simulation](#12-monte-carlo-simulation)
13. [Volatility](#13-volatility)
14. [Practice Problems](#14-practice-problems)
15. [Interview Cheat Sheet](#15-interview-cheat-sheet)
16. [Practice Set 1 — Core Quant Thinking](#16-practice-set-1--core-quant-thinking)
17. [Interview-Style Questions — Real Level](#17-interview-style-questions--real-level)
18. [Mini Coding Task — Monte Carlo in Python](#18-mini-coding-task--monte-carlo-in-python)

---

## 1. What is a Random Variable?

### Simple Definition
A **random variable** is just a number whose value depends on chance.

Think of it like this:
- You don't know tomorrow's stock price → it's random
- You don't know if your trade will profit → it's random

### Two Types

| Type | Meaning | Example |
|------|---------|---------|
| **Discrete** | Countable values | Dice roll: 1,2,3,4,5,6 |
| **Continuous** | Any real value in a range | Stock return: -5.3%, +2.1%, etc. |

### In Trading
```
Stock return tomorrow     → Continuous random variable
Option payoff at expiry   → Can be discrete or continuous
Number of winning trades  → Discrete random variable
```

### Interview Question
> "What type of random variable is a stock's daily return?"

**Answer:** Continuous — because it can take any real value (e.g., +1.23%, -0.87%)

---

## 2. Expected Value — Your Edge

### What It Means
Expected value = **average outcome if you repeat the same trade thousands of times**

This is the most important concept in quant trading. Every trade decision comes down to this.

### Formula

```
E[X] = Σ (value × probability)

For discrete:   E[X] = x₁·p₁ + x₂·p₂ + ... + xₙ·pₙ
For continuous: E[X] = ∫ x · f(x) dx
```

### Example 1 — Basic Trade
```
60% chance: +₹100
40% chance: −₹50

E[X] = (0.6 × 100) + (0.4 × -50)
     = 60 - 20
     = ₹40  ✅ Positive edge → TAKE THIS TRADE
```

### Example 2 — Options Thinking
```
You buy a call option for ₹20 premium.
At expiry:
  50% chance: option worth ₹80  → profit = ₹60
  50% chance: option expires worthless → loss = ₹20

E[X] = (0.5 × 60) + (0.5 × -20)
     = 30 - 10
     = ₹20  ✅ Positive edge
```

### Key Rule
```
E[X] > 0  →  Good trade (positive edge)
E[X] = 0  →  Fair game (casino breaks even)
E[X] < 0  →  Bad trade (avoid)
```

### Properties (Interview Favorites)
```
E[aX + b]    = a·E[X] + b          (linearity)
E[X + Y]     = E[X] + E[Y]         (always true)
E[X · Y]     = E[X]·E[Y]           (ONLY if X,Y independent)
```

### Interview Question
> "A coin flip: heads you win ₹150, tails you lose ₹100. Should you play?"

```
E[X] = (0.5 × 150) + (0.5 × -100)
     = 75 - 50
     = ₹25  ✅ Yes, play it
```

---

## 3. Variance & Standard Deviation — Your Risk

### What It Means
- **Variance** = how spread out your outcomes are = **risk**
- **Std Dev** = square root of variance = same units as your returns

### Formulas
```
Var(X) = E[(X - μ)²]
       = E[X²] - (E[X])²     ← easier to compute

Std Dev (σ) = √Var(X)
```

### Step-by-Step Example
```
Trade outcomes:
  50% chance: +₹100
  50% chance: −₹60

Step 1: Find mean
  μ = E[X] = (0.5 × 100) + (0.5 × -60) = 50 - 30 = ₹20

Step 2: Find variance
  Var(X) = 0.5 × (100-20)² + 0.5 × (-60-20)²
         = 0.5 × 6400 + 0.5 × 6400
         = 3200 + 3200
         = 6400

Step 3: Std Dev
  σ = √6400 = ₹80
```

### In Markets
```
High σ  →  Volatile stock (risky, big swings)
Low σ   →  Stable stock (predictable, small swings)
```

### Properties
```
Var(aX + b) = a²·Var(X)      (constant b doesn't affect spread)
Var(X + Y)  = Var(X) + Var(Y)  (ONLY if X,Y independent)
```

### Interview Question
> "Two strategies: A has E[X]=₹50, σ=₹10. B has E[X]=₹50, σ=₹100. Which do you prefer?"

**Answer:** Strategy A — same expected return but far less risk.

---

## 4. Key Distributions

### 4.1 Normal Distribution (Gaussian) — The King

```
Shape:    Bell curve, symmetric
Defined by: mean (μ) and std dev (σ)
Notation: X ~ N(μ, σ²)
```

**The 68-95-99.7 Rule (MEMORIZE THIS)**
```
μ ± 1σ  →  68% of outcomes
μ ± 2σ  →  95% of outcomes
μ ± 3σ  →  99.7% of outcomes
```

**In Trading:**
```
Daily returns ~ N(0.05%, 1%)  (roughly)

Meaning:
  68% of days: return between -0.95% and +1.05%
  95% of days: return between -1.95% and +2.05%
```

**Why Used:**
- Math is clean and tractable
- Central Limit Theorem justifies it (see section 9)
- Black-Scholes model assumes normal log-returns

**Limitation:**
```
Real markets have FAT TAILS
→ Crashes happen more often than normal distribution predicts
→ 2008 crisis was a "25-sigma event" (should happen once in universe's lifetime)
→ Normal distribution UNDERESTIMATES extreme risk
```

---

### 4.2 Log-Normal Distribution — For Prices

**Why prices can't be normal:**
```
If price ~ Normal → price could go negative ❌
Stock price can't be -₹50
```

**Solution: Model log(returns) as normal**
```
If log(Pₜ/Pₜ₋₁) ~ Normal
Then Pₜ ~ Log-Normal  ✅

Price is always positive ✅
```

**Formula:**
```
If X ~ N(μ, σ²)
Then Y = eˣ ~ LogNormal(μ, σ²)
```

**In Trading:**
```
Black-Scholes assumes: log(S_T/S_0) ~ Normal
→ So S_T ~ Log-Normal
```

---

### 4.3 Uniform Distribution — Simple but Useful

```
Every outcome equally likely
X ~ Uniform(a, b)

E[X]   = (a + b) / 2
Var(X) = (b - a)² / 12
```

**Used in:** Monte Carlo simulations (generating random numbers)

---

### 4.4 Bernoulli & Binomial — Win/Loss Counting

```
Bernoulli: single trade → win (p) or lose (1-p)
Binomial:  n trades → how many wins?

If X ~ Binomial(n, p):
  E[X]   = n·p
  Var(X) = n·p·(1-p)
```

**Example:**
```
60% win rate, 100 trades:
  Expected wins = 100 × 0.6 = 60
  Std Dev = √(100 × 0.6 × 0.4) = √24 ≈ 4.9
```

---

### 4.5 Poisson Distribution — Rare Events

```
Models: number of rare events in a time period
X ~ Poisson(λ)  where λ = average rate

E[X]   = λ
Var(X) = λ
```

**In Trading:**
```
Number of large price jumps per month
Number of defaults in a bond portfolio
```

---

## 5. Conditional Probability

### Definition
```
P(A | B) = probability of A, GIVEN that B already happened

Formula:
P(A | B) = P(A ∩ B) / P(B)
```

### Trading Example
```
Question: What's the probability market goes UP, given inflation dropped?

P(market up | inflation dropped) = P(market up AND inflation dropped) / P(inflation dropped)
```

### Example with Numbers
```
From historical data:
  P(market up)                    = 0.55
  P(inflation dropped)            = 0.40
  P(market up AND inflation drop) = 0.30

P(market up | inflation drop) = 0.30 / 0.40 = 0.75

→ When inflation drops, 75% chance market goes up (vs base 55%)
→ This is your EDGE signal
```

### Independence Check
```
X and Y are independent if:
  P(A | B) = P(A)
  → knowing B gives NO information about A

In trading: if two stocks are truly independent, knowing one's move
            tells you nothing about the other
```

### Interview Question
> "P(A) = 0.4, P(B) = 0.5, P(A∩B) = 0.2. Are A and B independent?"

```
Check: P(A) × P(B) = 0.4 × 0.5 = 0.20
P(A∩B) = 0.20

0.20 = 0.20  ✅  Yes, they are independent
```

---

## 6. Bayes' Theorem

### Formula
```
P(A|B) = [P(B|A) × P(A)] / P(B)
```

### What It Does
Updates your **prior belief** with **new evidence** to get **posterior belief**

```
Prior    →  what you believed before
Evidence →  new data you received
Posterior → updated belief
```

### Trading Example — Crash Prediction
```
Setup:
  P(crash)              = 0.05   (5% base rate of crash)
  P(vol spike | crash)  = 0.80   (80% of crashes have vol spike before)
  P(vol spike)          = 0.15   (vol spikes 15% of the time)

Question: Volatility just spiked. What's P(crash | vol spike)?

P(crash | vol spike) = [P(vol spike | crash) × P(crash)] / P(vol spike)
                     = [0.80 × 0.05] / 0.15
                     = 0.04 / 0.15
                     = 0.267

→ Crash probability jumped from 5% → 26.7% after vol spike!
→ This is how quants update risk models in real time
```

### Classic Interview Problem — Rare Disease Test
```
Disease affects 1% of population
Test is 99% accurate (both ways)

You test positive. What's P(you have disease)?

P(disease)           = 0.01
P(positive | disease) = 0.99
P(positive)          = P(pos|disease)×P(disease) + P(pos|no disease)×P(no disease)
                     = 0.99×0.01 + 0.01×0.99
                     = 0.0099 + 0.0099
                     = 0.0198

P(disease | positive) = (0.99 × 0.01) / 0.0198
                      = 0.0099 / 0.0198
                      = 0.50  ← Only 50%! Surprising right?
```

**Why this matters in trading:**
Even a good signal has many false positives if the event is rare.

---

## 7. Mean vs Median vs Mode

### Definitions
```
Mean   = sum of all values / count  (average)
Median = middle value when sorted
Mode   = most frequently occurring value
```

### Example
```
Returns: -50%, -2%, +1%, +2%, +3%, +4%, +5%

Mean   = (-50 + -2 + 1 + 2 + 3 + 4 + 5) / 7 = -37/7 = -5.3%
Median = +2%  (middle value)
Mode   = no repeat, so no mode
```

### Why Median Matters in Finance
```
One crash (-50%) destroyed the mean
But median (+2%) shows the typical day

→ Median is more robust to outliers
→ Use median for "typical" performance
→ Use mean for expected value calculations
```

### Skewness (Related Concept)
```
Right skew (positive): mean > median  (few large positive outliers)
Left skew (negative):  mean < median  (few large negative outliers)

Stock returns: usually LEFT skewed
→ Most days small gains, occasional large crashes
```

---

## 8. Law of Large Numbers

### Statement
As number of trials → ∞, sample mean → true expected value

```
(X₁ + X₂ + ... + Xₙ) / n  →  E[X]  as n → ∞
```

### Trading Meaning
```
1 trade:      result = mostly luck
10 trades:    still noisy
100 trades:   starting to see the edge
1000 trades:  actual results ≈ expected value
```

### Example
```
Strategy with E[X] = ₹10 per trade, σ = ₹100

After 1 trade:    could be anywhere from -₹100 to +₹100
After 100 trades: average ≈ ₹10 ± ₹10
After 10000 trades: average ≈ ₹10 ± ₹1
```

### Interview Insight
> "Why do casinos always win?"

Because they play millions of hands. Law of Large Numbers guarantees their edge (house edge ~2%) materializes. Same logic applies to quant strategies.

---

## 9. Central Limit Theorem (CLT)

### Statement
Sum (or average) of many independent random variables → Normal distribution, regardless of original distribution

```
If X₁, X₂, ..., Xₙ are i.i.d. with mean μ and variance σ²

Then: (X₁ + X₂ + ... + Xₙ) ~ N(n·μ, n·σ²)  for large n

Or equivalently: X̄ ~ N(μ, σ²/n)
```

### Why It's Powerful
```
Individual stock returns: NOT normal (fat tails, skewed)
Portfolio of 50 stocks:   approximately normal  ✅

→ CLT justifies using normal distribution for portfolios
→ Enables risk calculations (VaR, etc.)
```

### Example
```
Each trade: uniform distribution between -₹100 and +₹100
  Mean = 0, Variance = 10000/3 ≈ 3333

After 100 trades:
  Total P&L ~ N(0, 100 × 3333) = N(0, 333300)
  Std Dev of total = √333300 ≈ ₹577
```

### Rule of Thumb
```
n ≥ 30  →  CLT kicks in (approximately)
n ≥ 100 →  Very good approximation
```

---

## 10. Correlation & Covariance

### Covariance
```
Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)]
           = E[XY] - E[X]·E[Y]

Positive Cov → X and Y move together
Negative Cov → X and Y move opposite
Zero Cov     → No linear relationship
```

### Correlation
```
ρ(X,Y) = Cov(X,Y) / (σₓ · σᵧ)

Range: -1 to +1

+1  →  perfect positive (move exactly together)
 0  →  no linear relationship
-1  →  perfect negative (move exactly opposite)
```

### Portfolio Variance Formula (CRITICAL)
```
Two assets A and B with weights wₐ and w_b:

Var(Portfolio) = wₐ²·σₐ² + w_b²·σ_b² + 2·wₐ·w_b·Cov(A,B)

Or using correlation:
Var(Portfolio) = wₐ²·σₐ² + w_b²·σ_b² + 2·wₐ·w_b·ρ·σₐ·σ_b
```

### Diversification Magic
```
If ρ = +1:  No diversification benefit
If ρ =  0:  Some diversification
If ρ = -1:  Maximum diversification (can eliminate all risk!)

Example:
  Stock A: σ = 20%, Stock B: σ = 20%, equal weights

  ρ = +1:  Portfolio σ = 20%  (no benefit)
  ρ =  0:  Portfolio σ = 14.1%  (√2 reduction)
  ρ = -1:  Portfolio σ = 0%   (perfect hedge!)
```

### Interview Question
> "You have two stocks both with 20% volatility and correlation 0.5. What's portfolio volatility with equal weights?"

```
Var = (0.5)²(0.2)² + (0.5)²(0.2)² + 2(0.5)(0.5)(0.5)(0.2)(0.2)
    = 0.01 + 0.01 + 0.01
    = 0.03

σ = √0.03 = 17.3%  (less than 20% → diversification worked!)
```

---

## 11. Sharpe Ratio

### Formula
```
Sharpe = (E[R] - Rƒ) / σ

Where:
  E[R] = expected return of strategy
  Rƒ   = risk-free rate (e.g., 6% in India)
  σ    = standard deviation of returns
```

### Interpretation
```
Sharpe > 1.0  →  Good
Sharpe > 2.0  →  Very good
Sharpe > 3.0  →  Exceptional (rare)
Sharpe < 0    →  Worse than risk-free
```

### Example
```
Strategy returns: 18% per year
Risk-free rate:   6%
Std Dev:          8%

Sharpe = (18% - 6%) / 8% = 12% / 8% = 1.5  ✅ Good strategy
```

### Limitations
```
1. Assumes normal distribution (penalizes upside volatility too)
2. Doesn't capture tail risk
3. Can be gamed by smoothing returns

Better alternatives:
  Sortino Ratio  → only penalizes downside volatility
  Calmar Ratio   → return / max drawdown
```

---

## 12. Monte Carlo Simulation

### What It Is
Simulate thousands of random scenarios to estimate outcomes

### Steps
```
1. Define model (e.g., stock follows geometric Brownian motion)
2. Generate random numbers
3. Simulate price paths
4. Compute outcome for each path
5. Average outcomes → estimated value
```

### Stock Price Simulation (GBM)
```
Sₜ = S₀ × exp[(μ - σ²/2)·t + σ·√t·Z]

Where:
  S₀ = current price
  μ  = drift (expected return)
  σ  = volatility
  t  = time
  Z  ~ N(0,1) = random standard normal
```

### Python-Style Pseudocode
```python
import numpy as np

S0 = 100      # current price
mu = 0.10     # 10% annual return
sigma = 0.20  # 20% volatility
T = 1         # 1 year
N = 10000     # simulations

Z = np.random.standard_normal(N)
ST = S0 * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

expected_price = np.mean(ST)
percentile_5   = np.percentile(ST, 5)   # 5% worst case
```

### Use Cases
```
1. Option pricing (when no closed form exists)
2. VaR (Value at Risk) estimation
3. Portfolio stress testing
4. Risk scenario analysis
```

---

## 13. Volatility

### Two Types

**Historical Volatility (HV)**
```
= Standard deviation of past returns (annualized)

HV = σ_daily × √252   (252 trading days in a year)

Example:
  Daily std dev = 1%
  Annual HV = 1% × √252 = 15.87%
```

**Implied Volatility (IV)**
```
= Volatility implied by current option market prices
= Market's expectation of future volatility

Derived by: plug option price into Black-Scholes, solve for σ
```

### IV vs HV
```
IV > HV  →  Options are expensive (market fears future volatility)
IV < HV  →  Options are cheap (market calm vs recent history)

Trading signal:
  IV >> HV  →  Sell options (collect premium)
  IV << HV  →  Buy options (cheap insurance)
```

### VIX
```
VIX = implied volatility of S&P 500 options (30-day)
    = "Fear index"

VIX < 15   →  Calm market
VIX 15-25  →  Normal
VIX > 30   →  Fear / high uncertainty
VIX > 50   →  Extreme fear (2008, 2020 COVID)
```

---

## 14. Practice Problems

### Problem 1 — Expected Value
```
A trade:
  70% chance: +₹200
  30% chance: −₹150

Q1: What is the expected value?
Q2: Is this a good trade?
Q3: What is the variance?
```

**Solution:**
```
Q1: E[X] = 0.7×200 + 0.3×(-150)
         = 140 - 45
         = ₹95  ✅

Q2: Yes! E[X] = ₹95 > 0 → positive edge

Q3: Var(X) = 0.7×(200-95)² + 0.3×(-150-95)²
           = 0.7×11025 + 0.3×60025
           = 7717.5 + 18007.5
           = 25725
    σ = √25725 ≈ ₹160
```

---

### Problem 2 — Bayes' Theorem
```
A stock pattern appears before a breakout 80% of the time.
The pattern appears randomly 20% of the time.
Breakouts happen 10% of the time.

You see the pattern. What's P(breakout)?
```

**Solution:**
```
P(breakout)           = 0.10
P(pattern | breakout) = 0.80
P(pattern)            = 0.20

P(breakout | pattern) = (0.80 × 0.10) / 0.20
                      = 0.08 / 0.20
                      = 0.40  → 40% chance of breakout
```

---

### Problem 3 — Portfolio Variance
```
Stock A: weight=60%, σ=15%
Stock B: weight=40%, σ=25%
Correlation = 0.3

What is portfolio volatility?
```

**Solution:**
```
Var = (0.6)²(0.15)² + (0.4)²(0.25)² + 2(0.6)(0.4)(0.3)(0.15)(0.25)
    = 0.36×0.0225 + 0.16×0.0625 + 2×0.6×0.4×0.3×0.0375
    = 0.0081 + 0.01 + 0.0054
    = 0.0235

σ_portfolio = √0.0235 ≈ 15.3%
```

---

### Problem 4 — Sharpe Ratio
```
Strategy A: Return=15%, σ=10%
Strategy B: Return=25%, σ=20%
Risk-free rate = 5%

Which strategy is better on risk-adjusted basis?
```

**Solution:**
```
Sharpe A = (15-5)/10 = 1.0
Sharpe B = (25-5)/20 = 1.0

Equal Sharpe! Despite B having higher return,
it takes proportionally more risk.
→ Both are equally efficient
```

---

### Problem 5 — CLT Application
```
A trader makes 100 independent trades.
Each trade: E[X] = ₹50, σ = ₹200

What is the probability total P&L > ₹7000?
```

**Solution:**
```
Total P&L ~ N(100×50, 100×200²) = N(5000, 4000000)
Std Dev of total = √4000000 = ₹2000

P(Total > 7000) = P(Z > (7000-5000)/2000)
               = P(Z > 1.0)
               = 1 - 0.8413
               = 15.87%
```

---

## 15. Interview Cheat Sheet

### Must-Know Formulas

```
Expected Value:    E[X] = Σ xᵢ·pᵢ
Variance:          Var(X) = E[X²] - (E[X])²
Std Dev:           σ = √Var(X)
Covariance:        Cov(X,Y) = E[XY] - E[X]·E[Y]
Correlation:       ρ = Cov(X,Y) / (σₓ·σᵧ)
Conditional Prob:  P(A|B) = P(A∩B) / P(B)
Bayes:             P(A|B) = P(B|A)·P(A) / P(B)
Sharpe:            (E[R] - Rƒ) / σ
Portfolio Var:     wₐ²σₐ² + w_b²σ_b² + 2wₐw_bρσₐσ_b
Annualize Vol:     σ_annual = σ_daily × √252
```

### Key Numbers to Remember
```
Normal distribution:
  ±1σ → 68%
  ±2σ → 95%
  ±3σ → 99.7%

Correlation range:  -1 to +1
Probability range:   0 to 1
Sharpe > 1 = good, > 2 = great, > 3 = exceptional
252 trading days in a year
```

### Common Interview Traps
```
❌ "E[X·Y] = E[X]·E[Y]" → Only true if independent!
❌ "Var(X+Y) = Var(X)+Var(Y)" → Only true if independent!
❌ "High return = good strategy" → Check Sharpe ratio!
❌ "Normal distribution models crashes well" → Fat tails exist!
❌ "Correlation = causation" → Never!
```

### Quick Mental Math
```
√2 ≈ 1.41
√3 ≈ 1.73
√252 ≈ 15.87  (use 16 for quick calc)
e ≈ 2.718
ln(2) ≈ 0.693
```

---

## 🗓️ Study Plan

| Week | Focus | Goal |
|------|-------|------|
| Week 1 | Sections 1–3 (RV, EV, Variance) | Solve 20 EV problems |
| Week 2 | Sections 4–6 (Distributions, Bayes) | Master Bayes problems |
| Week 3 | Sections 7–10 (Stats, CLT, Correlation) | Portfolio variance problems |
| Week 4 | Sections 11–13 (Sharpe, Monte Carlo, Vol) | Full mock interviews |

---

> 💡 **Golden Rule:** Every quant decision = maximize E[X] while controlling σ. That's it.

---

## 16. Practice Set 1 — Core Quant Thinking

> Try solving each problem BEFORE reading the solution. That's how real interviews work.

---

### ✅ Q1: Expected Value (must be instant)

```
A strategy:
  70% chance: +₹200
  30% chance: −₹150
```

**Solution:**
```
E = 0.7×200 + 0.3×(-150)
  = 140 - 45
  = ₹95
```
✔️ Positive expected value → good trade (in theory)

---

### ✅ Q2: Variance (risk understanding)

Same trade as Q1. Find the variance.

**Solution:**
```
Step 1: Mean = ₹95

Step 2: Squared deviations
  (200 - 95)²  = 11025
  (-150 - 95)² = 60025

Step 3: Weighted average
  Var = 0.7×11025 + 0.3×60025
      = 7717.5 + 18007.5
      = 25725

  σ = √25725 ≈ ₹160
```
⚠️ High variance = high risk — even though the trade is profitable!

---

### ✅ Q3: Conditional Probability (Total Probability)

```
60% of days market goes up
On "up" days,   your strategy wins 80%
On "down" days, your strategy wins 30%

What is total win probability?
```

**Solution:**
```
P(win) = P(win|up)×P(up) + P(win|down)×P(down)
       = 0.8×0.6 + 0.3×0.4
       = 0.48 + 0.12
       = 0.60  (60%)
```
✔️ This is the **Law of Total Probability** — used constantly in quant models.

---

### ✅ Q4: Bayes' Theorem (interview favorite)

```
Strategy wins 60% overall
When it wins, 70% of time market was up
Market is up 50% of time

Find: P(win | market up)
```

**Solution:**
```
P(win|up) = [P(up|win) × P(win)] / P(up)
          = (0.7 × 0.6) / 0.5
          = 0.42 / 0.5
          = 0.84
```
🔥 If market is up → **84% chance strategy wins** (vs 60% base rate)

This is your **conditional edge** — the core of signal-based trading.

---

### ✅ Q5: Correlation (portfolio thinking)

```
You have 2 strategies:
  Both individually profitable
  Correlation = -0.5

What happens if you combine them?
```

**Answer:**
```
✔️ Portfolio risk REDUCES (negative correlation = natural hedge)
✔️ Returns become smoother (less day-to-day swings)
✔️ Drawdowns become smaller (losses in one offset by gains in other)

Portfolio Var = w₁²σ₁² + w₂²σ₂² + 2·w₁·w₂·(-0.5)·σ₁·σ₂
                                      ↑ this term is NEGATIVE → reduces total variance
```
👉 This is exactly how hedge funds build multi-strategy portfolios.

---

## 17. Interview-Style Questions — Real Level

---

### 🔥 Q6: Trick Question — Fair Coin Game

```
A fair coin:
  Heads → you get ₹100
  Tails → you lose ₹100

Expected value?
```

**Answer:**
```
E[X] = 0.5×100 + 0.5×(-100) = 0
```

**BUT** — the interviewer will follow up:
> "Would you play this game 1000 times?"

**Smart answer:**
```
Expected value = 0  (no edge)
But variance is high → each game swings ±₹100

With 1000 games:
  Total variance = 1000 × 10000 = 10,000,000
  Std Dev of total P&L = ₹3162

So you could easily be down ₹5000+ just by bad luck.
→ Only play if you have deep enough bankroll to survive variance
→ This is the Kelly Criterion / risk of ruin concept
```

---

### 🔥 Q7: Law of Large Numbers — Casino Question

> "Why do casinos always win?"

**Answer:**
```
Each game has a small negative expected value for the player
  (e.g., roulette house edge ≈ 2.7%)

With millions of bets:
  Law of Large Numbers guarantees actual results → expected value
  Casino's edge materializes with near certainty

Same logic applies to quant trading:
  One trade = luck
  1000 trades = your strategy's true edge shows up
```

---

### 🔥 Q8: Risk-Adjusted Return — Which Strategy?

```
Strategy A: high return, high variance
Strategy B: lower return, low variance

Which is better?
```

**Answer:**
```
Neither is automatically better — compare Sharpe Ratios:

  Sharpe = (Return - Risk-free rate) / Std Dev

Example:
  A: Return=30%, σ=25%, Rf=6%  →  Sharpe = 24/25 = 0.96
  B: Return=18%, σ=10%, Rf=6%  →  Sharpe = 12/10 = 1.20

→ Strategy B is BETTER on risk-adjusted basis
→ Raw return is misleading without context of risk taken
```

---

## 18. Mini Coding Task — Monte Carlo in Python

> This is a core quant skill. Interviewers at quant firms often ask you to code this.

### Task: Simulate 1000 trades and verify expected value

```python
import numpy as np

# Trade setup: 70% chance +200, 30% chance -150
trades = np.random.choice([200, -150], size=1000, p=[0.7, 0.3])

print("Average P&L per trade:", np.mean(trades))   # should be ≈ 95
print("Std Dev:",               np.std(trades))    # should be ≈ 160
print("Total P&L:",             np.sum(trades))    # should be ≈ 95000
```

**Expected output (approximately):**
```
Average P&L per trade: 94.8
Std Dev: 159.3
Total P&L: 94800
```

### Extended: Simulate cumulative P&L path

```python
import numpy as np
import matplotlib.pyplot as plt

trades = np.random.choice([200, -150], size=1000, p=[0.7, 0.3])
cumulative_pnl = np.cumsum(trades)

plt.plot(cumulative_pnl)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Cumulative P&L over 1000 trades')
plt.xlabel('Trade number')
plt.ylabel('P&L (₹)')
plt.show()
```

### What this teaches you:
```
1. Law of Large Numbers in action → average converges to ₹95
2. Variance is real → path is NOT a straight line up
3. Even with positive edge, you can have losing streaks
4. This IS Monte Carlo simulation — core quant skill
```

---

## 🎯 What You've Learned (Summary)

| Concept | Trading Meaning |
|---------|----------------|
| Expected Value | Your profit edge per trade |
| Variance | Your risk per trade |
| Conditional Probability | How strategy behaves in different market regimes |
| Bayes' Theorem | Update your edge when new information arrives |
| Correlation | How to combine strategies to reduce risk |
| Sharpe Ratio | True measure of strategy quality |
| Monte Carlo | Simulate and stress-test before going live |

> 🚀 You now think better than most developers applying for quant roles.

---

*Built for interview prep — Sonu Kumar (@sonufsd)*

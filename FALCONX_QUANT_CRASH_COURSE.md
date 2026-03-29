# ⚡ FalconX Quant Crash Course — Interview Ready in One Flow
> Probability + Black-Scholes + Greeks + Monte Carlo + Trading Intuition

---

## 🗺️ Table of Contents

1. [Probability & Statistics — Core](#1-probability--statistics--core)
2. [Black-Scholes Model](#2-black-scholes-model)
3. [Greeks — Option Sensitivities](#3-greeks--option-sensitivities)
4. [Monte Carlo Option Pricing](#4-monte-carlo-option-pricing)
5. [Trading Intuition & Risk](#5-trading-intuition--risk)
6. [Python Snippet — MC Pricing + Delta](#6-python-snippet--mc-pricing--delta)
7. [Interview-Ready Summary](#7-interview-ready-summary)

---

## 1. Probability & Statistics — Core

### Expected Value (Mean)
```
E[X] = Σ xᵢ · P(xᵢ)

Example:
  60% chance: +₹100
  40% chance: −₹50
  E[X] = 0.6×100 + 0.4×(−50) = ₹40  ✅ Take this trade
```

### Variance
```
Var(X) = E[X²] − (E[X])²

→ Measures spread of outcomes = RISK
→ Std Dev σ = √Var(X)
```

### Binomial Probability
```
n trials, k successes, probability p:

P(X=k) = C(n,k) × pᵏ × (1−p)^(n−k)

Example: 60% win rate, 10 trades, exactly 7 wins?
P(X=7) = C(10,7) × 0.6⁷ × 0.4³ = 0.215
```

### Normal Approximation (Large n)
```
For large n:
  X ~ N(np, np(1−p))

Example: 100 trades, 60% win rate
  Mean wins = 60
  Std Dev   = √(100 × 0.6 × 0.4) ≈ 4.9
```

### Conditional Probability
```
P(A|B) = P(A ∩ B) / P(B)

Trading example:
  P(market up | inflation dropped) = P(both) / P(inflation dropped)
  → Gives you an EDGE signal
```

### Bayes' Theorem
```
P(A|B) = [P(B|A) × P(A)] / P(B)

→ Update your belief with new evidence
→ Used in real-time risk model updates
```

### ✅ Interview Tip
```
Always check: are events INDEPENDENT or CONDITIONAL?

Independent: P(A|B) = P(A)  → knowing B tells you nothing about A
Conditional: P(A|B) ≠ P(A)  → knowing B changes your estimate of A

In trading: correlated assets are NOT independent
```

---

## 2. Black-Scholes Model

### Formula — European Call Option
```
C = S · N(d₁) − K · e^(−rT) · N(d₂)

d₁ = [ ln(S/K) + (r + σ²/2)·T ] / (σ·√T)
d₂ = d₁ − σ·√T
```

### Variables
| Symbol | Meaning |
|--------|---------|
| S | Current stock price |
| K | Strike price |
| r | Risk-free interest rate |
| σ | Volatility (annualized) |
| T | Time to expiry (years) |
| N() | Cumulative normal distribution |

### Put Option (Put-Call Parity)
```
P = K · e^(−rT) · N(−d₂) − S · N(−d₁)
```

### Intuition (Don't Memorize — Understand)
```
C = (what you get) − (what you pay)

S · N(d₁)           = expected value of receiving the stock
K · e^(−rT) · N(d₂) = present value of paying the strike

→ Option price = expected gain − expected cost
```

### What Drives Option Price?
```
Stock price S ↑  →  Call ↑   (deeper in the money)
Volatility σ ↑   →  Call ↑   (more upside chance)
Time T ↑         →  Call ↑   (more opportunity)
Strike K ↑       →  Call ↓   (harder to profit)
```

### Key Assumptions (Know These for Interview)
```
1. Log-normal stock prices (GBM)
2. Constant volatility
3. No transaction costs
4. Continuous trading possible
5. No dividends
6. No arbitrage
```

### What Breaks in Real Markets
```
❌ Volatility smile — implied vol varies by strike
❌ Fat tails — crashes happen more than BS predicts
❌ Price jumps — stocks gap overnight
❌ Transaction costs — continuous hedging is expensive
```

---

## 3. Greeks — Option Sensitivities

### Summary Table
| Greek | Measures | Formula | Intuition |
|-------|----------|---------|-----------|
| Delta (Δ) | ∂Price/∂S | ≈ N(d₁) | Directional exposure |
| Gamma (Γ) | ∂Δ/∂S | ∂²Price/∂S² | How fast delta changes |
| Vega (ν) | ∂Price/∂σ | S·N'(d₁)·√T | Volatility risk |
| Theta (θ) | ∂Price/∂T | (negative) | Time decay per day |
| Rho (ρ) | ∂Price/∂r | K·T·e^(−rT)·N(d₂) | Interest rate sensitivity |

### Delta
```
Call delta: 0 to 1
  Deep ITM  → Δ ≈ 1    (moves like stock)
  ATM       → Δ ≈ 0.5
  Deep OTM  → Δ ≈ 0    (barely moves)

Use: hedge ratio — buy Δ shares to offset 1 option
```

### Gamma
```
Highest at ATM options
→ Delta changes fastest near the money
→ Gamma = cost of hedging (you must rebalance frequently)

Long option → positive Gamma (you want big moves)
Short option → negative Gamma (big moves hurt you)
```

### Vega
```
Always positive for both calls and puts
→ Higher vol = higher option price (always)

Long option → positive Vega (you want vol to increase)
Short option → negative Vega (you want vol to decrease)
```

### Theta
```
Usually negative (options lose value daily)
→ Option buyers fight against theta
→ Option sellers earn theta

"Theta is the rent you pay for holding an option"
```

### Key Trading Intuition
```
Option BUYERS want:
  ✅ Big price moves (positive Gamma)
  ✅ Volatility to increase (positive Vega)
  ❌ Fight against time decay (negative Theta)

Option SELLERS want:
  ✅ Quiet markets (negative Gamma — small moves)
  ✅ Volatility to decrease (negative Vega)
  ✅ Time to pass (positive Theta)

Delta hedging:
  → Removes directional risk
  → But Gamma and Vega remain
  → Must rebalance as stock moves
```

---

## 4. Monte Carlo Option Pricing

### Core Algorithm — European Call
```
Step 1: Simulate S_T many times
        S_T = S₀ × exp[(r − 0.5σ²)T + σ√T × Z]   Z ~ N(0,1)

Step 2: Compute payoff for each simulation
        payoff = max(S_T − K, 0)

Step 3: Average all payoffs

Step 4: Discount to present
        Price = e^(−rT) × mean(payoffs)
```

### Why Monte Carlo Over Black-Scholes?
```
BS works for: simple European options only

MC works for:
  ✅ Asian options (average price)
  ✅ Barrier options (path must not cross B)
  ✅ Lookback options (min/max of path)
  ✅ Any exotic payoff structure
  ✅ Multi-asset correlated portfolios
```

### Path-Dependent Example — Asian Option
```
Payoff = max(average(S) − K, 0)

→ Must simulate full price path (every time step)
→ Average price over path → compute payoff
→ BS cannot handle this; MC handles it naturally
```

### Greeks via Monte Carlo
```
Finite Difference (simple):
  Delta ≈ [Price(S+ε) − Price(S−ε)] / 2ε
  Vega  ≈ [Price(σ+ε) − Price(σ−ε)] / 2ε

Pathwise Derivative (efficient):
  Delta = e^(−rT) × E[ 1(S_T > K) × S_T/S ]
  → Single simulation, lower noise, faster

Likelihood Ratio Method:
  → For discontinuous payoffs (digital options)
```

### Performance Tips
```
1. Vectorization     → NumPy arrays, no Python loops
2. Antithetic        → use Z and −Z pairs (2x accuracy)
3. Control variates  → correct using known BS price (10x accuracy)
4. Sobol sequences   → quasi-random for better coverage (100x)
5. Common random #s  → same Z for bumped simulations (stable Greeks)
6. GPU computing     → CuPy/CUDA for 1000x speedup
```

### Convergence
```
Error ∝ 1/√N

10K sims   → ~±0.3 accuracy
100K sims  → ~±0.1 accuracy
1M sims    → ~±0.03 accuracy

To halve error → need 4× more simulations
```

---

## 5. Trading Intuition & Risk

### Think in Greeks, Not Just Prices
```
Amateur: "The option costs ₹10"
Quant:   "Delta=0.5, Gamma=0.02, Vega=0.38, Theta=−0.017"

The quant knows:
  → How much the position moves with the stock (Delta)
  → How the hedge will drift (Gamma)
  → How vol changes affect PnL (Vega)
  → How much they lose per day (Theta)
```

### Risk Management Framework
```
1. Delta risk   → hedge with underlying stock
2. Gamma risk   → monitor and rebalance frequently
3. Vega risk    → hedge with other options (vol trading)
4. Theta risk   → manage position size and time horizon
```

### VaR — Value at Risk
```
"Maximum loss at X% confidence over N days"

95% 1-day VaR = ₹50,000
→ 95% of days, loss < ₹50,000
→ 5% of days, loss > ₹50,000

CVaR (Expected Shortfall):
→ Average loss on the bad 5% of days
→ More informative than VaR
→ Preferred by regulators (Basel III)
```

### Real Quant Mindset
```
Price is what you pay.
Value is what you get.
Risk is what you don't see.

→ Every trade has an expected value AND a risk profile
→ Good quants optimize the ratio (Sharpe ratio)
→ Monte Carlo lets you simulate the full distribution of outcomes
```

---

## 6. Python Snippet — MC Pricing + Delta

```python
import numpy as np
from scipy.stats import norm

# --- Monte Carlo Call Pricer ---
def monte_carlo_call(S, K, T, r, sigma, sims=100_000):
    Z = np.random.randn(sims)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r*T) * np.mean(payoff)

# --- Delta via Finite Difference ---
def delta_mc(S, K, T, r, sigma, eps=0.5):
    return (monte_carlo_call(S+eps, K, T, r, sigma) -
            monte_carlo_call(S-eps, K, T, r, sigma)) / (2*eps)

# --- Black-Scholes (for comparison) ---
def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# --- Run ---
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

print(f"MC Call Price : {monte_carlo_call(S, K, T, r, sigma):.4f}")
print(f"BS Call Price : {bs_call(S, K, T, r, sigma):.4f}")
print(f"MC Delta      : {delta_mc(S, K, T, r, sigma):.4f}")
print(f"BS Delta      : {norm.cdf((np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))):.4f}")
```

### Expected Output
```
MC Call Price : 10.4489   ← converges to BS
BS Call Price : 10.4506
MC Delta      : 0.6371
BS Delta      : 0.6368
```

---

## 7. Interview-Ready Summary

### The 5 Things You Must Know

```
1. Probability
   → Expected value, variance, binomial, normal, Bayes
   → Always check independence vs conditional

2. Black-Scholes
   → Formula: C = S·N(d₁) − K·e^(−rT)·N(d₂)
   → Intuition: expected gain minus expected cost
   → Assumptions and where they break

3. Greeks
   → Delta = directional exposure (hedge ratio)
   → Gamma = how fast delta changes (rebalancing cost)
   → Vega = volatility sensitivity (vol trading)
   → Theta = time decay (option sellers earn this)

4. Monte Carlo
   → Simulate S_T → compute payoff → average → discount
   → Use for path-dependent options (Asian, Barrier)
   → Greeks via finite difference or pathwise
   → Variance reduction: antithetic, control variates, Sobol

5. Risk Thinking
   → Think in Greeks, not just prices
   → VaR and CVaR for portfolio risk
   → Monte Carlo for stress testing
```

### One-Line Answers for Common Questions

```
Q: Why does vol increase option price?
A: Asymmetric payoff — downside capped, upside unlimited.

Q: What is delta hedging?
A: Buy Δ shares per option to remove directional risk.

Q: Why MC over BS?
A: BS only handles simple European options; MC handles any payoff.

Q: What breaks Black-Scholes?
A: Volatility smile, fat tails, price jumps, transaction costs.

Q: What is Gamma risk?
A: Your delta hedge drifts as stock moves — must rebalance constantly.

Q: What is implied volatility?
A: The vol that makes BS price equal to market price — market's forecast of future vol.
```

### Cheat Sheet
```
BS Call:  C = S·N(d₁) − K·e^(−rT)·N(d₂)
d₁ = [ln(S/K) + (r+σ²/2)T] / σ√T
d₂ = d₁ − σ√T

Greeks:
  Delta = N(d₁)          → 0 to 1 for calls
  Gamma = N'(d₁)/Sσ√T   → highest ATM
  Vega  = S·N'(d₁)·√T   → always positive
  Theta = negative        → daily bleed

MC:
  S_T = S₀·exp[(r−0.5σ²)T + σ√T·Z]
  Price = e^(−rT) × mean[max(S_T−K, 0)]
  Error ∝ 1/√N
```

---

*Part of the Quant Interview Prep Series — [@sonufsd](https://github.com/sonufsd)*

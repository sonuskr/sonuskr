# 📉 Black-Scholes — From Zero to Interview Ready
> The model that separates a normal dev from a quant dev

---

## 🗺️ Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [Understand the Instrument — Call Option](#2-understand-the-instrument--call-option)
3. [Key Ideas Behind Black-Scholes](#3-key-ideas-behind-black-scholes)
4. [The Formula](#4-the-formula)
5. [Intuition — Don't Memorize, Understand](#5-intuition--dont-memorize-understand)
6. [What Drives Option Price?](#6-what-drives-option-price)
7. [The Greeks — Interview Must](#7-the-greeks--interview-must)
8. [Python Implementation](#8-python-implementation)
9. [Interview Questions](#9-interview-questions)
10. [Reality Check](#10-reality-check)
11. [What to Do Next](#11-what-to-do-next)

---

## 1. What Problem Are We Solving?

We want to answer one question:

> **"What is the fair price of an option today?"**

### Example Setup
```
Stock price today  = ₹100
Call option strike = ₹100
Expiry             = 1 month

👉 What should this option cost right now?
```

This is exactly what Black-Scholes answers.

---

## 2. Understand the Instrument — Call Option

A **call option** gives the buyer the **right (not obligation)** to buy a stock at the strike price.

### Payoff at Expiry
```
Payoff = max(Stock Price - Strike, 0)
```

### Examples
```
Stock = ₹120  →  Payoff = max(120 - 100, 0) = ₹20  ✅ In the money
Stock = ₹90   →  Payoff = max(90 - 100, 0)  = ₹0   ❌ Out of the money
```

### Key Property
```
Loss    = limited (only premium paid)
Upside  = unlimited

👉 This asymmetry is why options have value
```

---

## 3. Key Ideas Behind Black-Scholes

### 3.1 Geometric Brownian Motion (GBM)
Stock prices follow **random movement** with:
- **Drift** — general upward trend
- **Volatility** — random noise around that trend

```
dS = μS dt + σS dW

Where:
  μ  = drift (expected return)
  σ  = volatility
  dW = random Wiener process (pure randomness)
```

### 3.2 No Arbitrage
```
No risk-free profit should exist in the market.

👉 This is the backbone of all derivative pricing.
   If arbitrage existed, traders would exploit it until it disappears.
```

### 3.3 Continuous Hedging (Delta Hedging)
```
By continuously buying/selling the underlying stock,
you can construct a RISK-FREE portfolio.

Risk-free portfolio → must earn risk-free rate (r)
→ This constraint gives us the pricing equation
```

---

## 4. The Formula

### Call Option Price
```
C = S · N(d1) − K · e^(−rT) · N(d2)
```

### Where
```
d1 = [ ln(S/K) + (r + σ²/2) · T ] / (σ · √T)
d2 = d1 − σ · √T
```

### Variables
| Symbol | Meaning |
|--------|---------|
| `S` | Current stock price |
| `K` | Strike price |
| `r` | Risk-free interest rate |
| `T` | Time to expiry (in years) |
| `σ` | Volatility (annualized std dev of returns) |
| `N()` | Cumulative normal distribution (probability) |

### Put Option Price (via Put-Call Parity)
```
P = K · e^(−rT) · N(−d2) − S · N(−d1)
```

---

## 5. Intuition — Don't Memorize, Understand

Break the formula into two parts:

### First Term: `S · N(d1)`
```
= Expected value of RECEIVING the stock

N(d1) ≈ probability-weighted value of owning the stock
```

### Second Term: `K · e^(−rT) · N(d2)`
```
= Present value of PAYING the strike price

K · e^(−rT) = strike discounted to today
N(d2)       = probability option ends in the money
```

### So the whole formula says:
```
Option Price = Expected gain from stock − Expected cost of buying it

C = (what you get) − (what you pay)
```

---

## 6. What Drives Option Price?

| Factor | Change | Effect on Call | Why |
|--------|--------|---------------|-----|
| Stock Price `S` | ↑ | ↑ | Closer to / deeper in the money |
| Volatility `σ` | ↑ | ↑ 🔥 | More uncertainty = more upside chance |
| Time `T` | ↑ | ↑ | More time = more opportunity |
| Interest Rate `r` | ↑ | ↑ | Strike discount increases |
| Strike `K` | ↑ | ↓ | Harder to profit |

### 🔥 Most Important: Volatility
```
Options have ASYMMETRIC payoff:
  - Downside is capped (lose only premium)
  - Upside is unlimited

So MORE volatility = MORE chance of big upside = HIGHER option price

This is why traders say: "You're not trading the stock, you're trading volatility"
```

---

## 7. The Greeks — Interview Must

Greeks measure **sensitivity** of option price to each input.

### Delta (Δ) — Sensitivity to Stock Price
```
Δ = ∂C/∂S ≈ N(d1)

Call delta range: 0 to 1
  Deep ITM  → Δ ≈ 1    (moves like stock)
  ATM       → Δ ≈ 0.5
  Deep OTM  → Δ ≈ 0    (barely moves)

Meaning: If stock moves ₹1 → option moves ₹Δ
```

### Gamma (Γ) — Rate of Change of Delta
```
Γ = ∂Δ/∂S = ∂²C/∂S²

Highest at ATM options
→ Delta changes fastest when you're at the money
→ Important for hedging — your hedge needs frequent adjustment
```

### Vega (ν) — Sensitivity to Volatility
```
ν = ∂C/∂σ

Always positive for both calls and puts
→ Higher vol = higher option price (always)

🔥 Most important Greek for options traders
```

### Theta (θ) — Time Decay
```
θ = ∂C/∂T  (usually negative)

Options LOSE value every day as expiry approaches
→ Option buyers fight against theta
→ Option sellers profit from theta

"Theta is the rent you pay for holding an option"
```

### Rho (ρ) — Sensitivity to Interest Rate
```
ρ = ∂C/∂r

Usually small effect
Matters more for long-dated options
```

### Greeks Summary Table
| Greek | Measures | Call Value | Key Insight |
|-------|----------|-----------|-------------|
| Delta | Stock price sensitivity | 0 to 1 | Hedge ratio |
| Gamma | Delta's rate of change | Always + | Hedging cost |
| Vega | Volatility sensitivity | Always + | Vol trading |
| Theta | Time decay | Usually − | Daily bleed |
| Rho | Interest rate sensitivity | Usually + | Minor effect |

---

## 8. Python Implementation

```python
import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    S     : current stock price
    K     : strike price
    T     : time to expiry in years
    r     : risk-free rate (e.g. 0.05 for 5%)
    sigma : volatility (e.g. 0.2 for 20%)
    """
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return round(price, 4)


def greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100        # per 1% vol change
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365  # per day

    return {"delta": round(delta, 4), "gamma": round(gamma, 4),
            "vega": round(vega, 4), "theta": round(theta, 4)}


# --- Example ---
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

print("Call Price :", black_scholes(S, K, T, r, sigma, 'call'))
print("Put Price  :", black_scholes(S, K, T, r, sigma, 'put'))
print("Greeks     :", greeks(S, K, T, r, sigma))
```

### Expected Output
```
Call Price : 10.4506
Put Price  : 5.5735
Greeks     : {'delta': 0.6368, 'gamma': 0.0188, 'vega': 0.3752, 'theta': -0.0177}
```

### Experiment: See How Volatility Affects Price
```python
for vol in [0.1, 0.2, 0.3, 0.4, 0.5]:
    price = black_scholes(100, 100, 1, 0.05, vol)
    print(f"σ = {int(vol*100)}%  →  Call = ₹{price}")
```

```
σ = 10%  →  Call = ₹6.80
σ = 20%  →  Call = ₹10.45
σ = 30%  →  Call = ₹14.23
σ = 40%  →  Call = ₹18.03
σ = 50%  →  Call = ₹21.79
```

---

## 9. Interview Questions

### ❓ Q1: Why does higher volatility increase option price?
```
Because option payoff is ASYMMETRIC:
  - Downside is capped at premium paid
  - Upside is unlimited

Higher volatility = higher chance of large upside
→ Option buyer benefits more from vol than they're hurt by it
→ So option price increases with volatility
```

### ❓ Q2: What are the assumptions of Black-Scholes?
```
1. Stock follows Geometric Brownian Motion (log-normal prices)
2. Volatility is CONSTANT over the life of the option
3. No transaction costs or taxes
4. Continuous trading is possible
5. No dividends
6. Risk-free rate is constant
7. No arbitrage opportunities
```

### ❓ Q3: What breaks in real markets?
```
🔥 Strong answer for interviews:

1. Volatility smile/skew
   → Implied vol varies by strike (BS assumes constant)
   → OTM puts are more expensive than BS predicts (crash fear)

2. Fat tails (leptokurtosis)
   → Real returns have more extreme moves than normal distribution
   → 2008 crash was a "25-sigma event" under BS assumptions

3. Price jumps
   → Stocks can gap overnight (earnings, news)
   → BS assumes continuous movement

4. Liquidity constraints
   → Continuous hedging is impossible in practice

5. Transaction costs
   → Frequent rebalancing is expensive
```

### ❓ Q4: What is delta hedging?
```
If you sell a call option (delta = 0.5):
  → You're exposed if stock moves up

To hedge: buy 0.5 shares of stock per option sold

Now if stock moves ₹1:
  Option loses ₹0.5 (you're short)
  Stock gains ₹0.5 (you're long)
  → Net = ₹0  ✅ Hedged

Problem: delta changes as stock moves (that's gamma)
→ You must continuously rebalance → this has a cost
```

### ❓ Q5: What is implied volatility?
```
Black-Scholes takes volatility as INPUT → outputs price

Implied volatility does the REVERSE:
  Takes market price as INPUT → solves for volatility

"What volatility does the market imply given this option price?"

→ IV is the market's forecast of future volatility
→ VIX index = implied vol of S&P 500 options
```

---

## 10. Reality Check

```
Black-Scholes is the BASELINE, not the final word.

In real trading desks:
  ✅ Everyone knows BS deeply
  ✅ Used as a common language (quoting in "vol" terms)
  ❌ Not used as-is for actual pricing

Real models used:
  - Stochastic volatility (Heston model)
  - Jump-diffusion models (Merton)
  - Local volatility models (Dupire)

But you CANNOT understand these without mastering BS first.
```

---

## 11. What to Do Next

### Practice
```
1. Run the Python code — change S, K, T, σ and observe price changes
2. Plot option price vs volatility (should be linear-ish)
3. Plot option price vs stock price (should be convex — that's gamma)
```

### Build Project: Options Pricing Tool
```
Input:  S, K, T, σ, r
Output: Call price, Put price, All Greeks

Extend it:
  - Add a payoff diagram at expiry
  - Show how price changes as time decays (theta visualization)
  - Plot the volatility smile
```

### Next Topics to Study
```
1. Binomial Tree Model (discrete version of BS)
2. Implied Volatility & Volatility Surface
3. Heston Model (stochastic volatility)
4. Monte Carlo option pricing
5. Put-Call Parity (arbitrage relationship)
```

---

## ⚡ Quick Reference Cheat Sheet

```
Black-Scholes Call:
  C = S·N(d1) − K·e^(−rT)·N(d2)

  d1 = [ln(S/K) + (r + σ²/2)T] / σ√T
  d2 = d1 − σ√T

Greeks:
  Delta = N(d1)                                          → stock sensitivity
  Gamma = N'(d1) / (S·σ·√T)                             → delta sensitivity
  Vega  = S·N'(d1)·√T                                   → vol sensitivity
  Theta = −[S·N'(d1)·σ/2√T + r·K·e^(−rT)·N(d2)]       → time decay

Key Rules:
  Vol ↑  → Option price ↑  (always)
  Time ↑ → Option price ↑  (always)
  ATM delta ≈ 0.5
  Deep ITM delta → 1
  Deep OTM delta → 0
```

---

*Part of the Quant Interview Prep Series — [@sonufsd](https://github.com/sonufsd)*

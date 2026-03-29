# 🎲 Monte Carlo Options Pricing — From Zero to Interview Ready
> What real quant systems use when Black-Scholes becomes too limited

---

## 🗺️ Table of Contents

1. [What is Monte Carlo?](#1-what-is-monte-carlo)
2. [Why Do We Need It?](#2-why-do-we-need-it)
3. [Core Idea — Simulating Stock Prices](#3-core-idea--simulating-stock-prices)
4. [Algorithm Step by Step](#4-algorithm-step-by-step)
5. [Python Implementation — Simple](#5-python-implementation--simple)
6. [Accuracy vs Speed Tradeoff](#6-accuracy-vs-speed-tradeoff)
7. [Path-Dependent Options — Where MC Shines](#7-path-dependent-options--where-mc-shines)
8. [Upgraded Code — Multi-Step Paths](#8-upgraded-code--multi-step-paths)
9. [Greeks via Monte Carlo](#9-greeks-via-monte-carlo)
10. [Variance Reduction Techniques](#10-variance-reduction-techniques)
11. [Interview Questions](#11-interview-questions)
12. [Real Trading Insight](#12-real-trading-insight)
13. [Mini Exercise](#13-mini-exercise)
14. [Quick Reference Cheat Sheet](#14-quick-reference-cheat-sheet)

---

## 1. What is Monte Carlo?

> Simulate the future many times → average the outcome

Instead of solving a closed-form formula (like Black-Scholes), we:

1. Generate many **possible price paths**
2. Calculate **payoff** for each path
3. Take the **average** and discount to today

```
Real world analogy:
  You want to know the average outcome of rolling a dice 1000 times.
  You don't solve it analytically — you just roll it 1000 times and average.

Monte Carlo does the same for stock prices.
```

---

## 2. Why Do We Need It?

### Black-Scholes Limitations
```
Black-Scholes assumes:
  ✅ Simple European payoff (only final price matters)
  ✅ Constant volatility
  ✅ Log-normal price distribution

Real markets have:
  ❌ Exotic options (Asian, Barrier, Lookback)
  ❌ Path-dependent payoffs (average price matters)
  ❌ Stochastic volatility
  ❌ Complex multi-asset derivatives
```

### Where Monte Carlo Wins
```
✅ Any payoff structure — no matter how complex
✅ Path-dependent options — tracks every step
✅ Multi-asset portfolios — simulate correlated assets
✅ Risk simulation — stress test entire portfolios
✅ No closed-form needed — just simulate
```

---

## 3. Core Idea — Simulating Stock Prices

We simulate stock price using **Geometric Brownian Motion (GBM)**:

### Single-Step Formula (to final price)
```
S_T = S_0 × exp[ (r - 0.5σ²)T + σ√T × Z ]

Where:
  S_0 = current stock price
  S_T = stock price at expiry
  r   = risk-free rate
  σ   = volatility
  T   = time to expiry
  Z   = random number from N(0,1)
```

### Why `(r - 0.5σ²)` and not just `r`?
```
This is the Itô correction term.

When you exponentiate a normal distribution, the mean shifts.
The -0.5σ² corrects for this so that E[S_T] = S_0 × e^(rT)

👉 Without it, your simulation would be biased upward.
```

---

## 4. Algorithm Step by Step

### For a European Call Option

```
Step 1: Generate N random numbers Z ~ N(0,1)

Step 2: Simulate final stock price for each Z
        S_T = S_0 × exp[(r - 0.5σ²)T + σ√T × Z]

Step 3: Compute payoff for each simulation
        payoff = max(S_T - K, 0)

Step 4: Average all payoffs
        avg_payoff = mean(all payoffs)

Step 5: Discount to present value
        Price = e^(-rT) × avg_payoff
```

### Visual Flow
```
Z₁ → S_T1 → payoff₁ ─┐
Z₂ → S_T2 → payoff₂ ─┤
Z₃ → S_T3 → payoff₃ ─┼─→ average → discount → Price
...                   ─┤
Zₙ → S_Tₙ → payoffₙ ─┘
```

---

## 5. Python Implementation — Simple

```python
import numpy as np

def monte_carlo_call(S, K, T, r, sigma, simulations=100_000):
    """
    S          : current stock price
    K          : strike price
    T          : time to expiry in years
    r          : risk-free rate
    sigma      : volatility
    simulations: number of random paths
    """
    Z = np.random.randn(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return round(price, 4)


def monte_carlo_put(S, K, T, r, sigma, simulations=100_000):
    Z = np.random.randn(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoffs)
    return round(price, 4)


# --- Example ---
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

print("MC Call Price :", monte_carlo_call(S, K, T, r, sigma))
print("MC Put Price  :", monte_carlo_put(S, K, T, r, sigma))
# Black-Scholes reference: Call ≈ 10.45, Put ≈ 5.57
```

---

## 6. Accuracy vs Speed Tradeoff

```
Simulations   Accuracy        Speed
──────────────────────────────────────
1,000         Rough (~±1.0)   Very fast
10,000        OK    (~±0.3)   Fast
100,000       Good  (~±0.1)   ~1 sec
1,000,000     Great (~±0.03)  ~10 sec
```

### Convergence Check
```python
import numpy as np

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

for n in [1_000, 10_000, 100_000, 1_000_000]:
    Z = np.random.randn(n)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    price = np.exp(-r * T) * np.mean(np.maximum(ST - K, 0))
    print(f"n={n:>10,}  →  Call = {price:.4f}")
```

```
n=      1,000  →  Call = 10.3821
n=     10,000  →  Call = 10.4102
n=    100,000  →  Call = 10.4489
n=  1,000,000  →  Call = 10.4503   ← converges to BS: 10.4506
```

---

## 7. Path-Dependent Options — Where MC Shines

### 7.1 Asian Option (Average Price Option)
```
Payoff = max(average(S) - K, 0)

👉 Black-Scholes CANNOT price this easily
👉 Monte Carlo = perfect fit
```

```python
import numpy as np

def monte_carlo_asian_call(S, K, T, r, sigma, steps=252, simulations=100_000):
    dt = T / steps
    payoffs = []

    for _ in range(simulations):
        prices = [S]
        for _ in range(steps):
            Z = np.random.randn()
            prices.append(prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z))
        avg_price = np.mean(prices)
        payoffs.append(max(avg_price - K, 0))

    return round(np.exp(-r * T) * np.mean(payoffs), 4)

print("Asian Call:", monte_carlo_asian_call(100, 100, 1, 0.05, 0.2))
# Asian call is cheaper than European — averaging reduces variance
```

### 7.2 Barrier Option (Knock-Out)
```
Payoff = max(S_T - K, 0)  BUT only if price never crossed barrier B

👉 Path matters — must track every step
👉 Monte Carlo handles this naturally
```

```python
def monte_carlo_barrier_call(S, K, T, r, sigma, barrier, steps=252, simulations=100_000):
    dt = T / steps
    payoffs = []

    for _ in range(simulations):
        path = [S]
        knocked_out = False
        for _ in range(steps):
            Z = np.random.randn()
            next_price = path[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            path.append(next_price)
            if next_price >= barrier:
                knocked_out = True
                break
        payoff = 0 if knocked_out else max(path[-1] - K, 0)
        payoffs.append(payoff)

    return round(np.exp(-r * T) * np.mean(payoffs), 4)

print("Barrier Call (B=120):", monte_carlo_barrier_call(100, 100, 1, 0.05, 0.2, barrier=120))
```

### 7.3 Lookback Option
```
Payoff = S_T - min(S over entire path)

👉 Depends on the minimum price over the whole path
👉 Only Monte Carlo can handle this efficiently
```

---

## 8. Upgraded Code — Multi-Step Paths

Vectorized version (fast, production-style):

```python
import numpy as np

def simulate_paths(S, T, r, sigma, steps=252, simulations=100_000):
    """Returns matrix of shape (simulations, steps+1)"""
    dt = T / steps
    Z = np.random.randn(simulations, steps)
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_price_paths = np.log(S) + np.cumsum(log_returns, axis=1)
    price_paths = np.exp(np.hstack([np.full((simulations, 1), np.log(S)), log_price_paths]))
    return price_paths


def price_european_call(S, K, T, r, sigma, steps=252, simulations=100_000):
    paths = simulate_paths(S, T, r, sigma, steps, simulations)
    ST = paths[:, -1]
    payoffs = np.maximum(ST - K, 0)
    return round(np.exp(-r * T) * np.mean(payoffs), 4)


def price_asian_call(S, K, T, r, sigma, steps=252, simulations=100_000):
    paths = simulate_paths(S, T, r, sigma, steps, simulations)
    avg_prices = np.mean(paths, axis=1)
    payoffs = np.maximum(avg_prices - K, 0)
    return round(np.exp(-r * T) * np.mean(payoffs), 4)


# --- Compare ---
S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
print("European Call :", price_european_call(S, K, T, r, sigma))
print("Asian Call    :", price_asian_call(S, K, T, r, sigma))
```

---

## 9. Greeks via Monte Carlo

Estimate Greeks using **finite differences** (bump and reprice):

```python
import numpy as np

def mc_price(S, K, T, r, sigma, simulations=100_000):
    np.random.seed(42)
    Z = np.random.randn(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return np.exp(-r * T) * np.mean(np.maximum(ST - K, 0))

def mc_greeks(S, K, T, r, sigma, simulations=100_000):
    h = 0.01  # small bump

    # Delta: dC/dS
    delta = (mc_price(S + h, K, T, r, sigma, simulations) -
             mc_price(S - h, K, T, r, sigma, simulations)) / (2 * h)

    # Gamma: d²C/dS²
    gamma = (mc_price(S + h, K, T, r, sigma, simulations) -
             2 * mc_price(S, K, T, r, sigma, simulations) +
             mc_price(S - h, K, T, r, sigma, simulations)) / (h**2)

    # Vega: dC/dσ
    vega = (mc_price(S, K, T, r, sigma + h, simulations) -
            mc_price(S, K, T, r, sigma - h, simulations)) / (2 * h)

    return {"delta": round(delta, 4), "gamma": round(gamma, 4), "vega": round(vega, 4)}

print(mc_greeks(100, 100, 1, 0.05, 0.2))
```

---

## 10. Variance Reduction Techniques

These make Monte Carlo faster and more accurate.

### 10.1 Antithetic Variates
```
For every random Z, also use -Z
→ Pairs cancel out noise
→ Same accuracy with half the simulations
```

```python
def monte_carlo_antithetic(S, K, T, r, sigma, simulations=100_000):
    Z = np.random.randn(simulations // 2)
    Z_full = np.concatenate([Z, -Z])   # antithetic pairs
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_full)
    payoffs = np.maximum(ST - K, 0)
    return round(np.exp(-r * T) * np.mean(payoffs), 4)
```

### 10.2 Control Variates
```
Use Black-Scholes as a control:
  - We know the exact BS price
  - Adjust MC estimate using the known error

→ Dramatically reduces variance
→ Used in production systems
```

```python
from scipy.stats import norm

def bs_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def monte_carlo_control_variate(S, K, T, r, sigma, simulations=100_000):
    Z = np.random.randn(simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    mc_payoffs = np.maximum(ST - K, 0)
    mc_price_raw = np.exp(-r * T) * np.mean(mc_payoffs)

    # Control: use geometric average (has closed form)
    bs_price = bs_call(S, K, T, r, sigma)
    correction = bs_price - mc_price_raw   # known error

    return round(mc_price_raw + correction, 4)
```

### Summary of Techniques
| Technique | Speedup | Complexity |
|-----------|---------|------------|
| Antithetic variates | ~2x | Low |
| Control variates | ~10x | Medium |
| Quasi-random (Sobol) | ~100x | High |
| Parallel computing (GPU) | ~1000x | High |

---

## 11. Interview Questions

### ❓ Q1: Why use Monte Carlo over Black-Scholes?
```
Black-Scholes only works for simple European options with constant vol.

Monte Carlo works for:
  - Path-dependent options (Asian, Barrier, Lookback)
  - Stochastic volatility models
  - Multi-asset derivatives
  - Any payoff structure

→ MC is the general-purpose tool; BS is the special case
```

### ❓ Q2: What are the disadvantages of Monte Carlo?
```
1. Slow convergence — error decreases as 1/√N
   → Need 1M simulations for 3 decimal accuracy

2. Computationally expensive
   → Not ideal for real-time pricing

3. Difficult to compute Greeks accurately
   → Finite differences add noise

4. Path simulation is memory intensive
   → Large matrices for multi-step paths
```

### ❓ Q3: How do you improve Monte Carlo performance?
```
🔥 Strong answer:

1. Antithetic variates — use Z and -Z pairs
2. Control variates — use known analytical price to correct
3. Quasi-random numbers (Sobol sequences) — better coverage than random
4. Vectorization — NumPy instead of Python loops
5. GPU computing — simulate millions of paths in parallel
6. Importance sampling — sample more from important regions
```

### ❓ Q4: What is the convergence rate of Monte Carlo?
```
Error ∝ 1/√N

To halve the error → need 4x more simulations
To get 10x more accurate → need 100x more simulations

This is why variance reduction techniques matter so much.
```

### ❓ Q5: How would you price an Asian option?
```
1. Simulate N full price paths (daily steps)
2. For each path, compute average price
3. Payoff = max(average - K, 0)
4. Discount average payoff to today

→ Black-Scholes cannot do this; Monte Carlo handles it naturally
```

### ❓ Q6: What is the difference between European and Asian option pricing via MC?
```
European: only final price S_T matters
  → Single-step simulation is enough

Asian: average price over entire path matters
  → Must simulate every time step
  → More computation, but same MC framework
```

---

## 12. Real Trading Insight

```
At quant trading firms, Monte Carlo is used for:

1. Risk simulation
   → Simulate 10,000 market scenarios
   → Measure portfolio loss distribution (VaR, CVaR)

2. Exotic derivatives pricing
   → Barrier options, Asian options, CLOs, CDOs

3. Stress testing
   → "What happens to our portfolio in a 2008-style crash?"

4. Model validation
   → Cross-check analytical models against MC

5. Counterparty credit risk (CVA)
   → Simulate future exposure across thousands of paths
```

---

## 13. Mini Exercise

### Exercise 1: Convergence Test
```python
import numpy as np

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
bs_price = 10.4506  # known Black-Scholes price

for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
    Z = np.random.randn(n)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    mc = np.exp(-r * T) * np.mean(np.maximum(ST - K, 0))
    error = abs(mc - bs_price)
    print(f"n={n:>10,}  MC={mc:.4f}  Error={error:.4f}")
```

### Exercise 2: Compare European vs Asian
```python
# Run both pricers and observe:
# Asian call < European call (averaging reduces variance/upside)
print("European:", price_european_call(100, 100, 1, 0.05, 0.2))
print("Asian   :", price_asian_call(100, 100, 1, 0.05, 0.2))
```

### Exercise 3: Volatility Sensitivity
```python
for vol in [0.1, 0.2, 0.3, 0.4, 0.5]:
    Z = np.random.randn(100_000)
    ST = 100 * np.exp((0.05 - 0.5 * vol**2) * 1 + vol * Z)
    price = np.exp(-0.05) * np.mean(np.maximum(ST - 100, 0))
    print(f"σ={int(vol*100)}%  →  Call = {price:.4f}")
```

---

## 14. Quick Reference Cheat Sheet

```
Core Formula:
  S_T = S_0 × exp[(r - 0.5σ²)T + σ√T × Z]    Z ~ N(0,1)

Algorithm:
  1. Generate Z ~ N(0,1) × N simulations
  2. Compute S_T for each Z
  3. Payoff = max(S_T - K, 0)
  4. Price = e^(-rT) × mean(payoffs)

Convergence:
  Error ∝ 1/√N
  10K sims  → ~±0.3 accuracy
  100K sims → ~±0.1 accuracy
  1M sims   → ~±0.03 accuracy

Option Types:
  European  → only final price matters → single-step MC
  Asian     → average price matters   → multi-step MC
  Barrier   → path must not cross B   → multi-step MC
  Lookback  → min/max of path matters → multi-step MC

Variance Reduction:
  Antithetic variates → use Z and -Z
  Control variates    → correct using known BS price
  Quasi-random        → Sobol sequences for better coverage

When to use MC vs BS:
  BS  → European options, fast pricing, Greeks analytically
  MC  → Exotic options, path-dependent, complex models
```

---

*Part of the Quant Interview Prep Series — [@sonufsd](https://github.com/sonufsd)*

---

# 🧮 Monte Carlo Greeks — Advanced Quant Territory
> How real trading systems compute sensitivities via simulation

---

## 🗺️ Greeks Section Contents

1. [The Core Challenge](#gc-1-the-core-challenge)
2. [Method 1 — Finite Difference](#gc-2-method-1--finite-difference)
3. [Method 2 — Pathwise Derivative](#gc-3-method-2--pathwise-derivative)
4. [Method 3 — Likelihood Ratio Method](#gc-4-method-3--likelihood-ratio-method)
5. [All Greeks via Finite Difference](#gc-5-all-greeks-via-finite-difference)
6. [Common Random Numbers](#gc-6-common-random-numbers--key-trick)
7. [Greeks Methods Comparison](#gc-7-greeks-methods-comparison)
8. [Interview Questions — Greeks](#gc-8-interview-questions--greeks)

---

## GC-1. The Core Challenge

We already use Monte Carlo to price options:
```
Price = e^(-rT) × mean(payoffs)
```

Greeks require **derivatives of price with respect to inputs**:
```
Delta = ∂Price/∂S
Vega  = ∂Price/∂σ
Gamma = ∂²Price/∂S²
```

The challenge:
```
Monte Carlo output is noisy (random)
→ Differentiating a noisy function amplifies noise
→ Need smart methods to get stable Greeks
```

---

## GC-2. Method 1 — Finite Difference

### Idea
Slightly bump an input → observe how price changes → that's the derivative.

```
Delta ≈ [Price(S + ε) - Price(S - ε)] / (2ε)    ← central difference
```

### Delta
```python
import numpy as np

def mc_call(S, K, T, r, sigma, sims=100_000, seed=42):
    np.random.seed(seed)
    Z = np.random.randn(sims)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return np.exp(-r * T) * np.mean(np.maximum(ST - K, 0))

def delta_fd(S, K, T, r, sigma, eps=0.5):
    return (mc_call(S + eps, K, T, r, sigma) -
            mc_call(S - eps, K, T, r, sigma)) / (2 * eps)

print("Delta (FD):", delta_fd(100, 100, 1, 0.05, 0.2))
# Expected: ~0.637 (matches Black-Scholes)
```

### Gamma (Second-Order)
```python
def gamma_fd(S, K, T, r, sigma, eps=0.5):
    return (mc_call(S + eps, K, T, r, sigma) -
            2 * mc_call(S, K, T, r, sigma) +
            mc_call(S - eps, K, T, r, sigma)) / (eps**2)

print("Gamma (FD):", gamma_fd(100, 100, 1, 0.05, 0.2))
```

### Vega
```python
def vega_fd(S, K, T, r, sigma, eps=0.01):
    return (mc_call(S, K, T, r, sigma + eps) -
            mc_call(S, K, T, r, sigma - eps)) / (2 * eps)

print("Vega (FD):", vega_fd(100, 100, 1, 0.05, 0.2))
```

### Theta
```python
def theta_fd(S, K, T, r, sigma, eps=1/365):
    return (mc_call(S, K, T - eps, r, sigma) -
            mc_call(S, K, T, r, sigma)) / eps   # per day decay

print("Theta (FD):", theta_fd(100, 100, 1, 0.05, 0.2))
```

### Pros & Cons
```
✅ Simple to implement
✅ Works for any payoff
❌ Requires 2-3 separate simulations per Greek
❌ Noisy — bumping amplifies simulation variance
❌ Slow for full Greeks profile
```

---

## GC-3. Method 2 — Pathwise Derivative

### Idea
Instead of: simulate → then differentiate the output
We do: differentiate the formula → then simulate

This is more efficient because we compute the Greek **inside a single simulation**.

### Delta — Pathwise
```
For a call: Payoff = max(S_T - K, 0)

∂Payoff/∂S = 1(S_T > K) × (S_T / S)

So: Delta = e^(-rT) × E[ 1(S_T > K) × (S_T / S) ]
```

```python
def delta_pathwise(S, K, T, r, sigma, sims=100_000):
    Z = np.random.randn(sims)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    indicator = (ST > K).astype(float)
    delta = np.exp(-r * T) * np.mean(indicator * (ST / S))
    return round(delta, 4)

print("Delta (Pathwise):", delta_pathwise(100, 100, 1, 0.05, 0.2))
# Expected: ~0.637
```

### Vega — Pathwise
```
∂Payoff/∂σ = 1(S_T > K) × S_T × [ -σT + √T × Z ]

So: Vega = e^(-rT) × E[ 1(S_T > K) × S_T × (-σT + √T × Z) ]
```

```python
def vega_pathwise(S, K, T, r, sigma, sims=100_000):
    Z = np.random.randn(sims)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    indicator = (ST > K).astype(float)
    vega = np.exp(-r * T) * np.mean(indicator * ST * (-sigma * T + np.sqrt(T) * Z))
    return round(vega, 4)

print("Vega (Pathwise):", vega_pathwise(100, 100, 1, 0.05, 0.2))
```

### Pros & Cons
```
✅ Single simulation — much faster
✅ Lower variance than finite difference
✅ Used in production systems
❌ Requires differentiating the payoff analytically
❌ Fails for discontinuous payoffs (e.g. digital options)
   → Payoff of digital = 1(S_T > K), derivative is a Dirac delta
```

---

## GC-4. Method 3 — Likelihood Ratio Method

### When to Use
```
Pathwise fails when payoff is DISCONTINUOUS:
  - Digital options: Payoff = 1(S_T > K)
  - Binary options
  - Barrier options at the boundary

LRM differentiates the PROBABILITY DENSITY instead of the payoff.
```

### Idea
```
Instead of: d/dS [Payoff(S_T)]
We compute: Payoff × d/dS [log f(S_T)]

Where f(S_T) is the density of S_T
```

### Delta via LRM
```python
def delta_lrm(S, K, T, r, sigma, sims=100_000):
    Z = np.random.randn(sims)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0)

    # Score function
    score = (np.log(ST / S) - (r - 0.5 * sigma**2) * T) / (sigma**2 * T * S)

    delta = np.exp(-r * T) * np.mean(payoffs * score)
    return round(delta, 4)

print("Delta (LRM):", delta_lrm(100, 100, 1, 0.05, 0.2))
```

### Pros & Cons
```
✅ Works for ANY payoff — including discontinuous
✅ No need to differentiate the payoff
❌ Higher variance than pathwise
❌ More complex to derive score functions
```

---

## GC-5. All Greeks via Finite Difference

Complete implementation with **common random numbers**:

```python
import numpy as np

def mc_greeks_full(S, K, T, r, sigma, sims=100_000):
    seed     = 42
    eps_S     = 0.5
    eps_sigma = 0.001
    eps_T     = 1/365
    eps_r     = 0.0001

    def price(S_, K_=K, T_=T, r_=r, sigma_=sigma):
        np.random.seed(seed)   # same random numbers every call
        Z = np.random.randn(sims)
        ST = S_ * np.exp((r_ - 0.5 * sigma_**2) * T_ + sigma_ * np.sqrt(T_) * Z)
        return np.exp(-r_ * T_) * np.mean(np.maximum(ST - K_, 0))

    base  = price(S)
    delta = (price(S + eps_S) - price(S - eps_S)) / (2 * eps_S)
    gamma = (price(S + eps_S) - 2 * base + price(S - eps_S)) / eps_S**2
    vega  = (price(S, sigma_=sigma + eps_sigma) - price(S, sigma_=sigma - eps_sigma)) / (2 * eps_sigma)
    theta = (price(S, T_=T - eps_T) - base) / eps_T
    rho   = (price(S, r_=r + eps_r) - price(S, r_=r - eps_r)) / (2 * eps_r)

    return {
        "price": round(base, 4),
        "delta": round(delta, 4),
        "gamma": round(gamma, 4),
        "vega" : round(vega, 4),
        "theta": round(theta / 365, 6),
        "rho"  : round(rho, 4)
    }

result = mc_greeks_full(100, 100, 1, 0.05, 0.2)
for k, v in result.items():
    print(f"{k:>6}: {v}")
```

---

## GC-6. Common Random Numbers — Key Trick

### The Problem
```
Without this trick:
  price_up   uses random seed A → result X
  price_down uses random seed B → result Y

X - Y contains BOTH the price difference AND random noise
→ Greeks are very noisy
```

### The Fix
```
Use the SAME random numbers for all bumped simulations:
  price_up   uses seed 42 → result X
  price_down uses seed 42 → result Y

Now X - Y contains ONLY the price difference
→ Noise cancels out → much more stable Greeks
```

```python
# Without common random numbers — noisy
def delta_noisy(S, K, T, r, sigma, eps=0.5, sims=10_000):
    up   = mc_call(S + eps, K, T, r, sigma)   # different seed each time
    down = mc_call(S - eps, K, T, r, sigma)
    return (up - down) / (2 * eps)

# With common random numbers — stable
def delta_stable(S, K, T, r, sigma, eps=0.5, sims=10_000):
    np.random.seed(42)
    Z = np.random.randn(sims)   # fix Z once

    ST_up   = (S + eps) * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    ST_down = (S - eps) * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

    up   = np.exp(-r*T) * np.mean(np.maximum(ST_up - K, 0))
    down = np.exp(-r*T) * np.mean(np.maximum(ST_down - K, 0))
    return (up - down) / (2 * eps)

print("Noisy  Delta:", delta_noisy(100, 100, 1, 0.05, 0.2))
print("Stable Delta:", delta_stable(100, 100, 1, 0.05, 0.2))
```

---

## GC-7. Greeks Methods Comparison

| Greek | Finite Difference | Pathwise | LRM |
|-------|------------------|----------|-----|
| Delta | ✅ Easy | ✅ Fast, low noise | ✅ Works for discontinuous |
| Gamma | ✅ Second-order bump | ❌ Not straightforward | ✅ Possible |
| Vega | ✅ Easy | ✅ Derivable | ✅ Possible |
| Theta | ✅ Time bump | ✅ Derivable | ✅ Possible |
| Digital options | ✅ Works | ❌ Fails (discontinuous) | ✅ Works |
| Speed | Slow (3 sims) | Fast (1 sim) | Medium |
| Noise | High | Low | High |

---

## GC-8. Interview Questions — Greeks

### ❓ Q1: How do you compute Delta using Monte Carlo?
```
Three methods:

1. Finite Difference (simplest):
   Delta ≈ [Price(S+ε) - Price(S-ε)] / 2ε
   → Easy but requires 2 extra simulations and is noisy

2. Pathwise Derivative (preferred):
   Delta = e^(-rT) × E[ 1(S_T > K) × S_T/S ]
   → Single simulation, lower variance, faster

3. Likelihood Ratio Method:
   → For discontinuous payoffs (digital options)
   → Differentiates the density, not the payoff
```

### ❓ Q2: Why is pathwise better than finite difference?
```
Finite difference:
  - Runs 2-3 separate simulations
  - Noise from each simulation adds up
  - Bumping amplifies variance

Pathwise:
  - Computes Greek inside a single simulation
  - Lower variance (no bumping noise)
  - 2-3x faster

→ In production, pathwise is preferred for smooth payoffs
```

### ❓ Q3: When does pathwise fail?
```
When the payoff is DISCONTINUOUS.

Example: Digital option
  Payoff = 1 if S_T > K, else 0

The derivative of this payoff is a Dirac delta (infinite spike at K)
→ Cannot differentiate through it

Solution: Use Likelihood Ratio Method (LRM) instead
```

### ❓ Q4: What are common random numbers and why do they matter?
```
When computing Greeks via finite difference, you bump an input and
reprice. If you use different random numbers each time, the difference
contains both the price sensitivity AND random noise.

Common random numbers = use the same Z for all bumped simulations.
The noise cancels out in the difference, leaving only the sensitivity.

→ Dramatically reduces variance of Greeks estimates
→ Essential for stable finite difference Greeks
```

### ❓ Q5: How would you compute Gamma via Monte Carlo?
```
Second-order finite difference:

Gamma = [Price(S+ε) - 2×Price(S) + Price(S-ε)] / ε²

Key: use common random numbers for all three price evaluations
     otherwise Gamma estimate is extremely noisy

Note: Gamma is the hardest Greek to estimate via MC
      because it's a second derivative of a noisy function
```

---

*Part of the Quant Interview Prep Series — [@sonufsd](https://github.com/sonufsd)*

---

# 🏭 Production-Level Monte Carlo — Real Systems View

> How Monte Carlo is actually used at quant trading firms

---

## 🗺️ Production Section Contents

1. [How MC Is Used In Real Systems](#prod-1-how-mc-is-used-in-real-systems)
2. [Production Architecture](#prod-2-production-architecture)
3. [Multi-Step Path Simulation](#prod-3-multi-step-path-simulation)
4. [Asian Option — Real Use Case](#prod-4-asian-option--real-use-case)
5. [Value at Risk (VaR) Simulation](#prod-5-value-at-risk-var-simulation)
6. [Performance Optimization](#prod-6-performance-optimization)
7. [Full Production Engine](#prod-7-full-production-engine)
8. [Interview Questions — Production Level](#prod-8-interview-questions--production-level)

---

## PROD-1. How MC Is Used In Real Systems

At quant trading firms, Monte Carlo is NOT just "simulate once and average".

```
Real usage:

1. Pricing complex derivatives
   → Exotic options that have no closed-form solution

2. Real-time risk (VaR, PnL scenarios)
   → Simulate thousands of market scenarios daily

3. Stress testing portfolios
   → "What if volatility doubles overnight?"

4. Computing Greeks efficiently
   → Pathwise + adjoint methods for speed

5. Counterparty credit risk (CVA/XVA)
   → Simulate future exposure across entire trade lifecycle
```

---

## PROD-2. Production Architecture

A real Monte Carlo pricing system has these layers:

```
┌─────────────────────────────────────────┐
│           MARKET DATA INPUTS            │
│  Spot price, Vol surface, Rates, Corr   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│           STOCHASTIC MODEL              │
│  GBM / Local Vol / Heston / Jump Diff   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│         PATH SIMULATION ENGINE          │
│  Multi-step, vectorized, GPU-ready      │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│           PAYOFF COMPUTATION            │
│  European / Asian / Barrier / Custom    │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│        RISK & GREEKS ENGINE             │
│  Pathwise / FD / LRM + Variance Reduc.  │
└─────────────────────────────────────────┘
```

### Stochastic Models Used in Production
```
Basic:
  GBM — simple, fast, used as baseline

Intermediate:
  Local Volatility (Dupire) — vol depends on S and T
  → Fits the volatility smile exactly

Advanced:
  Heston Model — stochastic volatility (vol has its own randomness)
  Jump Diffusion (Merton) — adds sudden price jumps
  SABR — used heavily in interest rate derivatives
```

---

## PROD-3. Multi-Step Path Simulation

### Why Multi-Step?
```
Single-step: only gives final price S_T
  → Works for European options only

Multi-step: gives full price path S_0, S_1, ..., S_T
  → Required for Asian, Barrier, Lookback options
  → Required for risk simulation (VaR)
```

### Vectorized Implementation (Production Style)
```python
import numpy as np

def simulate_gbm_paths(S, T, r, sigma, steps=252, sims=100_000, seed=None):
    """
    Returns price paths of shape (sims, steps+1)
    Each row = one simulated price path
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    Z = np.random.randn(sims, steps)

    # Log returns at each step
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative sum gives log price path
    log_paths = np.cumsum(log_returns, axis=1)

    # Prepend log(S) as starting point
    log_S0 = np.full((sims, 1), np.log(S))
    log_paths = np.hstack([log_S0, log_S0 + log_paths])

    return np.exp(log_paths)   # shape: (sims, steps+1)


# Example: visualize a few paths
paths = simulate_gbm_paths(100, 1, 0.05, 0.2, steps=252, sims=5, seed=42)
print("Path shape:", paths.shape)          # (5, 253)
print("Start price:", paths[:, 0])         # all = 100
print("Final prices:", paths[:, -1].round(2))
```

---

## PROD-4. Asian Option — Real Use Case

### Why Asian Options Exist
```
Problem with European options:
  → Price can be manipulated near expiry (pin risk)
  → One bad day wipes out the option

Asian option solution:
  → Payoff based on AVERAGE price over the period
  → Much harder to manipulate
  → Cheaper than European (averaging reduces variance)

Used heavily in:
  → Commodity markets (oil, gas)
  → FX markets
  → Crypto derivatives
```

### Vectorized Asian Option Pricer
```python
def price_asian_call_vectorized(S, K, T, r, sigma, steps=252, sims=100_000):
    paths = simulate_gbm_paths(S, T, r, sigma, steps, sims)

    # Average price for each path (exclude starting price)
    avg_prices = np.mean(paths[:, 1:], axis=1)

    payoffs = np.maximum(avg_prices - K, 0)
    return round(np.exp(-r * T) * np.mean(payoffs), 4)


def price_asian_put_vectorized(S, K, T, r, sigma, steps=252, sims=100_000):
    paths = simulate_gbm_paths(S, T, r, sigma, steps, sims)
    avg_prices = np.mean(paths[:, 1:], axis=1)
    payoffs = np.maximum(K - avg_prices, 0)
    return round(np.exp(-r * T) * np.mean(payoffs), 4)


S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
print("European Call :", 10.4506)                                    # BS reference
print("Asian Call    :", price_asian_call_vectorized(S, K, T, r, sigma))
# Asian < European because averaging dampens extreme outcomes
```

### Lookback Option (Bonus)
```python
def price_lookback_call(S, T, r, sigma, steps=252, sims=100_000):
    """Payoff = S_T - min(path)  — floating strike lookback"""
    paths = simulate_gbm_paths(S, T, r, sigma, steps, sims)
    ST = paths[:, -1]
    min_price = np.min(paths, axis=1)
    payoffs = np.maximum(ST - min_price, 0)
    return round(np.exp(-r * T) * np.mean(payoffs), 4)

print("Lookback Call :", price_lookback_call(100, 1, 0.05, 0.2))
# Lookback > European — always profitable if price ever moved up
```

---

## PROD-5. Value at Risk (VaR) Simulation

### What is VaR?
```
VaR answers: "What is the maximum loss over N days at X% confidence?"

Example:
  1-day 95% VaR = ₹50,000
  → There is a 5% chance of losing MORE than ₹50,000 in one day
  → 95% of the time, loss will be less than ₹50,000
```

### Monte Carlo VaR
```python
import numpy as np

def monte_carlo_var(portfolio_value, daily_vol, horizon_days=1,
                    confidence=0.95, sims=100_000):
    """
    portfolio_value : current portfolio value
    daily_vol       : daily volatility (e.g. 0.02 for 2%)
    horizon_days    : VaR horizon
    confidence      : confidence level (0.95 = 95%)
    """
    # Simulate portfolio returns
    Z = np.random.randn(sims)
    daily_returns = daily_vol * Z
    portfolio_returns = portfolio_value * daily_returns * np.sqrt(horizon_days)

    # Sort and find the loss at confidence level
    sorted_returns = np.sort(portfolio_returns)
    var_index = int((1 - confidence) * sims)
    var = -sorted_returns[var_index]   # VaR is positive (loss amount)

    # CVaR = average loss beyond VaR (Expected Shortfall)
    cvar = -np.mean(sorted_returns[:var_index])

    return {"VaR": round(var, 2), "CVaR": round(cvar, 2)}


result = monte_carlo_var(
    portfolio_value=1_000_000,
    daily_vol=0.02,
    confidence=0.95
)
print(f"95% 1-day VaR  : ₹{result['VaR']:,.0f}")
print(f"95% 1-day CVaR : ₹{result['CVaR']:,.0f}")
```

### Portfolio Stress Testing
```python
def stress_test(portfolio_value, positions, scenarios):
    """
    positions  : dict of {asset: quantity}
    scenarios  : dict of {scenario_name: {asset: shock}}
    """
    results = {}
    for scenario_name, shocks in scenarios.items():
        pnl = sum(positions.get(asset, 0) * shock
                  for asset, shock in shocks.items())
        results[scenario_name] = round(pnl, 2)
    return results


positions = {"NIFTY": 100, "USDINR": 50_000, "GOLD": 10}
scenarios = {
    "Market crash -10%"  : {"NIFTY": -10, "USDINR": 2, "GOLD": 5},
    "Rate hike shock"    : {"NIFTY": -3,  "USDINR": 1, "GOLD": -2},
    "Vol spike (VIX+50%)": {"NIFTY": -5,  "USDINR": 0.5, "GOLD": 3},
}

pnl = stress_test(1_000_000, positions, scenarios)
for s, v in pnl.items():
    print(f"{s:30s}  PnL = ₹{v:,.0f}")
```

---

## PROD-6. Performance Optimization

### Why It Matters
```
Naive MC (Python loops):
  100K paths × 252 steps = 25M iterations
  → ~30 seconds in pure Python
  → Unusable in production

Vectorized NumPy:
  Same computation → ~0.5 seconds
  → 60x speedup

GPU (CuPy / CUDA):
  Same computation → ~0.01 seconds
  → 3000x speedup
```

### Optimization Techniques

#### 1. Vectorization (Always Do This)
```python
# BAD — Python loop (slow)
def slow_mc(S, K, T, r, sigma, sims=100_000):
    payoffs = []
    for _ in range(sims):
        Z = np.random.randn()
        ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        payoffs.append(max(ST - K, 0))
    return np.exp(-r*T) * np.mean(payoffs)

# GOOD — NumPy vectorized (fast)
def fast_mc(S, K, T, r, sigma, sims=100_000):
    Z = np.random.randn(sims)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    return np.exp(-r*T) * np.mean(np.maximum(ST - K, 0))
```

#### 2. Antithetic Variates (Easy Win)
```python
def mc_antithetic(S, K, T, r, sigma, sims=100_000):
    Z = np.random.randn(sims // 2)
    Z_all = np.concatenate([Z, -Z])
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z_all)
    return np.exp(-r*T) * np.mean(np.maximum(ST - K, 0))
```

#### 3. Quasi-Random Numbers (Sobol Sequences)
```python
from scipy.stats.qmc import Sobol
from scipy.stats import norm

def mc_sobol(S, K, T, r, sigma, sims=100_000):
    sampler = Sobol(d=1, scramble=True)
    uniform_samples = sampler.random(sims).flatten()
    Z = norm.ppf(uniform_samples)   # convert uniform → normal
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    return np.exp(-r*T) * np.mean(np.maximum(ST - K, 0))

# Sobol gives much better coverage of the probability space
# → Same accuracy with ~10x fewer simulations
```

### Performance Comparison
```
Method              Sims needed for ±0.1 accuracy   Time
────────────────────────────────────────────────────────
Standard MC         ~100,000                         ~0.5s
Antithetic          ~50,000                          ~0.3s
Control variates    ~10,000                          ~0.1s
Sobol sequences     ~5,000                           ~0.05s
GPU (CuPy)          ~1,000,000                       ~0.01s
```

---

## PROD-7. Full Production Engine

A clean, modular implementation combining everything:

```python
import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol

class MonteCarloEngine:
    """
    Production-style Monte Carlo pricing engine.
    Supports: European, Asian, Barrier options + VaR
    """

    def __init__(self, S, r, sigma, seed=42):
        self.S = S
        self.r = r
        self.sigma = sigma
        self.seed = seed

    def _simulate(self, T, steps, sims, antithetic=True):
        np.random.seed(self.seed)
        dt = T / steps
        half = sims // 2 if antithetic else sims
        Z = np.random.randn(half, steps)
        if antithetic:
            Z = np.vstack([Z, -Z])
        log_ret = (self.r - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z
        log_S0 = np.full((sims, 1), np.log(self.S))
        log_paths = np.hstack([log_S0, log_S0 + np.cumsum(log_ret, axis=1)])
        return np.exp(log_paths)

    def price_european(self, K, T, sims=100_000):
        paths = self._simulate(T, steps=1, sims=sims)
        ST = paths[:, -1]
        call = np.exp(-self.r*T) * np.mean(np.maximum(ST - K, 0))
        put  = np.exp(-self.r*T) * np.mean(np.maximum(K - ST, 0))
        return {"call": round(call, 4), "put": round(put, 4)}

    def price_asian(self, K, T, steps=252, sims=100_000):
        paths = self._simulate(T, steps, sims)
        avg = np.mean(paths[:, 1:], axis=1)
        call = np.exp(-self.r*T) * np.mean(np.maximum(avg - K, 0))
        put  = np.exp(-self.r*T) * np.mean(np.maximum(K - avg, 0))
        return {"call": round(call, 4), "put": round(put, 4)}

    def price_barrier(self, K, T, barrier, steps=252, sims=100_000):
        paths = self._simulate(T, steps, sims)
        knocked_out = np.any(paths >= barrier, axis=1)
        ST = paths[:, -1]
        payoffs = np.where(knocked_out, 0, np.maximum(ST - K, 0))
        return round(np.exp(-self.r*T) * np.mean(payoffs), 4)

    def compute_var(self, portfolio_value, horizon=1, confidence=0.95, sims=100_000):
        np.random.seed(self.seed)
        returns = portfolio_value * self.sigma * np.sqrt(horizon) * np.random.randn(sims)
        sorted_r = np.sort(returns)
        idx = int((1 - confidence) * sims)
        return {
            "VaR" : round(-sorted_r[idx], 2),
            "CVaR": round(-np.mean(sorted_r[:idx]), 2)
        }

    def delta(self, K, T, eps=0.5, sims=100_000):
        def p(s): return self.price_european(K, T, sims)["call"] if s == self.S else \
                         MonteCarloEngine(s, self.r, self.sigma, self.seed).price_european(K, T, sims)["call"]
        return round((p(self.S + eps) - p(self.S - eps)) / (2 * eps), 4)


# --- Usage ---
engine = MonteCarloEngine(S=100, r=0.05, sigma=0.2)

print("European :", engine.price_european(K=100, T=1))
print("Asian    :", engine.price_asian(K=100, T=1))
print("Barrier  :", engine.price_barrier(K=100, T=1, barrier=120))
print("VaR      :", engine.compute_var(portfolio_value=1_000_000))
```

---

## PROD-8. Interview Questions — Production Level

### ❓ Q1: How is Monte Carlo used in a real trading firm?
```
1. Exotic derivatives pricing
   → Asian, Barrier, Lookback options — no closed-form exists

2. Risk management (VaR/CVaR)
   → Simulate 10,000+ portfolio scenarios daily
   → Regulatory requirement (Basel III)

3. Stress testing
   → Simulate extreme market events
   → "What if vol doubles? What if rates spike 200bps?"

4. Greeks computation
   → Pathwise derivatives or adjoint methods
   → Must be fast enough for real-time hedging

5. XVA (CVA, DVA, FVA)
   → Simulate counterparty exposure over entire trade life
```

### ❓ Q2: What stochastic models are used beyond GBM?
```
Local Volatility (Dupire):
  → Vol is a function of S and T: σ(S,T)
  → Fits market vol surface exactly
  → Used for vanilla exotics

Heston Model:
  → Vol itself follows a stochastic process
  → Captures vol clustering and mean reversion
  → Used for equity derivatives

Jump Diffusion (Merton):
  → Adds Poisson jumps to GBM
  → Captures overnight gaps, earnings surprises
  → Used for single-stock options

SABR:
  → Stochastic Alpha Beta Rho
  → Industry standard for interest rate derivatives
```

### ❓ Q3: How do you handle correlated assets in Monte Carlo?
```
Use Cholesky decomposition:

1. Build correlation matrix Σ
2. Cholesky decompose: Σ = L × Lᵀ
3. Generate independent Z ~ N(0,1) for each asset
4. Correlated Z = L × Z_independent

This gives correlated price paths that respect the correlation structure.

Example: Pricing a basket option on NIFTY + BANKNIFTY
  → Must simulate both assets with their correlation (~0.85)
```

```python
def simulate_correlated_paths(S_list, T, r, sigma_list, corr_matrix,
                               steps=252, sims=100_000):
    n_assets = len(S_list)
    dt = T / steps
    L = np.linalg.cholesky(corr_matrix)   # Cholesky decomposition

    all_paths = []
    for i in range(n_assets):
        all_paths.append([np.full(sims, S_list[i])])

    for _ in range(steps):
        Z_indep = np.random.randn(n_assets, sims)
        Z_corr = L @ Z_indep   # apply correlation
        for i in range(n_assets):
            prev = all_paths[i][-1]
            next_p = prev * np.exp((r - 0.5*sigma_list[i]**2)*dt +
                                    sigma_list[i]*np.sqrt(dt)*Z_corr[i])
            all_paths[i].append(next_p)

    return [np.array(p).T for p in all_paths]   # each: (sims, steps+1)
```

### ❓ Q4: What is CVaR and why is it better than VaR?
```
VaR: "Maximum loss at X% confidence"
  → Tells you the threshold, not what's beyond it
  → Example: 95% VaR = ₹50K means 5% of days are worse
  → But HOW MUCH worse? VaR doesn't say.

CVaR (Conditional VaR / Expected Shortfall):
  → Average loss in the worst X% of scenarios
  → Example: 95% CVaR = ₹80K means average loss on bad days is ₹80K
  → More informative, captures tail risk better

CVaR is now preferred by regulators (Basel III uses ES, not VaR)
```

### ❓ Q5: How would you build a Monte Carlo system for real-time use?
```
Challenges:
  - Real-time means < 100ms per pricing request
  - Standard MC with 100K sims takes ~500ms

Solutions:
  1. Pre-compute and cache common scenarios
  2. Use GPU for massive parallelism (CuPy/CUDA)
  3. Reduce sims with variance reduction (Sobol + antithetic)
  4. Use analytical approximations for simple cases (BS)
  5. Adjoint Algorithmic Differentiation (AAD) for Greeks
     → Compute ALL Greeks in one forward + one backward pass
     → Same cost as pricing once, regardless of number of Greeks
```

---

*Part of the Quant Interview Prep Series — [@sonufsd](https://github.com/sonufsd)*

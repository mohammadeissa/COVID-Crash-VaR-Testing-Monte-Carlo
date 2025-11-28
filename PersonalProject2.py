import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Portfolio definition --------
tickers = ["SPY"]  # start simple; you can expand later
weights = np.array([1.0])

# -------- Time windows --------
in_sample_start = "2010-01-01"
in_sample_end   = "2019-12-31"

crash_start = "2020-02-19"
crash_end   = "2020-03-23"

#S&P 500 Peak: February 19, 2020
#S&P 500 Trough: March 23, 2020

#Source:
#Federal Reserve Economic Data (FRED) – SP500 series


# -------- Download historical prices --------
# yfinance's default switched to auto_adjust=True, which drops the "Adj Close" column.
data = yf.download(tickers, start=in_sample_start, end=in_sample_end, auto_adjust=False)["Adj Close"]

# Handle single vs multiple tickers shape
if isinstance(data, pd.Series):
    data = data.to_frame()

# -------- Log returns --------
log_returns = np.log(data / data.shift(1)).dropna()

mu_vec = log_returns.mean()       # per-asset mean
cov_mat = log_returns.cov()       # per-asset covariance

# Portfolio daily returns pre-COVID (for sanity + VaR)
pre_covid_portfolio_returns = log_returns.dot(weights)
print(pre_covid_portfolio_returns.describe())


# -------- Actual COVID crash returns for comparison --------
covid_prices = yf.download(tickers, start=crash_start, end=crash_end, auto_adjust=False)["Adj Close"]

if isinstance(covid_prices, pd.Series):
    covid_prices = covid_prices.to_frame()

covid_log_returns = np.log(covid_prices / covid_prices.shift(1)).dropna()
covid_portfolio_returns = covid_log_returns.dot(weights)

# Cumulative return over crash period
actual_covid_cum_return = np.exp(covid_portfolio_returns.sum()) - 1
print("Actual COVID crash cumulative return:", actual_covid_cum_return)


from numpy.random import default_rng

rng = default_rng(42)

T = len(covid_portfolio_returns)    # same length as crash period
N = 10000                           # number of simulation paths

# Cholesky for correlated shocks (even though single asset now, keeps it general)
chol = np.linalg.cholesky(cov_mat)

sim_cum_returns = []

for _ in range(N):
    # Generate T x n_assets normal shocks
    Z = rng.standard_normal(size=(T, len(tickers)))
    # Correlated shocks
    shocks = Z @ chol.T
    # Add mean (broadcast over time)
    sim_log_rets = mu_vec.values * 1 + shocks  # (you can scale dt if needed)
    # Portfolio returns for this path
    sim_port_rets = sim_log_rets @ weights
    # Cumulative return for this path
    cum_ret = np.exp(sim_port_rets.sum()) - 1
    sim_cum_returns.append(cum_ret)

sim_cum_returns = np.array(sim_cum_returns)


# VaR over the crash horizon
VaR_95 = np.percentile(sim_cum_returns, 5)
VaR_99 = np.percentile(sim_cum_returns, 1)

print("Monte Carlo 95% VaR (horizon):", VaR_95)
print("Monte Carlo 99% VaR (horizon):", VaR_99)

# Where does actual COVID crash fall in this distribution?
percentile_covid = (sim_cum_returns < actual_covid_cum_return).mean() * 100
print(f"Actual COVID crash is around the {percentile_covid:.5f}th percentile of simulated outcomes")


# -----------------------------
# COVID-Style Stress Scenario
# -----------------------------

# 1. Raise volatility ~3x (market was WAY jumpier during COVID)
volatility_multiplier = 3.0
# per-asset volatilities from the in-sample covariance
vols = np.sqrt(np.diag(cov_mat))

# 2. Increase correlations so everything moves together
high_corr_value = 0.9
n = len(tickers)

# Create a high-correlation matrix
stressed_corr = np.full((n, n), high_corr_value)
np.fill_diagonal(stressed_corr, 1.0)

# Convert stressed_corr back into a covariance matrix
stressed_cov = np.outer(vols, vols) * stressed_corr
stressed_cov *= (volatility_multiplier ** 2)

# 3. Run Monte Carlo again with stressed covariance
chol_stress = np.linalg.cholesky(stressed_cov)

sim_cum_returns_stress = []

for _ in range(N):
    Z = rng.standard_normal(size=(T, n))
    shocks = Z @ chol_stress.T
    sim_log_rets = mu_vec.values * 1 + shocks
    sim_port_rets = sim_log_rets @ weights
    cum_ret = np.exp(sim_port_rets.sum()) - 1
    sim_cum_returns_stress.append(cum_ret)

sim_cum_returns_stress = np.array(sim_cum_returns_stress)

# New stressed VaR
VaR_95_stress = np.percentile(sim_cum_returns_stress, 5)
VaR_99_stress = np.percentile(sim_cum_returns_stress, 1)

print("COVID-Stress 95% VaR =", VaR_95_stress)
print("COVID-Stress 99% VaR =", VaR_99_stress)

# -----------------------------
# 2008 Crash Window (33 days)
# -----------------------------
crash_2008_start = "2008-09-15"   # Lehman collapse
crash_2008_end   = "2008-10-27"   # major trough period

prices_2008 = yf.download(tickers, start=crash_2008_start, end=crash_2008_end, auto_adjust=False)["Adj Close"]

# Handle single ticker case
if isinstance(prices_2008, pd.Series):
    prices_2008 = prices_2008.to_frame()

log_returns_2008 = np.log(prices_2008 / prices_2008.shift(1)).dropna()
portfolio_returns_2008 = log_returns_2008.dot(weights)

# Cumulative return for the 2008 window
actual_2008_cum_return = np.exp(portfolio_returns_2008.sum()) - 1

print("2008 Crash Cumulative Return:", actual_2008_cum_return)



plt.figure(figsize=(10,6))

# 1. Normal Monte Carlo Distribution
plt.hist(sim_cum_returns, bins=60, alpha=0.5, label="Normal Monte Carlo", density=True)

# 2. COVID-Stress Monte Carlo Distribution
plt.hist(sim_cum_returns_stress, bins=60, alpha=0.5, label="COVID-Stress Monte Carlo", density=True)

# 3. Actual COVID Crash Line
plt.axvline(actual_covid_cum_return, color="red", linestyle="--", linewidth=2,
            label=f"Actual COVID Crash ({actual_covid_cum_return:.2%})")

plt.title("Distribution of Simulated Portfolio Returns\nNormal vs COVID-Stress vs Actual COVID Crash")
plt.xlabel("Cumulative Return")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)

plt.show()



plt.figure(figsize=(10,6))

# Normal Monte Carlo
plt.hist(sim_cum_returns, bins=60, alpha=0.4, label="Normal Monte Carlo", density=True)

# COVID-Stress Monte Carlo
plt.hist(sim_cum_returns_stress, bins=60, alpha=0.4, label="COVID-Stress Monte Carlo", density=True)

# COVID crash
plt.axvline(actual_covid_cum_return, color="red", linestyle="--", linewidth=2,
            label=f"COVID Crash ({actual_covid_cum_return:.2%})")

# 2008 crash
plt.axvline(actual_2008_cum_return, color="purple", linestyle="-.", linewidth=2,
            label=f"2008 Crash ({actual_2008_cum_return:.2%})")

plt.title("Normal Monte Carlo vs Stress Monte Carlo\nCOVID Crash vs 2008 Crash")
plt.xlabel("Cumulative Return")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

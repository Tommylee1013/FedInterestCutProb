# FedInterestCutProb
A Python-based Federal Reserve interest-rate decision probability estimator (FedWatch-style).

This project reconstructs historical and real-time probabilities of rate cuts, holds, and hikes for each FOMC meeting using:
- ZQ Federal Funds Futures (CBOT)
- EFFR time series
- FOMC meeting calendar
- Contract-expiry smoothing
- Robust data retrieval through TradingView

The goal is to provide a transparent, reproducible alternative to CME FedWatch.

---

## Overview
Federal Funds Futures embed market expectations of future policy rates.  
FedWatch interprets these futures prices as probabilities of discrete rate outcomes (e.g., -25bp, 0bp, +25bp).  
This project implements the same methodology in Python, with additions such as:
- Historical reconstruction from 2000 onward  
- Single-contract or two-contract blended models  
- Smoothing near contract expiry  
- Automatic ZQ symbol generation  
- Retry-safe TradingView data collection

It produces a continuous daily probability time series for the “next upcoming” FOMC meeting.

---

## Features
- Full historical reconstruction (2000 → present)
- Single-contract model with optional moving-average smoothing near expiry
- Optional blended model: front + next ZQ contracts
- Robust retry-safe TradingView fetch (`safe_get_hist`)
- Automatic contract symbol generation (ZQ + MonthCode + Year)
- Meeting-by-meeting probability panels
- Final unified daily probability series

---

## Methodology

### 1. Futures Price → Implied Rate
ZQ price definition:

$$P_t = 100 - \text{Average Effective Federal Funds Rate for the month}$$

thus

$$\text{ImplRate} = 100 - P_t$$

### 2. Monthly Average Rate Decomposition
The expected average rate in the meeting month is constructed from:
- Known EFFR values up to the meeting date  
- Possible terminal rates after the meeting

For each bp scenario:

$$\text{TerminalRate}_i = \text{CurrentRate} + (\text{bp}_i / 100)$$

The monthly average rate under scenario i:

$$\text{AvgRate}_i = \text{Weighted}\left(\text{EFFR_before_meeting, TerminalRate_i_after_meeting}\right)$$

### 3. Probability Solving
Given the futures-implied average rate:

$$\sum {p_i} \cdot \text{AvgRate}_i = \text{ImplRate}$$

with constraints :

- $p_i \geq 0$
- $\sum {p_i} = 1$

The model solves this linear system for probabilities.

---

## Contract-Expiry Smoothing

Near contract expiry, ZQ futures often distort.  
This project supports two solutions:

### Option A: Moving Average Smoothing

Applied only when days-to-expiry ≤ `ma_apply_days`.

Parameters:

```aiignore
ma_window = 5
ma_apply_days = 20
```

### Option B: Two-Contract Blending (Optional)
Weighted blend of front and next contracts:

$$P = w_{\text{front}} * P_{\text{front}} + w_{\text{next}} * P_{\text{next}}$$

Weights taper as the meeting approaches.

---

## Code Structure

````aiignore
fed_interest_prob/
│
├── contracts.py
├── data_loader.py
└── utils.py
````

---

## Usage Example

```{cmd}
python main.py
```

Example output:

```aiignore
date         meeting_date   prob_-25   prob_0   prob_25
--------------------------------------------------------
2024-11-01   2024-12-18      0.12       0.78     0.10
2024-11-02   2024-12-18      0.11       0.79     0.10
...
```

## Limitations
- Requires TradingView account access for futures data
- Crisis periods may produce noisy pricing near expiry
- Some emergency meetings (e.g., 2020-03-03, 2020-03-15) do not follow standard FedWatch logic
- Historical reconstruction depends on available ZQ price history

## Future Work

- Incorporating SOFR futures
- Real-time websocket streaming module
- Comparison utilities vs official CME FedWatch data

## License / Disclaimer

This project is for research and educational purposes only.
It is not affiliated with CME Group, the Federal Reserve, or TradingView.
No investment advice is provided.
# Executive Summary — Bitcoin Sentiment × Trader Performance
**Primetrade.ai | Data Science Hiring Assignment**

---

## The Question
Does Bitcoin's Fear & Greed Index meaningfully predict trader behavior and outcomes on Hyperliquid perpetuals?

**Short answer: Yes — and the effect is statistically significant (p=0.001).**

---

## Dataset at a Glance
- **211,224 trades** across 32 accounts, full year 2024
- **2,644 sentiment days** (Feb 2018–May 2025) mapped by date
- **246 symbols** traded; HYPE, BTC, ETH, SOL dominate
- **$10.3M total realized PnL** across all accounts

---

## Top 5 Findings

### 1. Extreme Greed = Peak Performance
Win rate hits **46.5%** in Extreme Greed — nearly 10 percentage points above Extreme Fear (37.1%). This is the single strongest sentiment-performance relationship in the data.

### 2. Fear Makes Traders Reckless
During Fear, the average trade size balloons to **$7,816** — 2.5× the $3,112 seen in Extreme Greed. This classic "averaging down" behavior amplifies losses precisely when the market is most hostile.

### 3. Greed Zone = Better Capital Efficiency
Despite lower average trade sizes in the Greed zone ($4,574 vs $7,182 in Fear), average PnL per trade is **9.5% higher** ($53.9 vs $49.2). Smaller bets, better returns.

### 4. Smart Money Sells Into Greed
The Greed zone shows **47,779 SELL trades vs 42,516 BUY** — a distinctly sell-heavy posture. Top performers consistently take profits during euphoria, consistent with classic contrarian positioning.

### 5. Symbol Selection Must Be Regime-Aware
- **Greed regime**: @107 alone generated $2.71M — a momentum-driven asset that thrives on euphoria
- **Fear regime**: HYPE dominates with $1.32M — a more fundamentally-driven asset

---

## Statistical Validation
| Test | Statistic | p-value | Conclusion |
|------|-----------|---------|------------|
| Mann-Whitney U (Fear vs Greed PnL) | 3.73B | 0.001 | Significant ✓ |
| Chi-squared (win rate × sentiment) | — | <0.05 | Significant ✓ |
| Kruskal-Wallis (trade size × sentiment) | — | <0.05 | Significant ✓ |

---

## Recommendations for Primetrade.ai

1. **Build a sentiment-aware position sizing engine** — scale up in Extreme Greed, scale down in Fear
2. **Create symbol watchlists by sentiment regime** — different assets lead in different environments
3. **Flag fear-driven oversizing** — implement automatic position caps during Fear/Extreme Fear
4. **Exploit the SELL-into-greed pattern** — backtest systematic profit-taking rules in high-FGI environments
5. **Use FGI as a risk-management input** — beyond directional signals, use it to set daily drawdown limits

---

## Portfolio Presentation Talking Points

> *"I discovered that traders consistently mismanage risk during Fear periods — taking on 2.5× larger positions precisely when win rates are lowest. This isn't rational risk-taking; it's emotional averaging down. The data suggests a systematic, sentiment-conditioned position sizing strategy could meaningfully improve returns."*

> *"The statistical tests confirm this isn't noise. With p=0.001 on the Mann-Whitney U test, sentiment is a real edge — one that most retail traders don't systematically exploit."*

> *"The top account made $2.14M while using a similar average trade size to median performers. The differentiator isn't capital, it's strategy discipline and sentiment timing."*

---

*Analysis by: [Your Name] | Primetrade.ai Data Science Assignment | 2024*

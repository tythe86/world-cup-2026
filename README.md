# ⚽ FIFA World Cup 2026 — AI Match Prediction Model
AI match prediction model for FIFA World Cup 2026. Elo ratings + XGBoost trained on 50,000+ international matches.
> A complete Python pipeline for predicting match outcomes, generating
> expected goals, and running Monte Carlo tournament simulations.
> Built for YouTube creators who want AI-powered soccer analysis.

---

## 🚀 Quick Start

```bash
# 1. Clone / download this folder
cd fifa_predictor

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Run the demo (downloads data automatically on first run)
python predictor.py

# 4. Predict a specific match
python predictor.py --match "Brazil" "France"

# 5. Full tournament simulation (10,000 runs)
python predictor.py --simulate
```

Charts are saved to the `visuals/` folder — ready to use in videos.

---

## 🗓️ 每日自动预测报告（中文 / GitHub Actions）

本仓库内置一个每天自动运行的 GitHub Action：跑通 Elo + XGBoost 流水线，生成一份**中文**预测报告并提交回仓库。

- **最新报告**：[`reports/预测报告.md`](reports/预测报告.md)
- **历史归档**：[`reports/archive/`](reports/archive)
- **定时**：每天北京时间约 07:07 自动执行（cron `7 23 * * *` UTC）
- **手动触发**：在仓库 **Actions** 页面选 `每日世界杯预测报告（中文）` → `Run workflow`
- **查看本次运行的报告**：点开对应的 Action 运行，摘要（Summary）里直接渲染完整中文报告

报告内容：① AI 实力排名（Elo）② 比赛胜率 / xG 预测 ③ 蒙特卡洛夺冠概率模拟。

### 实时赛程（可选）
比赛期间，设置仓库 Secret `BALLDONTLIE_API_KEY`（[fifa.balldontlie.io](https://fifa.balldontlie.io/) 免费 key）后，报告会自动改用**实时赛程**预测当天及之后未开始的比赛。Key 未配置或失效时自动回退到内置重点对决，不影响报告生成。

```bash
# 在本地手动生成一次报告（可选）
python report_zh.py
# 可调参数：SIM_RUNS（模拟次数，默认 10000）、TOP_N（排名数，默认 20）
```

---


## 📁 Project Structure

```
fifa_predictor/
├── predictor.py            ← Main model (all-in-one)
├── tutorial_notebook.ipynb ← Jupyter walkthrough for YouTube tutorials
├── requirements.txt
├── data/
│   └── results.csv         ← Auto-downloaded on first run
├── models/                 ← Saved model files
└── visuals/                ← Generated charts (PNG, 150 dpi)
    ├── elo_rankings.png
    ├── championship_probs.png
    ├── feature_importance.png
    └── pred_brazil_france.png  (etc.)
```

---

## 🧠 How It Works

### 1. Data
- **Historical matches:** 50,000+ international results from 1990–2026
  - Source: [martj42/international_results](https://github.com/martj42/international_results) (free, no key)
- **Live WC26 schedule:** hardcoded from [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json)
- **During tournament:** plug in [BALLDONTLIE FIFA API](https://fifa.balldontlie.io/) (free tier)

### 2. Elo Rating System
Dynamic Elo ratings updated after every match since 1990.  
Key design choices:
- Tournament importance weights (World Cup k=60, Friendly k=20)
- Home advantage: +100 Elo unless neutral venue
- Goal-difference multiplier (winning 3-0 counts more than 1-0)

### 3. Features Used by the Model

| Feature | Description |
|---|---|
| `elo_diff` | Elo gap between teams (adjusted for venue) |
| `elo_home / elo_away` | Absolute ratings |
| `home_form_scored` | Avg goals scored, last 5 matches |
| `home_form_conceded` | Avg goals conceded, last 5 matches |
| `form_diff` | Net form advantage |
| `h2h_home_winrate` | Head-to-head record, last 10 meetings |
| `is_wc` | Tournament context (World Cup = 1) |
| `neutral` | Neutral venue flag |

### 4. Model
- **Algorithm:** XGBoost multi-class classifier
- **Target:** 3 outcomes — Home Win / Draw / Away Win
- **Validation:** 5-fold stratified cross-validation
- **Expected accuracy:** ~55–58% (benchmark: ~45% for random, ~50% for always-pick-favourite)

### 5. Expected Goals (xG)
Form-based xG estimate adjusted by Elo strength ratio using a
simplified Dixon-Coles-style formula. Good enough for video graphics;
upgrade with StatsBomb open data for shot-level xG.

---

## 🎬 YouTube Video Ideas

| Video | Chart to use | When to publish |
|---|---|---|
| "AI Power Rankings – WC2026" | `elo_rankings.png` | Now (pre-tournament) |
| "Who Will Win the World Cup?" | `championship_probs.png` | 2 weeks before kick-off |
| "Brazil vs Germany – AI Prediction" | `pred_brazil_germany.png` | Day before match |
| "Build a Soccer AI in Python" | `feature_importance.png` | Any time (tutorial) |
| "Was AI Right?" recap | Pred vs actual comparison | Day after match |

---

## 🔌 Adding Live Data (During Tournament)

```python
# Option A: BALLDONTLIE FIFA API (free tier, 2018/2022/2026 data)
import requests
r = requests.get(
    "https://api.balldontlie.io/fifa/v1/matches",
    headers={"Authorization": "YOUR_FREE_API_KEY"},
    params={"season": 2026}
)
matches = r.json()["data"]

# Option B: openfootball JSON (no key required)
r = requests.get(
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)
schedule = r.json()
```

---

## 📈 Improvements for Advanced Videos

- **Shot-level xG:** Pull StatsBomb open data and train a logistic model on shot position + body part
- **Player ratings:** Merge FIFA/SOFIFA ratings as extra features
- **NLP sentiment:** Add Twitter/Reddit fan sentiment score as a feature
- **Live updating:** Re-run Elo after every match and republish predictions
- **Calibration:** Apply Platt scaling to make probabilities better-calibrated

---

## 📄 License
MIT. Use freely for your YouTube channel, course, or SaaS.
Credit appreciated but not required.

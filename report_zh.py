#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026 世界杯每日 AI 预测报告（中文）
====================================
复用 predictor.py 的 Elo + XGBoost 流水线，生成一份中文 Markdown 报告：
  1. AI 实力排名（Elo Top 20）
  2. 今日 / 即将到来的比赛预测（优先用 BALLDONTLIE 实时赛程，失败回退到重点对决）
  3. 蒙特卡洛夺冠概率模拟
  4. 数据来源与 API key 状态说明

运行：
  python report_zh.py
环境变量：
  BALLDONTLIE_API_KEY  （可选）实时赛程 API key；未配置或失效时自动回退
  SIM_RUNS             （可选）蒙特卡洛模拟次数，默认 3000
  TOP_N                （可选）排名展示数量，默认 20
"""
from __future__ import annotations

import os
import sys
import json
import time
import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# 复用同目录下的模型代码
sys.path.insert(0, str(Path(__file__).resolve().parent))
import predictor as P  # noqa: E402

# ──────────────────────────────────────────────
# BALLDONTLIE FIFA World Cup API
# ──────────────────────────────────────────────
# openfootball 公开数据（免费、无需 key）：2026 世界杯完整赛程 + 真实比分
OPENFOOTBALL_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)

# 队名归一化：各种口径（openfootball / 赛程 / FIFA 官方）→ 【历史数据集 martj42】的标准名
# 关键：必须落到历史数据集的队名，否则 Elo 评分查不到（会默认成 1500）
TEAM_NAME_ALIASES = {
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Curacao": "Curaçao",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia &amp; Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Congo DR": "DR Congo",
}

# 球队中文名（覆盖全部 48 支参赛队 + 历史数据集中的别名 + 常见强队）
TEAM_CN = {
    "Mexico": "墨西哥", "South Africa": "南非", "South Korea": "韩国",
    "Czechia": "捷克", "Czech Republic": "捷克", "Canada": "加拿大", "Switzerland": "瑞士",
    "Qatar": "卡塔尔", "Bosnia and Herzegovina": "波黑",
    "Brazil": "巴西", "Morocco": "摩洛哥", "Haiti": "海地",
    "Scotland": "苏格兰", "USA": "美国", "United States": "美国", "Paraguay": "巴拉圭",
    "Australia": "澳大利亚", "Turkey": "土耳其", "Germany": "德国",
    "Curacao": "库拉索", "Curaçao": "库拉索", "Ivory Coast": "科特迪瓦", "Ecuador": "厄瓜多尔",
    "Netherlands": "荷兰", "Japan": "日本", "Sweden": "瑞典",
    "Tunisia": "突尼斯", "Belgium": "比利时", "Egypt": "埃及",
    "Iran": "伊朗", "New Zealand": "新西兰", "Spain": "西班牙",
    "Cape Verde": "佛得角", "Saudi Arabia": "沙特阿拉伯", "Uruguay": "乌拉圭",
    "France": "法国", "Senegal": "塞内加尔", "Iraq": "伊拉克",
    "Norway": "挪威", "Argentina": "阿根廷", "Algeria": "阿尔及利亚",
    "Austria": "奥地利", "Jordan": "约旦", "Portugal": "葡萄牙",
    "DR Congo": "刚果(金)", "Uzbekistan": "乌兹别克斯坦", "Colombia": "哥伦比亚",
    "England": "英格兰", "Croatia": "克罗地亚", "Ghana": "加纳", "Panama": "巴拿马",
    "Italy": "意大利", "Chile": "智利", "Denmark": "丹麦", "Poland": "波兰",
    "Nigeria": "尼日利亚", "Peru": "秘鲁", "Russia": "俄罗斯", "Ukraine": "乌克兰",
    "Wales": "威尔士", "Serbia": "塞尔维亚", "Cameroon": "喀麦隆",
}


def cn(name: str) -> str:
    """球队名 → 中文名，找不到则保留原名。"""
    return TEAM_CN.get(name, name)


def normalize_team(raw: str) -> str:
    """把各种口径的队名归一化到【历史数据集】队名（Elo 评分据此查找）。别名优先。"""
    if not raw:
        return raw
    raw = str(raw).strip()
    if raw in TEAM_NAME_ALIASES:                       # 精确别名优先
        return TEAM_NAME_ALIASES[raw]
    low = raw.lower()
    for alias, canonical in TEAM_NAME_ALIASES.items():  # 大小写 / HTML 实体近似
        if alias.lower() == low:
            return canonical
    return raw


def _is_placeholder(name: str) -> bool:
    """openfootball 里未确定的淘汰赛对阵用 W80 / L101 之类的占位符。"""
    import re
    return bool(re.fullmatch(r"[WL]\d+", str(name or "").strip()))


def _to_beijing(date_s: str, time_s: str):
    """openfootball 的 time 形如 '13:00 UTC-6' → 换算成北京时间 datetime。"""
    import re
    m = re.match(r"(\d{1,2}):(\d{2})\s*UTC\s*([+-]?\d+)", str(time_s or ""))
    if not m:
        return None
    hh, mm, off = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        local = dt.datetime.fromisoformat(str(date_s)[:10]).replace(hour=hh, minute=mm)
    except Exception:
        return None
    return (local - dt.timedelta(hours=off)) + dt.timedelta(hours=8)  # UTC→北京


# openfootball 本地缓存（网络抖动时回退用）
_OF_CACHE = Path(__file__).resolve().parent / "data" / "openfootball_2026.json"


def _fetch_of_json():
    """拉取 openfootball 2026 数据，带重试；全部失败则回退到本地缓存。"""
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(OPENFOOTBALL_URL, timeout=(10, 30))
            r.raise_for_status()
            data = r.json()
            _OF_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _OF_CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except Exception as e:
            last_err = e
            print(f"  ⚠️ openfootball 拉取第 {attempt + 1}/3 次失败：{type(e).__name__}")
            time.sleep(2)
    if _OF_CACHE.exists():
        print(f"  ↩️  改用本地缓存（最近一次错误：{last_err}）")
        return json.loads(_OF_CACHE.read_text(encoding="utf-8"))
    raise last_err


def fetch_openfootball_upcoming(today, days_ahead: int = 5, limit: int = 8):
    """
    从 openfootball 拉取 2026 世界杯「未踢且双方均已确定」的 upcoming 比赛。
    以【北京时间】判断是否在未来 days_ahead 天内。返回 (list, status_str)。
    """
    data = _fetch_of_json()
    matches = data.get("matches", [])

    horizon = today + dt.timedelta(days=days_ahead)
    picked = []
    for m in matches:
        if (m.get("score") or {}).get("ft") is not None:        # 已踢
            continue
        t1, t2 = m.get("team1"), m.get("team2")
        if _is_placeholder(t1) or _is_placeholder(t2):
            continue
        bj = _to_beijing(m.get("date"), m.get("time"))
        if not bj:
            continue
        if bj.date() < today or bj.date() > horizon:             # 按北京时间过滤
            continue
        home, away = normalize_team(t1), normalize_team(t2)
        if not home or not away or home == away:
            continue
        picked.append({
            "home": home, "away": away,
            "stage": m.get("round") or "",
            "date": bj.strftime("%Y-%m-%d"),
            "bj": bj.strftime("%m-%d %H:%M"),                    # 北京时间开赛
        })

    picked.sort(key=lambda x: x["bj"])
    picked = picked[:limit]
    return picked, f"openfootball 实时赛程，共 {len(picked)} 场未踢（未来 {days_ahead} 天，按北京时间）"


# ──────────────────────────────────────────────
# 报告渲染
# ──────────────────────────────────────────────

def render_rankings(top) -> str:
    lines = ["| 排名 | 球队 | Elo 评分 |", "|---:|:--|--:"]
    for i, row in enumerate(top.itertuples(), 1):
        lines.append(f"| {i} | {cn(row.team)} ({row.team}) | {row.elo:.0f} |")
    return "\n".join(lines)


def render_prediction(pred) -> str:
    home, away = pred["home"], pred["away"]
    fav = pred["favorite"]
    lines = [
        f"#### 🏟️ {cn(home)} vs {cn(away)}",
        "",
        "| 结果 | 概率 |",
        "|:--|--:|",
        f"| 主胜 · {cn(home)} | **{pred['p_home']*100:.1f}%** |",
        f"| 平局 | {pred['p_draw']*100:.1f}% |",
        f"| 客胜 · {cn(away)} | **{pred['p_away']*100:.1f}%** |",
        "",
        f"- 预期进球 (xG)：{cn(home)} **{pred['xg_home']:.2f}** – {pred['xg_away']:.2f} **{cn(away)}**",
        f"- Elo 评分：{cn(home)} {pred['elo_home']:.0f}　/　{cn(away)} {pred['elo_away']:.0f}",
        f"- 🏅 最被看好：**{cn(fav)}**",
        "",
    ]
    return "\n".join(lines)


def simulate_tournament_fast(elo, model, feat_df, n_sims: int = 3000):
    """
    与 predictor.simulate_tournament 数学等价，但快得多。

    原版每次对阵都调用 predict_match（内部对 5 万行数据做全表扫描），
    蒙特卡洛里被调用上万次 → 极慢。由于 predict_match 是纯函数（给定两队
    结果不变），这里把全部 48×47 种对阵概率预算一次缓存，模拟时只查表。
    """
    print(f"\n🏆  运行 {n_sims:,} 次锦标赛模拟（快速版：预计算对阵概率）…")

    all_teams = sorted(set(normalize_team(t) for m in P.WC2026_MATCHES for t in (m[0], m[1])))

    # 预计算每一对（有序）对阵的 [主胜, 平, 客胜] 概率
    cache: dict[tuple[str, str], tuple[float, float, float]] = {}
    for h in all_teams:
        for a in all_teams:
            if h == a:
                continue
            p = P.predict_match(h, a, elo, model, feat_df, neutral=True, is_wc=True)
            cache[(h, a)] = (p["p_home"], p["p_draw"], p["p_away"])

    wins: dict[str, int] = defaultdict(int)
    for _ in range(n_sims):
        remaining = list(all_teams)
        while len(remaining) > 1:
            rng = np.random.default_rng()
            rng.shuffle(remaining)
            nxt = []
            for i in range(0, len(remaining) - 1, 2):
                h, a = remaining[i], remaining[i + 1]
                p_home, p_draw, _ = cache[(h, a)]
                roll = rng.random()
                if roll < p_home:
                    nxt.append(h)
                elif roll < p_home + p_draw:
                    nxt.append(h if rng.random() > 0.5 else a)  # 点球 50/50
                else:
                    nxt.append(a)
            if len(remaining) % 2 == 1:
                nxt.append(remaining[-1])  # 轮空
            remaining = nxt
        wins[remaining[0]] += 1

    results = pd.DataFrame([
        {"team": t, "championship_prob": round(wins.get(t, 0) / n_sims * 100, 2)}
        for t in sorted(all_teams, key=lambda x: -wins.get(x, 0))
    ])
    return results


def render_championship(sim, top_k=12) -> str:
    lines = ["| 排名 | 球队 | 夺冠概率 |", "|---:|:--|--:|"]
    for i, row in enumerate(sim.head(top_k).itertuples(), 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f"{i}"))
        lines.append(f"| {medal} | {cn(row.team)} ({row.team}) | **{row.championship_prob:.1f}%** |")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    today = dt.date.today()
    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    sim_runs = int(os.environ.get("SIM_RUNS", "3000"))
    top_n = int(os.environ.get("TOP_N", "20"))
    days_ahead = int(os.environ.get("DAYS_AHEAD", "5"))
    max_predict = int(os.environ.get("MAX_PREDICT", "8"))

    # 1. 跑通模型流水线
    print("▶ 加载历史数据 …", flush=True)
    df = P.load_results()
    print("▶ 计算 Elo 评分 …", flush=True)
    elo = P.EloRatingSystem().fit(df)
    feat_df = P.build_features(df, elo)
    print("▶ 训练 XGBoost 模型 …", flush=True)
    model = P.train_model(feat_df)

    # 2. 实力排名
    top = elo.top_n(top_n)

    # 3. 比赛预测：优先 openfootball 实时赛程（免费、无需 key）
    matches_to_predict = []
    live_status = "实时赛程不可用，使用内置重点对决"
    live_note = ""
    try:
        upcoming, info = fetch_openfootball_upcoming(today, days_ahead=days_ahead,
                                                     limit=max_predict)
        if upcoming:
            matches_to_predict = upcoming
            live_status = info
            live_note = "（数据来源：openfootball，免费、无需 key）"
        else:
            live_status = "实时赛程近期无未踢比赛，使用内置重点对决"
            live_note = "（openfootball 已拉取，但未来窗口内无对阵已定的比赛）"
    except Exception as e:
        live_status = f"⚠️ 实时赛程拉取失败：{type(e).__name__}: {e}。已回退到内置重点对决。"

    if not matches_to_predict:
        # 回退：一组看点十足的对决（覆盖各洲强队）
        matches_to_predict = [
            {"home": "Brazil", "away": "Germany", "stage": "焦点对决", "date": ""},
            {"home": "France", "away": "Argentina", "stage": "焦点对决", "date": ""},
            {"home": "England", "away": "Spain", "stage": "焦点对决", "date": ""},
            {"home": "Portugal", "away": "Netherlands", "stage": "焦点对决", "date": ""},
        ]

    predictions = []
    for m in matches_to_predict:
        try:
            pred = P.predict_match(m["home"], m["away"], elo, model, feat_df,
                                   neutral=True, is_wc=True)
            # xG 在双方实力悬殊时会算出虚高值，截到合理上限
            pred["xg_home"] = round(min(pred["xg_home"], 4.0), 2)
            pred["xg_away"] = round(min(pred["xg_away"], 4.0), 2)
            pred["stage"] = m.get("stage", "")
            pred["mdate"] = m.get("date", "")
            pred["bj"] = m.get("bj", "")
            predictions.append(pred)
        except Exception as e:
            print(f"  跳过 {m['home']} vs {m['away']}：{e}")

    # 4. 蒙特卡洛夺冠模拟（快速版：预计算对阵概率，避免万次全表扫描）
    print(f"▶ 蒙特卡洛模拟（{sim_runs:,} 次）…", flush=True)
    sim = simulate_tournament_fast(elo, model, feat_df, n_sims=sim_runs)

    # 5. 渲染 Markdown
    md = []
    md.append(f"# ⚽ 2026 世界杯 AI 每日预测报告\n")
    md.append(f"> 生成时间：**{now_str}** ｜ 模型：Elo + XGBoost（5 万+ 历史比赛训练）\n")
    md.append(f"> 数据说明：{live_status} {live_note}\n")
    md.append("\n---\n")

    md.append(f"## 📊 一、AI 实力排名（Elo Top {len(top)}）\n")
    md.append(render_rankings(top))
    md.append("\n---\n")

    md.append("## 🔮 二、比赛预测\n")
    if predictions:
        for pred in predictions:
            when = pred.get("bj") or pred.get("mdate", "")
            tag = f"　_（{pred.get('stage', '')} · 北京时间 {when}）_" if when else ""
            md.append(f"**{cn(pred['home'])} {pred['home']} — {pred['away']} {cn(pred['away'])}**{tag}\n")
            md.append(render_prediction(pred))
    else:
        md.append("_本期无可用比赛预测。_\n")
    md.append("\n---\n")

    md.append(f"## 🏆 三、夺冠概率模拟（蒙特卡洛 {sim_runs:,} 次）\n")
    md.append(render_championship(sim))
    md.append("\n---\n")

    md.append("## 📌 说明\n")
    md.append("- Elo 评分随每场比赛动态更新，世界杯比赛权重最高（k=60）。")
    md.append("- 胜率由 XGBoost 多分类模型给出（主胜 / 平 / 客胜）；xG 为基于状态与 Elo 的预期进球估计。")
    md.append("- 夺冠概率来自全赛程蒙特卡洛模拟，含点球 50/50 近似。")
    md.append("- 实时赛程来自 openfootball（免费、无需 key，赛后更新）；拉取失败时自动回退到内置重点对决。")
    md.append(f"\n_本报告由 GitHub Actions 每日自动生成。_\n")

    report = "\n".join(md)

    # 6. 写文件：覆盖最新版 + 归档历史
    reports_dir = Path(__file__).resolve().parent / "reports"
    archive_dir = reports_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    latest = reports_dir / "预测报告.md"
    latest.write_text(report, encoding="utf-8")
    (archive_dir / f"报告_{today.isoformat()}.md").write_text(report, encoding="utf-8")

    # 同时输出到 stdout（供 GitHub Actions 的 run summary 抓取）
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\n✅ 报告已写入：{latest}")
    return report


if __name__ == "__main__":
    main()

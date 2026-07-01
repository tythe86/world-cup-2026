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
API_BASE = "https://api.balldontlie.io/fifa/worldcup/v1"

# BALLDONTLIE 球队名（FIFA 官方名）→ predictor 使用的 martj42 数据集队名
TEAM_NAME_ALIASES = {
    "Korea Republic": "South Korea",
    "Korea DPR": "South Korea",
    "IR Iran": "Iran",
    "Islamic Republic of Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "United States": "USA",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "Congo (DR)": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
    "Curaçao": "Curacao",
    "Curacao": "Curacao",
    "Czech Republic": "Czechia",
}

# 球队中文名（覆盖全部 48 支参赛队 + 常见强队）
TEAM_CN = {
    "Mexico": "墨西哥", "South Africa": "南非", "South Korea": "韩国",
    "Czechia": "捷克", "Canada": "加拿大", "Switzerland": "瑞士",
    "Qatar": "卡塔尔", "Bosnia and Herzegovina": "波黑",
    "Brazil": "巴西", "Morocco": "摩洛哥", "Haiti": "海地",
    "Scotland": "苏格兰", "USA": "美国", "Paraguay": "巴拉圭",
    "Australia": "澳大利亚", "Turkey": "土耳其", "Germany": "德国",
    "Curacao": "库拉索", "Ivory Coast": "科特迪瓦", "Ecuador": "厄瓜多尔",
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


# predictor 数据集里出现过的全部队名（含世界杯参赛队）
WC_TEAM_NAMES = {t for m in P.WC2026_MATCHES for t in (m[0], m[1])}


def normalize_team(raw: str) -> str:
    """把 BALLDONTLIE 返回的队名归一化到 predictor 的队名空间。"""
    if not raw:
        return raw
    # 已是数据集里的标准名
    if raw in WC_TEAM_NAMES or raw in TEAM_CN:
        return raw
    # 精确别名
    if raw in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[raw]
    # 大小写 / 去重音近似
    key = raw.strip().lower()
    for alias, canonical in TEAM_NAME_ALIASES.items():
        if alias.lower() == key:
            return canonical
    return raw


def fetch_live_matches(api_key: str):
    """
    拉取 BALLDONTLIE 2026 赛季全部比赛。
    返回 (matches_list, status_str)。失败抛异常由调用方捕获。
    """
    headers = {"Authorization": api_key, "Accept": "application/json"}
    session = requests.Session()
    session.headers.update(headers)

    all_matches = []
    cursor = 0
    # 翻页：API 支持 per_page + cursor/next_cursor
    url = f"{API_BASE}/matches"
    for _ in range(20):  # 安全上限，避免死循环
        params = {"season": 2026, "per_page": 100}
        if cursor:
            params["cursor"] = cursor
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 401:
            raise PermissionError("API key 鉴权失败（401 Unauthorized）")
        r.raise_for_status()
        data = r.json()
        page = data.get("data") or data.get("matches") or []
        all_matches.extend(page)
        meta = data.get("meta") or {}
        cursor = meta.get("next_cursor")
        if not cursor or not page:
            break
    return all_matches, f"已拉取 {len(all_matches)} 场实时赛程"


def extract_team_name(side) -> str:
    """home_team/away_team 可能是 dict 或 str。"""
    if isinstance(side, dict):
        return side.get("name") or side.get("full_name") or ""
    return str(side or "")


def upcoming_from_live(live_matches, today, limit=8):
    """从实时赛程里挑出今天及之后未开始的比赛。"""
    UPCOMING = {"scheduled", "upcoming", "not_started", "pre"}
    picked = []
    for m in live_matches:
        status = str(m.get("status", "")).lower()
        if status and status not in UPCOMING:
            continue
        # 解析日期（去掉时区后比较）
        d = m.get("date") or m.get("start_date") or m.get("datetime")
        mdate = None
        if d:
            try:
                mdate = dt.datetime.fromisoformat(str(d).replace("Z", "+00:00"))
            except Exception:
                mdate = None
        if mdate and mdate.date() < today:
            continue
        home = normalize_team(extract_team_name(m.get("home_team")))
        away = normalize_team(extract_team_name(m.get("away_team")))
        if not home or not away or home == away:
            continue
        picked.append({
            "home": home, "away": away,
            "stage": m.get("stage") or m.get("round") or "",
            "date": str(d)[:10] if d else "",
        })
        if len(picked) >= limit:
            break
    return picked


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

    all_teams = sorted(set(t for m in P.WC2026_MATCHES for t in (m[0], m[1])))

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
    api_key = (os.environ.get("BALLDONTLIE_API_KEY") or "").strip()
    sim_runs = int(os.environ.get("SIM_RUNS", "3000"))
    top_n = int(os.environ.get("TOP_N", "20"))

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

    # 3. 比赛预测：优先实时赛程
    matches_to_predict = []
    live_status = "未配置 `BALLDONTLIE_API_KEY`，使用内置重点对决"
    live_note = ""
    if api_key:
        try:
            live_matches, info = fetch_live_matches(api_key)
            live_status = info
            upcoming = upcoming_from_live(live_matches, today, limit=8)
            if upcoming:
                matches_to_predict = upcoming
                live_note = "（数据来源：BALLDONTLIE 实时赛程）"
            else:
                live_note = "（已拉取实时赛程，但近期没有未开始的比赛，回退到内置重点对决）"
        except PermissionError as e:
            live_status = f"⚠️ 实时赛程不可用：{e}。已回退到内置重点对决。"
        except Exception as e:
            live_status = f"⚠️ 实时赛程拉取失败：{type(e).__name__}: {e}。已回退到内置重点对决。"

    if not matches_to_predict:
        # 回退：一组看点十足的对决（覆盖各洲强队）
        matches_to_predict = [
            {"home": "Brazil", "away": "Germany", "stage": "焦点对决", "date": "", "city": ""},
            {"home": "France", "away": "Argentina", "stage": "焦点对决", "date": "", "city": ""},
            {"home": "England", "away": "Spain", "stage": "焦点对决", "date": "", "city": ""},
            {"home": "Portugal", "away": "Netherlands", "stage": "焦点对决", "date": "", "city": ""},
        ]

    predictions = []
    for m in matches_to_predict:
        try:
            pred = P.predict_match(m["home"], m["away"], elo, model, feat_df,
                                   neutral=True, is_wc=True)
            pred["stage"] = m.get("stage", "")
            pred["mdate"] = m.get("date", "")
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
            tag = ""
            if pred.get("stage") or pred.get("mdate"):
                tag = f"　_（{pred.get('stage','')} {pred.get('mdate','')}）_".strip()
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
    md.append("- 实时赛程通过 `BALLDONTLIE_API_KEY` 获取；未配置或失效时自动回退，不影响报告生成。")
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

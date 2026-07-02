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


# 球队国旗 emoji（按【历史数据集】标准名 / 归一化后的队名索引）
# 英格兰/苏格兰/威尔士不是独立 ISO 国家，用 emoji 标签序列（现代微信/iOS/Android 均支持）
TEAM_FLAG = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Cameroon": "🇨🇲", "Canada": "🇨🇦", "Cape Verde": "🇨🇻", "Chile": "🇨🇱",
    "Colombia": "🇨🇴", "Croatia": "🇭🇷", "Curaçao": "🇨🇼",
    "Czech Republic": "🇨🇿", "Denmark": "🇩🇰", "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨", "Egypt": "🇪🇬", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "France": "🇫🇷", "Germany": "🇩🇪", "Ghana": "🇬🇭", "Haiti": "🇭🇹",
    "Iran": "🇮🇷", "Iraq": "🇮🇶", "Ivory Coast": "🇨🇮", "Italy": "🇮🇹",
    "Japan": "🇯🇵", "Jordan": "🇯🇴", "Mexico": "🇲🇽", "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱", "New Zealand": "🇳🇿", "Nigeria": "🇳🇬",
    "Norway": "🇳🇴", "Panama": "🇵🇦", "Paraguay": "🇵🇾", "Peru": "🇵🇪",
    "Poland": "🇵🇱", "Portugal": "🇵🇹", "Qatar": "🇶🇦", "Russia": "🇷🇺",
    "Saudi Arabia": "🇸🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Senegal": "🇸🇳",
    "Serbia": "🇷🇸", "South Africa": "🇿🇦", "South Korea": "🇰🇷",
    "Spain": "🇪🇸", "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Tunisia": "🇹🇳",
    "Turkey": "🇹🇷", "Ukraine": "🇺🇦", "United States": "🇺🇸",
    "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
}


def flag(name: str) -> str:
    """球队名 → 国旗 emoji，找不到返回空串。先归一化到标准名再查。"""
    if not name:
        return ""
    canonical = TEAM_NAME_ALIASES.get(name, name)
    return TEAM_FLAG.get(canonical) or TEAM_FLAG.get(name, "")


def cn_flag(name: str) -> str:
    """国旗 emoji + 中文名（用于纯文本 / Markdown 展示）。"""
    f = flag(name)
    return f"{f} {cn(name)}" if f else cn(name)


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
    beijing_now = dt.datetime.utcnow() + dt.timedelta(hours=8)   # 不依赖机器时区
    for m in matches:
        if (m.get("score") or {}).get("ft") is not None:        # 已踢完
            continue
        t1, t2 = m.get("team1"), m.get("team2")
        if _is_placeholder(t1) or _is_placeholder(t2):
            continue
        bj = _to_beijing(m.get("date"), m.get("time"))
        if not bj:
            continue
        if bj < beijing_now:                                     # 已开赛/进行中 → 不预测
            continue
        if bj.date() < today or bj.date() > horizon:            # 按北京时间日期窗口
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


def fetch_worldcup_alive_teams():
    """
    依据 openfootball 已完成的淘汰赛结果，返回【仍在争冠】的球队集合。
    规则：进入淘汰赛（Round of 32 起）的队为候选；任一已完成的淘汰赛里
    输掉的一方即被淘汰。胜负判定优先级：点球 p > 加时 et > 常规 ft。
    无法判定或异常时返回 None（调用方回退到全部球队）。
    """
    KNOCKOUT = ("round of 32", "round of 16", "quarter-final", "semi-final", "final")
    data = _fetch_of_json()
    participants, eliminated = set(), set()
    for m in data.get("matches", []):
        rnd = str(m.get("round") or "").lower()
        if not any(k in rnd for k in KNOCKOUT):
            continue
        t1, t2 = m.get("team1"), m.get("team2")
        if _is_placeholder(t1) or _is_placeholder(t2):
            continue
        h, a = normalize_team(t1), normalize_team(t2)
        participants.add(h)
        participants.add(a)
        s = m.get("score") or {}
        if s.get("ft") is None:
            continue                       # 未踢：两队仍为候选
        loser = None
        for key in ("p", "et", "ft"):      # 点球 > 加时 > 常规
            v = s.get(key)
            if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] != v[1]:
                loser = a if v[0] > v[1] else h
                break
        if loser:
            eliminated.add(loser)
    alive = participants - eliminated
    return alive if len(alive) >= 2 else None


# The Odds API（免费 500 次/月）：世界杯 h2h 赔率，反推市场隐含概率
ODDS_URL = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds/"


def fetch_odds(api_key: str):
    """拉取世界杯 h2h 赔率，返回 {frozenset({home,away}): odds_dict}。失败返回 ({},原因)。"""
    if not api_key:
        return {}, ""
    try:
        r = requests.get(ODDS_URL, params={
            "apiKey": api_key, "regions": "eu,uk,us",
            "markets": "h2h", "oddsFormat": "decimal",
        }, timeout=30)
        if r.status_code == 401:
            return {}, "⚠️ Odds API key 无效（401）"
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {}, f"⚠️ 赔率拉取失败：{type(e).__name__}"

    odds_map = {}
    for e in data:
        home = normalize_team(e.get("home_team"))
        away = normalize_team(e.get("away_team"))
        if not home or not away or home == away:
            continue
        # 各家博彩公司的 h2h 价格，按 outcome 取平均
        prices = defaultdict(list)  # 'Draw' 或 归一化队名 -> [价格]
        for bm in e.get("bookmakers", []):
            for m in bm.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    nm = o.get("name")
                    key = "Draw" if nm == "Draw" else normalize_team(nm)
                    if o.get("price"):
                        prices[key].append(float(o["price"]))
        if not prices:
            continue
        avg = {k: sum(v) / len(v) for k, v in prices.items()}
        ph, pd, pa = avg.get(home), avg.get("Draw"), avg.get(away)
        if not (ph and pd and pa):
            continue
        rh, rd, ra = 1 / ph, 1 / pd, 1 / pa       # 反推
        tot = rh + rd + ra                         # 去水（去 overround）
        odds_map[frozenset({home, away})] = {
            "ph": round(ph, 2), "pd": round(pd, 2), "pa": round(pa, 2),
            "imp_home": round(rh / tot * 100, 1),
            "imp_draw": round(rd / tot * 100, 1),
            "imp_away": round(ra / tot * 100, 1),
        }
    return odds_map, f"The Odds API 赔率，覆盖 {len(odds_map)} 场"


# PushPlus：把报告推送到微信（免费，token 来自 secret）
PUSHPLUS_URL = "http://www.pushplus.plus/send"


def push_to_wechat(token: str, title: str, content: str, template: str = "markdown",
                   summary: str = "") -> str:
    """通过 PushPlus 把报告推送到微信。template 可选 'markdown' / 'html'；
    summary 为消息摘要（通知预览/转发卡片用；HTML 模板务必提供，否则预览会抓到 HTML 头部代码）。"""
    if not token:
        return "未配置 PUSHPLUS_TOKEN，跳过微信推送"
    try:
        payload = {
            "token": token, "title": title, "content": content,
            "template": template, "channel": "wechat",
        }
        if summary:
            payload["summary"] = summary
        r = requests.post(PUSHPLUS_URL, json=payload, timeout=30)
        data = r.json()
        if data.get("code") == 200:
            return "已推送到微信（PushPlus）"
        return f"⚠️ 微信推送失败：{data.get('msg')}"
    except Exception as e:
        return f"⚠️ 微信推送异常：{type(e).__name__}: {e}"


# ──────────────────────────────────────────────
# 报告渲染
# ──────────────────────────────────────────────

def predicted_scoreline(xg_h: float, xg_a: float, max_goals: int = 6):
    """用 xG 做独立 Poisson，取概率最高的具体比分。返回 (score_str, prob)。"""
    from math import exp, factorial

    def pois(k: int, lam: float) -> float:
        return exp(-lam) * (lam ** k) / factorial(k)

    best, best_p = (0, 0), 0.0
    for i in range(max_goals + 1):
        pi = pois(i, xg_h)
        for j in range(max_goals + 1):
            p = pi * pois(j, xg_a)
            if p > best_p:
                best, best_p = (i, j), p
    return f"{best[0]}–{best[1]}", best_p


def render_summary(predictions) -> str:
    """顶部汇总表：预测比分 + 模型概率（有赔率时并列市场反推概率）。"""
    has_odds = any(p.get("odds") for p in predictions)
    if has_odds:
        lines = ["| 北京时间 | 比赛 | 预测比分 | 模型 主/平/客 | 市场赔率反推 主/平/客 |",
                 "|---|---|:--:|---:|---:|"]
    else:
        lines = ["| 北京时间 | 比赛 | 预测比分 | 最被看好 | 主胜/平/客胜 |",
                 "|---|---|:--:|---|---:|"]
    for pred in predictions:
        score, _ = predicted_scoreline(pred["xg_home"], pred["xg_away"])
        match = f"{cn_flag(pred['home'])} vs {cn_flag(pred['away'])}"
        when = pred.get("bj") or pred.get("mdate") or "—"
        model = f"{pred['p_home']*100:.0f}% / {pred['p_draw']*100:.0f}% / {pred['p_away']*100:.0f}%"
        if has_odds:
            o = pred.get("odds")
            market = f"{o['imp_home']:.0f}% / {o['imp_draw']:.0f}% / {o['imp_away']:.0f}%" if o else "—"
            lines.append(f"| {when} | {match} | **{score}** | {model} | {market} |")
        else:
            fav = pred["favorite"]
            fav_tag = f"{cn_flag(fav)} ({max(pred['p_home'],pred['p_draw'],pred['p_away'])*100:.0f}%)"
            lines.append(f"| {when} | {match} | **{score}** | {fav_tag} | {model} |")
    return "\n".join(lines)


def render_rankings(top) -> str:
    lines = ["| 排名 | 球队 | Elo 评分 |", "|---:|:--|--:"]
    for i, row in enumerate(top.itertuples(), 1):
        lines.append(f"| {i} | {cn_flag(row.team)} ({row.team}) | {row.elo:.0f} |")
    return "\n".join(lines)


def render_prediction(pred) -> str:
    home, away = pred["home"], pred["away"]
    fav = pred["favorite"]
    lines = [
        f"#### 🏟️ {cn_flag(home)} vs {cn_flag(away)}",
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
    ]
    o = pred.get("odds")
    if o:
        lines.append(f"- 💰 市场赔率反推：{cn(home)} {o['imp_home']:.0f}% / 平 {o['imp_draw']:.0f}% / {cn(away)} {o['imp_away']:.0f}%（赔率 {o['ph']}/{o['pd']}/{o['pa']}）")
    lines.append("")
    return "\n".join(lines)


def simulate_tournament_fast(elo, model, feat_df, n_sims: int = 3000, alive_teams=None):
    """
    与 predictor.simulate_tournament 数学等价，但快得多。

    原版每次对阵都调用 predict_match（内部对 5 万行数据做全表扫描），
    蒙特卡洛里被调用上万次 → 极慢。由于 predict_match 是纯函数（给定两队
    结果不变），这里把全部 48×47 种对阵概率预算一次缓存，模拟时只查表。
    """
    print(f"\n🏆  运行 {n_sims:,} 次锦标赛模拟（快速版：预计算对阵概率）…")

    all_teams = sorted(set(normalize_team(t) for m in P.WC2026_MATCHES for t in (m[0], m[1])))
    if alive_teams:                        # 只模拟仍在争冠的球队，剔除已淘汰者
        before = len(all_teams)
        all_teams = sorted(t for t in all_teams if t in alive_teams)
        if before > len(all_teams):
            print(f"  ℹ️ 仅模拟 {len(all_teams)} 支仍在争冠的球队（已排除 {before - len(all_teams)} 支被淘汰队）")

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
        lines.append(f"| {medal} | {cn_flag(row.team)} ({row.team}) | **{row.championship_prob:.1f}%** |")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 微信 HTML 报告（PushPlus template=html，样式可控、表格不挤）
# ──────────────────────────────────────────────

_HTML_CSS = """
*{box-sizing:border-box;}
body{margin:0;padding:0;background:#eef1f4;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;color:#2b2f33;line-height:1.6;}
.wrap{padding:10px;}
.hdr{background:#0a7d3c;background:linear-gradient(135deg,#0a7d3c 0%,#13ae5d 100%);color:#fff;border-radius:14px;padding:18px 16px;}
.hdr h1{margin:0 0 8px;font-size:19px;line-height:1.35;}
.hdr p{margin:4px 0;font-size:13px;opacity:.94;}
.card{background:#fff;border-radius:14px;padding:14px 12px 10px;margin-top:11px;}
.card h2{margin:0 0 10px;font-size:16px;color:#0a7d3c;border-bottom:2px solid #eaf5ee;padding-bottom:7px;}
table.grid{width:100%;border-collapse:collapse;font-size:14px;}
table.grid th{background:#f0f7f1;color:#3a7d4f;font-weight:600;padding:8px 5px;font-size:13px;text-align:center;}
table.grid td{padding:8px 5px;border-bottom:1px solid #f0f2f4;text-align:center;vertical-align:middle;}
table.grid td.l,table.grid th.l{text-align:left;}
table.grid tr:last-child td{border-bottom:none;}
.t{white-space:nowrap;color:#666;font-size:12.5px;}
.score{font-weight:700;color:#0a7d3c;font-size:15px;}
.vs{color:#b8bec5;font-size:12px;margin:0 3px;font-weight:400;}
.muted{color:#8a9099;font-size:12px;}
.en{color:#aab0b6;font-size:11px;}
.rk{color:#9aa0a6;font-weight:700;width:32px;}
.elo{text-align:right;font-variant-numeric:tabular-nums;}
.elo b{color:#2b2f33;}
.match{background:#fafbfc;border:1px solid #eef1f4;border-radius:12px;padding:11px 12px;margin:9px 0;}
.match-head{font-size:16px;font-weight:700;}
.match-sub{margin:2px 0 8px;}
.scoreline{margin:4px 0 10px;font-size:13px;}
.bigscore{font-size:20px;color:#0a7d3c;margin-left:6px;vertical-align:-1px;}
.bars{margin:6px 0 10px;}
.bar-title{font-size:11px;color:#8a9099;font-weight:600;margin:2px 0;}
.bar-row{display:block;margin:6px 0;font-size:13px;}
.bar-label{display:inline-block;width:14%;vertical-align:middle;color:#555;}
.bar-track{display:inline-block;vertical-align:middle;height:14px;background:#eef1f4;border-radius:7px;overflow:hidden;}
.bar-fill{display:block;height:100%;border-radius:7px;}
.bar-val{display:inline-block;width:16%;text-align:right;vertical-align:middle;font-weight:600;font-variant-numeric:tabular-nums;color:#2b2f33;}
.kv{font-size:13px;color:#444;margin:3px 0;}
ul.notes{margin:6px 0 0;padding-left:18px;font-size:12.5px;color:#666;}
ul.notes li{margin:4px 0;}
.footer{text-align:center;color:#aab0b6;font-size:11px;margin:14px 0 4px;}
"""


def build_digest(predictions, sim, max_games: int = 5) -> str:
    """一句话摘要：预测比分 + 夺冠概率前三。用于微信转发卡片 / 通知摘要（避免抓到 HTML 头）。"""
    bits = []
    if predictions:
        games = []
        for p in predictions[:max_games]:
            s, _ = predicted_scoreline(p["xg_home"], p["xg_away"])
            games.append(f"{cn(p['home'])}{s}{cn(p['away'])}")
        bits.append("预测 " + "、".join(games))
    if len(sim):
        names = [f"{cn(r.team)} {r.championship_prob:.0f}%" for r in sim.head(3).itertuples()]
        bits.append("夺冠概率 " + "、".join(names))
    return " ｜ ".join(bits)


def _h(s) -> str:
    """HTML 转义（球队名 / 赛程文本安全起见）。"""
    import html as _html
    return _html.escape(str(s), quote=False)


def _team_html(name: str) -> str:
    """国旗 + 中文名（HTML 展示用）。"""
    f = flag(name)
    return f"{f} {cn(name)}" if f else cn(name)


def _prob_bar(label: str, pct: float, color: str) -> str:
    pct = max(0.0, min(100.0, pct))
    return (
        '<div class="bar-row">'
        f'<span class="bar-label">{_h(label)}</span>'
        f'<span class="bar-track" style="width:68%"><span class="bar-fill" style="width:{pct:.1f}%;background:{color}"></span></span>'
        f'<span class="bar-val">{pct:.1f}%</span>'
        '</div>'
    )


def render_html_summary(predictions) -> str:
    """汇总表：只保留 4 列（时间 / 对阵 / 比分 / 主/平/客），手机端不挤。"""
    rows = []
    for pred in predictions:
        score, _ = predicted_scoreline(pred["xg_home"], pred["xg_away"])
        match = (f'{_team_html(pred["home"])} <span class="vs">VS</span> '
                 f'{_team_html(pred["away"])}')
        when = pred.get("bj") or pred.get("mdate") or "—"
        model = (f'{pred["p_home"]*100:.0f} / {pred["p_draw"]*100:.0f} / '
                 f'{pred["p_away"]*100:.0f}')
        o = pred.get("odds")
        if o:   # 有市场赔率时，在模型概率下用小字附上市场概率（不加列，避免拥挤）
            cell = (f'{model}<br><span class="muted" style="font-size:11px">'
                    f'市 {o["imp_home"]:.0f}/{o["imp_draw"]:.0f}/{o["imp_away"]:.0f}</span>')
        else:
            cell = model
        rows.append(
            f'<tr><td class="t">{_h(when)}</td><td class="l">{match}</td>'
            f'<td class="score">{_h(score)}</td><td>{cell}</td></tr>'
        )
    return ('<table class="grid"><thead><tr>'
            '<th>北京时间</th><th class="l">对阵</th><th>预测比分</th>'
            '<th>主/平/客 %<br><span class="muted" style="font-weight:400;font-size:10px">模型/市场</span></th>'
            '</tr></thead><tbody>' + "\n".join(rows) + '</tbody></table>')


def render_html_rankings(top) -> str:
    rows = []
    for i, row in enumerate(top.itertuples(), 1):
        rows.append(
            f'<tr><td class="rk">{i}</td>'
            f'<td class="l">{_team_html(row.team)} <span class="en">{_h(row.team)}</span></td>'
            f'<td class="elo"><b>{row.elo:.0f}</b></td></tr>'
        )
    return ('<table class="grid"><thead><tr>'
            '<th>排名</th><th class="l">球队</th><th>Elo</th>'
            '</tr></thead><tbody>' + "\n".join(rows) + '</tbody></table>')


def render_html_match(pred) -> str:
    home, away, fav = pred["home"], pred["away"], pred["favorite"]
    score, _ = predicted_scoreline(pred["xg_home"], pred["xg_away"])
    when = pred.get("bj") or pred.get("mdate") or ""
    stage = pred.get("stage", "")
    sub = " · ".join(x for x in [stage, (f"北京时间 {when}" if when else "")] if x)
    bars = ('<div class="bar-title">模型预测</div>'
            + _prob_bar("主胜", pred["p_home"] * 100, "#0fae57")
            + _prob_bar("平局", pred["p_draw"] * 100, "#d97706")
            + _prob_bar("客胜", pred["p_away"] * 100, "#2563eb"))
    o = pred.get("odds")
    odds_line = ""
    if o:   # 市场赔率：灰色条并列，与模型对照；赔率数值小字附后
        bars += ('<div class="bar-title" style="margin-top:8px">市场赔率反推</div>'
                 + _prob_bar("主胜", o["imp_home"], "#9aa0a6")
                 + _prob_bar("平局", o["imp_draw"], "#c8ccd1")
                 + _prob_bar("客胜", o["imp_away"], "#9aa0a6"))
        odds_line = (f'<div class="muted" style="font-size:11px;margin:-2px 0 6px">'
                     f'赔率 主 {o["ph"]} / 平 {o["pd"]} / 客 {o["pa"]}</div>')
    return (
        '<div class="match">'
        f'<div class="match-head">{_team_html(home)} <span class="vs">VS</span> {_team_html(away)}</div>'
        f'<div class="match-sub muted">{_h(sub)}</div>'
        f'<div class="scoreline"><span class="muted">预测比分</span><b class="bigscore">{_h(score)}</b></div>'
        f'<div class="bars">{bars}</div>'
        f'<div class="kv">📈 预期进球 xG：{_team_html(home)} <b>{pred["xg_home"]:.2f}</b> – <b>{pred["xg_away"]:.2f}</b> {_team_html(away)}</div>'
        f'<div class="kv">⚖️ Elo 评分：{_team_html(home)} {pred["elo_home"]:.0f} / {_team_html(away)} {pred["elo_away"]:.0f}</div>'
        f'<div class="kv">🏅 最被看好：<b>{_team_html(fav)}</b></div>'
        f'{odds_line}'
        '</div>'
    )


def render_html_championship(sim, top_k: int = 12) -> str:
    maxp = sim["championship_prob"].max() if len(sim) else 1.0
    rows = []
    for i, row in enumerate(sim.head(top_k).itertuples(), 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
        w = row.championship_prob / maxp * 100 if maxp else 0
        rows.append(
            f'<tr><td class="rk">{medal}</td>'
            f'<td class="l">{_team_html(row.team)}</td>'
            f'<td><span class="bar-track" style="width:58%"><span class="bar-fill" style="width:{w:.1f}%;background:#f0a500"></span></span> '
            f'<b style="margin-left:6px">{row.championship_prob:.1f}%</b></td></tr>'
        )
    return ('<table class="grid"><thead><tr>'
            '<th>排名</th><th class="l">球队</th><th>夺冠概率</th>'
            '</tr></thead><tbody>' + "\n".join(rows) + '</tbody></table>')


def render_html_report(now_str, live_status, live_note, odds_status,
                       predictions, top, sim, sim_runs, alive_note="", digest="") -> str:
    """组装完整的微信 HTML 报告。"""
    has_pred = bool(predictions)
    P = []  # noqa: E741  一段一段拼，可读性优先
    title_text = "⚽ 2026 世界杯 AI 每日预测报告"
    P.append('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">'
             f'<title>{_h(title_text)}</title>'
             f'<meta name="description" content="{_h(digest)}">'
             f'<meta property="og:title" content="{_h(title_text)}">'
             f'<meta property="og:description" content="{_h(digest)}">'
             f'<style>{_HTML_CSS}</style></head><body><div class="wrap">')

    # 顶部标题卡
    hdr = ['<div class="hdr"><h1>⚽ 2026 世界杯 AI 每日预测报告</h1>'
           f'<p>🕒 生成时间（北京）{_h(now_str)} ｜ 模型 Elo + XGBoost（5 万+ 历史比赛）</p>']
    if live_status:
        note = f" {_h(live_note)}" if live_note else ""
        hdr.append(f'<p>📋 {_h(live_status)}{note}</p>')
    if odds_status:
        hdr.append(f'<p>💰 {_h(odds_status)}</p>')
    hdr.append('</div>')
    P.append("".join(hdr))

    # 一、汇总
    if has_pred:
        P.append('<div class="card"><h2>⚽ 一、比赛预测汇总</h2>'
                 + render_html_summary(predictions)
                 + '<p class="muted" style="margin-top:8px">预测比分由 xG 经 Poisson 分布取概率最高者，仅供参考。</p></div>')

    # 二、实力排名
    i = "二" if has_pred else "一"
    P.append(f'<div class="card"><h2>📊 {i}、AI 实力排名（Elo Top {len(top)}）</h2>'
             + render_html_rankings(top) + '</div>')

    # 三、比赛预测（详细）
    i = "三" if has_pred else "二"
    body = "".join(render_html_match(p) for p in predictions) if has_pred \
        else '<p class="muted">本期无可用比赛预测。</p>'
    P.append(f'<div class="card"><h2>🔮 {i}、比赛预测（详细）</h2>{body}</div>')

    # 四、夺冠概率
    i = "四" if has_pred else "三"
    champ = f'<div class="card"><h2>🏆 {i}、夺冠概率模拟（蒙特卡洛 {sim_runs:,} 次）</h2>'
    if alive_note:
        champ += f'<p class="muted" style="margin:-2px 0 8px">{_h(alive_note)}</p>'
    champ += render_html_championship(sim) + '</div>'
    P.append(champ)

    # 说明
    P.append('<div class="card"><h2>📌 说明</h2><ul class="notes">'
             '<li>Elo 评分随每场比赛动态更新，世界杯比赛权重最高（k=60）。</li>'
             '<li>胜率由 XGBoost 多分类模型给出（主胜 / 平 / 客胜）；xG 为基于状态与 Elo 的预期进球估计。</li>'
             '<li>夺冠概率来自蒙特卡洛模拟，仅含仍在争冠的球队（已在淘汰赛出局者已剔除），含点球 50/50 近似。</li>'
             '<li>实时赛程与赛果来自 openfootball；拉取失败时回退到内置重点对决。</li>'
             '<li>市场赔率来自 The Odds API，取多家均值并去水反推为隐含概率。</li>'
             '</ul></div>')

    P.append('<div class="footer">— 本报告由 GitHub Actions 每日自动生成 —</div>')
    P.append('</div></body></html>')
    return "\n".join(P)


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    _bj_now = dt.datetime.utcnow() + dt.timedelta(hours=8)   # CI 运行在 UTC，统一按北京时间
    today = _bj_now.date()
    now_str = _bj_now.strftime("%Y-%m-%d %H:%M")
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
            live_note = ""   # live_status 已说明来自 openfootball，不重复
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

    # 3.5 附加市场赔率（The Odds API，可选）→ 每场挂上 odds
    odds_api_key = (os.environ.get("ODDS_API_KEY") or "").strip()
    odds_map, odds_status = fetch_odds(odds_api_key)
    if odds_map:
        for pred in predictions:
            pred["odds"] = odds_map.get(frozenset({pred["home"], pred["away"]}))

    # 4. 蒙特卡洛夺冠模拟（快速版：预计算对阵概率，避免万次全表扫描）
    #    只模拟【仍在争冠】的球队 —— 已在淘汰赛出局的队不再参与
    alive_teams, alive_note = None, ""
    try:
        alive_teams = fetch_worldcup_alive_teams()
        if alive_teams:
            alive_note = f"仅含 {len(alive_teams)} 支仍在争冠的球队（已剔除出局者）"
    except Exception as e:
        print(f"  ⚠️ 无法确定淘汰状态，按全部球队模拟：{e}")
    print(f"▶ 蒙特卡洛模拟（{sim_runs:,} 次）…", flush=True)
    sim = simulate_tournament_fast(elo, model, feat_df, n_sims=sim_runs, alive_teams=alive_teams)

    # 5. 渲染 Markdown
    md = []
    md.append(f"# ⚽ 2026 世界杯 AI 每日预测报告\n")
    md.append(f"> 生成时间：**{now_str}**（北京时间）｜ 模型：Elo + XGBoost（5 万+ 历史比赛训练）\n")
    md.append(f"> 数据说明：{live_status} {live_note}\n")
    if odds_status:
        md.append(f"> 赔率：{odds_status}\n")
    md.append("\n---\n")

    # 一、汇总：未踢比赛的预测比分（最显眼，放最前）
    if predictions:
        md.append("## ⚽ 一、比赛预测汇总（预测比分）\n")
        md.append(render_summary(predictions))
        md.append("\n_预测比分由 xG 经 Poisson 分布取概率最高者；仅供参考。_\n")
        md.append("\n---\n")

    md.append(f"## 📊 {'二' if predictions else '一'}、AI 实力排名（Elo Top {len(top)}）\n")
    md.append(render_rankings(top))
    md.append("\n---\n")

    md.append(f"## 🔮 {'三' if predictions else '二'}、比赛预测（详细）\n")
    if predictions:
        for pred in predictions:
            when = pred.get("bj") or pred.get("mdate", "")
            tag = f"　_（{pred.get('stage', '')} · 北京时间 {when}）_" if when else ""
            md.append(f"**{cn_flag(pred['home'])} ({pred['home']}) — {cn_flag(pred['away'])} ({pred['away']})**{tag}\n")
            md.append(render_prediction(pred))
    else:
        md.append("_本期无可用比赛预测。_\n")
    md.append("\n---\n")

    md.append(f"## 🏆 {'四' if predictions else '三'}、夺冠概率模拟（蒙特卡洛 {sim_runs:,} 次）\n")
    if alive_note:
        md.append(f"> _{alive_note}_\n")
    md.append(render_championship(sim))
    md.append("\n---\n")

    md.append("## 📌 说明\n")
    md.append("- Elo 评分随每场比赛动态更新，世界杯比赛权重最高（k=60）。")
    md.append("- 胜率由 XGBoost 多分类模型给出（主胜 / 平 / 客胜）；xG 为基于状态与 Elo 的预期进球估计。")
    md.append("- 夺冠概率来自蒙特卡洛模拟，仅含仍在争冠的球队（已在淘汰赛出局者已剔除），含点球 50/50 近似。")
    md.append("- 实时赛程与赛果来自 openfootball；拉取失败时自动回退到内置重点对决。")
    md.append("- 市场赔率来自 [The Odds API](https://the-odds-api.com)，取多家均值并去水反推为隐含概率，可与模型概率对照。")
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

    # 7. 推送到微信（PushPlus，用 HTML 模板：表格带样式、胜率带进度条，手机端更美观）
    pushplus_token = (os.environ.get("PUSHPLUS_TOKEN") or "").strip()
    n_matches = len(predictions)
    title = f"⚽ 2026世界杯AI预测 · {today.isoformat()}（{n_matches}场）"
    digest = build_digest(predictions, sim)
    html_report = render_html_report(
        now_str=now_str, live_status=live_status, live_note=live_note,
        odds_status=odds_status, predictions=predictions,
        top=top, sim=sim, sim_runs=sim_runs, alive_note=alive_note, digest=digest,
    )
    # 本地存一份 HTML 预览，方便在浏览器里直接看推送效果
    (reports_dir / "preview_wechat.html").write_text(html_report, encoding="utf-8")
    push_msg = push_to_wechat(pushplus_token, title, html_report, template="html", summary=digest)
    print(f"📲 {push_msg}")
    return report


if __name__ == "__main__":
    main()

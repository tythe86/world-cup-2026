#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026 世界杯 · 赔率版报告（市场视角，odds-led）
================================================
不以模型为主，直接用 The Odds API 的市场赔率：
  1. 决赛 h2h 赔率：按各家博彩公司分列，去水反推隐含概率，给共识
  2. 冠军 outright 赔率：直接反推市场隐含夺冠概率（替代蒙特卡洛）
  3. 各家博彩公司对比 / 共识 / 离散度

仅依赖 requests + 标准库（不拉模型栈，方便单独跑）。
运行：python report_odds.py
环境变量：ODDS_API_KEY（必需）
"""
from __future__ import annotations

import os
import json
import datetime as dt
from collections import defaultdict
from pathlib import Path

import requests

# ──────────────────────────────────────────────
# 队名归一化 / 中文名 / 国旗
# （与 report_zh.py 保持一致；此处独立复制，避免 import 拉起整个模型栈）
# ──────────────────────────────────────────────
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


def cn(name: str) -> str:
    return TEAM_CN.get(name, name)


def normalize_team(raw) -> str:
    if not raw:
        return raw
    raw = str(raw).strip()
    if raw in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[raw]
    low = raw.lower()
    for alias, canonical in TEAM_NAME_ALIASES.items():
        if alias.lower() == low:
            return canonical
    return raw


def flag(name: str) -> str:
    if not name:
        return ""
    canonical = TEAM_NAME_ALIASES.get(name, name)
    return TEAM_FLAG.get(canonical) or TEAM_FLAG.get(name, "")


def cn_flag(name: str) -> str:
    f = flag(name)
    return f"{f} {cn(name)}" if f else cn(name)


# ──────────────────────────────────────────────
# The Odds API
# ──────────────────────────────────────────────
BASE = "https://api.the-odds-api.com/v4"
MAIN_SPORT = "soccer_fifa_world_cup"
REGIONS = "eu,uk,us"


def _quota(r) -> str:
    rem = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    if rem is None and used is None:
        return ""
    return f"API 额度：本月已用 {used or '?'} / 剩余 {rem or '?'}（免费 500/月）"


def _bj_from_iso(iso: str) -> str:
    """ISO UTC 时间 -> 北京时间 'MM-DD HH:MM'。失败返回原串。"""
    if not iso:
        return ""
    try:
        t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (t + dt.timedelta(hours=8)).strftime("%m-%d %H:%M")
    except Exception:
        return iso


def list_wc_sports(key: str):
    """列出与世界杯相关的 sport key（含 has_outrights 标记），用于探测 outright 入口。"""
    try:
        r = requests.get(f"{BASE}/sports/", params={"apiKey": key}, timeout=30)
    except Exception as e:
        return [], None, f"sports 列表请求异常：{type(e).__name__}: {e}"
    if r.status_code != 200:
        return [], r, f"sports 列表 HTTP {r.status_code}：{r.text[:200]}"
    data = r.json()
    hits = []
    for s in data:
        blob = (str(s.get("key", "")) + " " + str(s.get("title", ""))).lower()
        if "fifa" in blob or "world_cup" in blob or "worldcup" in blob:
            hits.append(s)
    return hits, r, f"sports 列表 OK，世界杯相关 {len(hits)} 个：{[s.get('key') for s in hits]}"


def fetch_h2h(key: str):
    """拉 h2h 赔率，返回 (matches, resp, status)。matches: 每场含各家 bookmaker 明细。"""
    try:
        r = requests.get(f"{BASE}/sports/{MAIN_SPORT}/odds/", params={
            "apiKey": key, "regions": REGIONS, "markets": "h2h", "oddsFormat": "decimal",
        }, timeout=30)
    except Exception as e:
        return [], None, f"h2h 请求异常：{type(e).__name__}: {e}"
    if r.status_code == 401:
        return [], r, "⚠️ Odds API key 无效（401）"
    if r.status_code != 200:
        return [], r, f"h2h HTTP {r.status_code}：{r.text[:200]}"
    data = r.json()
    matches = []
    for ev in data:
        home = normalize_team(ev.get("home_team", ""))
        away = normalize_team(ev.get("away_team", ""))
        if not home or not away or home == away:
            continue
        books = []
        for bm in ev.get("bookmakers", []):
            ph = pd = pa = None
            for m in bm.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m.get("outcomes", []):
                    nm = o.get("name")
                    pr = o.get("price")
                    if not pr:
                        continue
                    if nm == "Draw":
                        pd = float(pr)
                    elif normalize_team(nm) == home:
                        ph = float(pr)
                    elif normalize_team(nm) == away:
                        pa = float(pr)
            if ph and pd and pa:
                ih, idg, ia = 1 / ph, 1 / pd, 1 / pa
                tot = ih + idg + ia
                books.append({
                    "key": bm.get("key", "?"),
                    "title": bm.get("title", bm.get("key", "?")),
                    "ph": ph, "pd": pd, "pa": pa,
                    "ih": round(ih / tot * 100, 1),
                    "id": round(idg / tot * 100, 1),
                    "ia": round(ia / tot * 100, 1),
                    "last_update": bm.get("last_update", ""),
                })
        if books:
            books.sort(key=lambda b: b["title"].lower())
            # 同一家博彩公司在多区可能用不同 book_key 但同名返回，按 title 去重；
            # 地区差异（如 Unibet (FR)/(UK)）title 不同，自然保留
            seen = set()
            books = [b for b in books if not (b["title"] in seen or seen.add(b["title"]))]
            matches.append({
                "home": home, "away": away,
                "commence": ev.get("commence_time", ""),
                "books": books,
            })
    return matches, r, f"h2h OK，覆盖 {len(matches)} 场"


def fetch_outrights(key: str, sport_hits):
    """
    防御式拉 outright 冠军赔率。尝试主 sport key + 其它世界杯相关 key，
    分别用 markets=outrights 与默认（不指定）两种方式探，收集所有
    「无 Draw 且 >=2 个球队结果」的市场为 outright 池。
    返回 (pools, raw_log)。pools: 每个 {sport_key, market_key, book, book_key, teams:{team:price}}。
    """
    raw_log = []
    pools = []
    seen = set()                          # (market_key, book_key) 去重，避免多区/多入口重复
    candidates = [MAIN_SPORT] + [s.get("key") for s in sport_hits if s.get("key") != MAIN_SPORT]
    for sk in dict.fromkeys(candidates):              # 去重保序
        for mp in ("outrights", None):
            params = {"apiKey": key, "regions": REGIONS, "oddsFormat": "decimal"}
            if mp:
                params["markets"] = mp
            label = f"[sport={sk} markets={mp}]"
            try:
                r = requests.get(f"{BASE}/sports/{sk}/odds/", params=params, timeout=30)
            except Exception as e:
                raw_log.append(f"{label} EXC {type(e).__name__}: {e}")
                continue
            if r.status_code != 200:
                raw_log.append(f"{label} HTTP {r.status_code}: {r.text[:160]}")
                continue
            data = r.json()
            raw_log.append(f"{label} {len(data)} events")
            for ev in data:
                for bm in ev.get("bookmakers", []):
                    for m in ev_markets_all(bm):
                        outs = m.get("outcomes", [])
                        names = [o.get("name") for o in outs]
                        if "Draw" in names:           # h2h，跳过
                            continue
                        teams = {}
                        for o in outs:
                            nm = o.get("name")
                            pr = o.get("price")
                            if not nm or not pr:
                                continue
                            t = normalize_team(nm)
                            if not t:
                                continue
                            teams[t] = float(pr)
                        if len(teams) >= 2:
                            dedup = (m.get("key", "?"), bm.get("title", bm.get("key", "?")))
                            if dedup in seen:
                                continue
                            seen.add(dedup)
                            pools.append({
                                "sport_key": sk,
                                "market_key": m.get("key", "?"),
                                "book": bm.get("title", bm.get("key", "?")),
                                "book_key": bm.get("key", "?"),
                                "teams": teams,
                            })
    return pools, raw_log


def ev_markets_all(bm):
    """某些响应里 market 嵌在 bm['markets']；兼容。"""
    return bm.get("markets", []) or []


# ──────────────────────────────────────────────
# 渲染
# ──────────────────────────────────────────────
def render_h2h_match(mt) -> list[str]:
    home, away = mt["home"], mt["away"]
    books = mt["books"]
    n = len(books)
    avg = lambda k: sum(b[k] for b in books) / n
    lines = [
        f"### {cn_flag(home)} vs {cn_flag(away)}",
        "",
    ]
    ct = mt.get("commence", "")
    if ct:
        lines.append(f"_开赛（北京）：{_bj_from_iso(ct)}　·　采样 {n} 家博彩公司_")
    else:
        lines.append(f"_采样 {n} 家博彩公司_")
    lines.append("")
    lines.append("| 博彩公司 | 主胜赔率 | 平局 | 客胜赔率 | 主胜%(去水) | 平%(去水) | 客胜%(去水) |")
    lines.append("|:--|--:|--:|--:|--:|--:|--:|")
    for b in books:
        lines.append(f"| {b['title']} | {b['ph']:.2f} | {b['pd']:.2f} | {b['pa']:.2f} | {b['ih']:.1f} | {b['id']:.1f} | {b['ia']:.1f} |")
    lines.append(f"| **共识（均值）** | **{avg('ph'):.2f}** | **{avg('pd'):.2f}** | **{avg('pa'):.2f}** "
                 f"| **{avg('ih'):.1f}** | **{avg('id'):.1f}** | **{avg('ia'):.1f}** |")
    lines.append("")

    # 离散度 / 点评
    phs = [b["ph"] for b in books]
    pas = [b["pa"] for b in books]
    spread_h = max(phs) - min(phs)
    spread_a = max(pas) - min(pas)
    fav_home = avg("ih") >= avg("ia")
    fav = home if fav_home else away
    fav_pct = avg("ih") if fav_home else avg("ia")
    lines.append(f"- 🏅 市场最看好：**{cn_flag(fav)}**（去水后 {fav_pct:.1f}%）")
    lines.append(f"- 主胜赔率区间 {min(phs):.2f}–{max(phs):.2f}（极差 {spread_h:.2f}），"
                 f"客胜 {min(pas):.2f}–{max(pas):.2f}（极差 {spread_a:.2f}）—— "
                 f"{'各家分歧较大' if max(spread_h, spread_a) >= 0.5 else '各家共识较强'}。")
    lines.append("")
    return lines


def render_outright(pools) -> list[str]:
    # 按 market_key 分组；选「覆盖公司最多的」市场作为夺冠概率主表
    by_mkt = defaultdict(list)
    for p in pools:
        by_mkt[p["market_key"]].append(p)
    chosen_key = max(by_mkt.keys(), key=lambda k: len(by_mkt[k]))
    chosen = by_mkt[chosen_key]

    # 每家去水后球队概率 -> 跨家均值
    team_probs = defaultdict(list)        # team -> [pct,...] per book
    team_odds = defaultdict(list)         # team -> [price,...] per book
    for p in chosen:
        inv = {t: 1 / pr for t, pr in p["teams"].items()}
        tot = sum(inv.values())
        for t, iv in inv.items():
            team_probs[t].append(iv / tot * 100)
            team_odds[t].append(p["teams"][t])

    consensus = sorted(
        ((t, sum(v) / len(v), len(v)) for t, v in team_probs.items()),
        key=lambda x: -x[1],
    )
    n_books = len(chosen)
    lines = [
        f"_市场类型 `{chosen_key}`，采样 {n_books} 家博彩公司，去水后跨家均值。_",
        "",
        "| 排名 | 球队 | 共识夺冠概率 | 赔率区间 | 样本公司 |",
        "|---:|:--|--:|:--|--:|",
    ]
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}
    for i, (t, p, nb) in enumerate(consensus, 1):
        od = team_odds[t]
        mk = medal.get(i, str(i))
        lines.append(f"| {mk} | {cn_flag(t)} ({t}) | **{p:.1f}%** | {min(od):.1f}–{max(od):.1f} | {nb} |")
    lines.append("")

    # 各家明细：展示每家公司开出的前二
    lines.append("**各家博彩公司开出的夺冠赔率（前二热门）：**\n")
    lines.append("| 博彩公司 | 第一热门 | 赔率 | 第二热门 | 赔率 |")
    lines.append("|:--|:--|--:|:--|--:|")
    for p in sorted(chosen, key=lambda x: x["book"]):
        ranked = sorted(p["teams"].items(), key=lambda kv: kv[1])
        t1, pr1 = ranked[0]
        t2, pr2 = ranked[1] if len(ranked) > 1 else ("", "")
        lines.append(f"| {p['book']} | {cn_flag(t1)} | {pr1:.1f} | {cn_flag(t2) if t2 else '-'} | {pr2:.1f} |")
    lines.append("")
    return lines


# ──────────────────────────────────────────────
# 微信推送（PushPlus，markdown 模板；token 来自 secret）
# ──────────────────────────────────────────────
PUSHPLUS_URL = "http://www.pushplus.plus/send"


def push_to_wechat(token: str, title: str, content: str, summary: str = "") -> str:
    """PushPlus 推送到微信（markdown 模板）。token 为空则跳过。"""
    if not token:
        return "未配置 PUSHPLUS_TOKEN，跳过微信推送"
    try:
        payload = {"token": token, "title": title, "content": content,
                   "template": "markdown", "channel": "wechat"}
        if summary:
            payload["summary"] = summary
        r = requests.post(PUSHPLUS_URL, json=payload, timeout=30)
        d = r.json()
        if d.get("code") == 200:
            return "已推送到微信（PushPlus）"
        return f"⚠️ 微信推送失败：{d.get('msg')}"
    except Exception as e:
        return f"⚠️ 微信推送异常：{type(e).__name__}: {e}"


def _outright_consensus(pools):
    """从 pools 取共识夺冠概率 [(team, pct), ...]，降序。"""
    by_mkt = defaultdict(list)
    for p in pools:
        by_mkt[p["market_key"]].append(p)
    if not by_mkt:
        return []
    chosen = by_mkt[max(by_mkt, key=lambda k: len(by_mkt[k]))]
    tp = defaultdict(list)
    for p in chosen:
        inv = {t: 1 / pr for t, pr in p["teams"].items()}
        tot = sum(inv.values())
        for t, iv in inv.items():
            tp[t].append(iv / tot * 100)
    return sorted(((t, sum(v) / len(v)) for t, v in tp.items()), key=lambda x: -x[1])


def build_wechat_md(matches, pools, now_str, quota) -> str:
    """手机端精简版：outright 全表 + h2h 共识 + 代表性几家。"""
    L = ["# ⚽ 世界杯 · 赔率版报告", ""]
    L.append(f"> {now_str}（北京）｜ The Odds API 市场赔率去水反推")
    if quota:
        L.append(f"> {quota}")
    L.append("")

    cons = _outright_consensus(pools) if pools else []
    if cons:
        L.append("## 🏆 捧杯概率（outright，含加时点球）")
        L.append("")
        for i, (t, p) in enumerate(cons, 1):
            mk = "🥇" if i == 1 else ("🥈" if i == 2 else f"{i}")
            L.append(f"{mk} {cn_flag(t)} **{p:.1f}%**")
        L.append("")

    if matches:
        for mt in matches:
            books = mt["books"]
            n = len(books)
            avg = lambda k: sum(b[k] for b in books) / n
            L.append(f"## 🏟️ 决赛 {cn_flag(mt['home'])} vs {cn_flag(mt['away'])}（90分钟）")
            L.append(f"_采样 {n} 家，下表挑代表性几家（完整 {n} 家见仓库文件）_")
            L.append("")
            L.append("| 公司 | 主胜 | 平 | 客胜 | 主% | 客% |")
            L.append("|:--|--:|--:|--:|--:|--:|")
            want = ["Pinnacle", "DraftKings", "Betfair", "William Hill",
                    "Betano (UK)", "BetUS", "888sport", "Unibet (FR)"]
            pick = [b for b in books if b["title"] in want]
            if len(pick) < 4:
                pick = books[:6]
            for b in pick:
                L.append(f"| {b['title']} | {b['ph']:.2f} | {b['pd']:.2f} | {b['pa']:.2f} | {b['ih']:.0f} | {b['ia']:.0f} |")
            L.append(f"| **共识** | **{avg('ph'):.2f}** | **{avg('pd'):.2f}** | **{avg('pa'):.2f}** | **{avg('ih'):.0f}** | **{avg('ia'):.0f}** |")
            L.append("")
            fav_home = avg("ih") >= avg("ia")
            fav = mt["home"] if fav_home else mt["away"]
            L.append(f"🏅 最看好 **{cn_flag(fav)}**（共识 {max(avg('ih'), avg('ia')):.0f}%）")
            L.append("")

    L.append("_去水=剔除博彩公司抽水后反推；完整 49 家见 reports/赔率报告.md_")
    return "\n".join(L)


def main():
    _bj = dt.datetime.utcnow() + dt.timedelta(hours=8)
    now_str = _bj.strftime("%Y-%m-%d %H:%M")
    today = _bj.date()

    key = (os.environ.get("ODDS_API_KEY") or "").strip()
    md = ["# ⚽ 2026 世界杯 · 赔率版报告（市场视角）\n"]
    md.append(f"> 生成时间：**{now_str}**（北京时间）｜ 数据源：[The Odds API](https://the-odds-api.com)"
              f"（h2h + outright，eu/uk/us 三区聚合）")

    if not key:
        md.append("> ⚠️ 未配置 ODDS_API_KEY，无法拉取赔率。")
        md.append("\n---\n")
        _write(today, "\n".join(md))
        print("\n".join(md))
        return

    # 探测 sport 列表（找 outright 入口）
    sports_hits, sports_resp, sports_status = list_wc_sports(key)
    md.append(f"> sport 探测：{sports_status}")

    quota = ""
    # 1. h2h
    matches, h2h_resp, h2h_status = fetch_h2h(key)
    if h2h_resp:
        quota = _quota(h2h_resp)
    md.append(f"> h2h：{h2h_status}")
    if quota:
        md.append(f"> {quota}")
    md.append("\n---\n")

    # 2. outright
    pools, outright_log = fetch_outrights(key, sports_hits)
    if sports_resp:
        q2 = _quota(sports_resp)
        if q2 and not quota:
            quota = q2
    md.append("## 🏟️ 一、决赛赔率（各家博彩公司对比 · 去水反推）\n")
    md.append("_主胜/平/客胜为**常规时间 90 分钟**结果（平=进入加时点球）；去水后各家均值=共识。_\n")
    if matches:
        for mt in matches:
            md.extend(render_h2h_match(mt))
    else:
        md.append("_当前无 h2h 赔率——决赛可能已开赛/结束，或 API 暂未挂该场。_\n")
    md.append("\n---\n")

    md.append("## 🏆 二、夺冠赔率（市场隐含夺冠概率，替代蒙特卡洛）\n")
    md.append("_outright = **捧杯概率**（含加时与点球），直接反推、替代模型蒙特卡洛。_\n")
    if pools:
        md.extend(render_outright(pools))
    else:
        md.append("_未取到 outright 冠军赔率——赛事可能已结束，或该 sport 不提供 outright。_\n")
        if outright_log:
            md.append("\n<details><summary>调试：outright 探测日志</summary>\n\n```\n"
                      + "\n".join(outright_log) + "\n```\n</details>\n")
    md.append("\n---\n")

    md.append("## 📌 说明\n")
    md.append("- **去水（de-vig）**：把赔率反推为隐含概率后按总和归一，剔除博彩公司抽水（overround），"
              "得到更接近「真实」的市场概率；各家分别去水后再取均值=共识。")
    md.append("- 决赛 h2h 为主胜/平/客胜三项；outright 直接反推夺冠概率，**替代原模型蒙特卡洛模拟**——"
              "这是「赔率为主」最直接的夺冠信号。")
    md.append("- 赔率区间=同一结果各家开出的最低~最高赔率；极差越大说明各家分歧越大。")
    md.append("- 数据来自 The Odds API（eu/uk/us 三区聚合），各公司更新时间不一；本报告为瞬时快照。")
    md.append(f"\n_本报告由 odds-report workflow 手动触发生成。_\n")

    report = "\n".join(md)
    _write(today, report)
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\n✅ 报告已写入：reports/赔率报告.md")
    if quota:
        print(f"📊 {quota}")

    # 推送精简版到微信（PushPlus）
    ppt = (os.environ.get("PUSHPLUS_TOKEN") or "").strip()
    wc_md = build_wechat_md(matches, pools, now_str, quota)
    digest = ""
    cons = _outright_consensus(pools) if pools else []
    if cons:
        digest = "捧杯 " + "、".join(f"{cn(t)} {p:.0f}%" for t, p in cons[:2])
    push_msg = push_to_wechat(ppt, f"⚽ 世界杯赔率版报告 · {today.isoformat()}",
                              wc_md, summary=digest)
    print(f"📲 {push_msg}")


def _write(today, report):
    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "赔率报告.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

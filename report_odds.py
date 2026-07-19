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


def push_to_wechat(token: str, title: str, content: str, template: str = "markdown",
                   summary: str = "") -> str:
    """PushPlus 推送到微信。template 可选 'markdown'/'html'；token 为空则跳过。
    html 模板下把纯文本 summary 放到 content 最前，避免消息预览抓到 HTML 标签代码。"""
    if not token:
        return "未配置 PUSHPLUS_TOKEN，跳过微信推送"
    try:
        if summary and template == "html":
            content = f"{summary}\n" + content
        payload = {"token": token, "title": title, "content": content,
                   "template": template, "channel": "wechat"}
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


# ──────────────────────────────────────────────
# 微信 HTML 报告（PushPlus template=html，样式与每日报告同款）
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
table.grid{width:100%;border-collapse:collapse;font-size:13.5px;}
table.grid th{background:#f0f7f1;color:#3a7d4f;font-weight:600;padding:7px 4px;font-size:12.5px;text-align:center;}
table.grid td{padding:7px 4px;border-bottom:1px solid #f0f2f4;text-align:center;vertical-align:middle;}
table.grid td.l,table.grid th.l{text-align:left;}
table.grid tr:last-child td{border-bottom:none;}
table.grid.consensus td{background:#f0f7f1;font-weight:700;}
table.grid.compact{font-size:12px;}
table.grid.compact td,table.grid.compact th{padding:5px 3px;}
.t{white-space:nowrap;color:#666;font-size:12.5px;}
.muted{color:#8a9099;font-size:12px;}
.en{color:#aab0b6;font-size:11px;}
.rk{color:#9aa0a6;font-weight:700;width:32px;}
.bars{margin:6px 0 10px;}
.bar-title{font-size:11px;color:#8a9099;font-weight:600;margin:2px 0;}
.bar-row{display:block;margin:6px 0;font-size:13px;}
.bar-label{display:inline-block;width:30%;vertical-align:middle;color:#555;}
.bar-track{display:inline-block;vertical-align:middle;height:14px;background:#eef1f4;border-radius:7px;overflow:hidden;}
.bar-fill{display:block;height:100%;border-radius:7px;}
.bar-val{display:inline-block;width:16%;text-align:right;vertical-align:middle;font-weight:600;font-variant-numeric:tabular-nums;color:#2b2f33;}
.kv{font-size:13px;color:#444;margin:4px 0;}
.match{background:#fafbfc;border:1px solid #eef1f4;border-radius:12px;padding:11px 12px;margin:9px 0;}
.match-head{font-size:16px;font-weight:700;}
.match-sub{margin:2px 0 8px;}
.vs{color:#b8bec5;font-size:12px;margin:0 3px;font-weight:400;}
ul.notes{margin:6px 0 0;padding-left:18px;font-size:12.5px;color:#666;}
ul.notes li{margin:4px 0;}
.footer{text-align:center;color:#aab0b6;font-size:11px;margin:14px 0 4px;}
"""


def _h(s) -> str:
    import html as _html
    return _html.escape(str(s), quote=False)


def _team_html(name: str) -> str:
    f = flag(name)
    return f"{f} {cn(name)}" if f else cn(name)


def _prob_bar(label: str, pct: float, color: str) -> str:
    pct = max(0.0, min(100.0, pct))
    return (
        '<div class="bar-row">'
        f'<span class="bar-label">{_h(label)}</span>'
        f'<span class="bar-track" style="width:52%"><span class="bar-fill" style="width:{pct:.1f}%;background:{color}"></span></span>'
        f'<span class="bar-val">{pct:.1f}%</span>'
        '</div>'
    )


def render_html_h2h(mt) -> str:
    home, away = mt["home"], mt["away"]
    books = mt["books"]
    n = len(books)
    avg = lambda k: sum(b[k] for b in books) / n
    ct = mt.get("commence", "")
    sub = (f"开赛（北京）{_bj_from_iso(ct)} · 采样 {n} 家博彩公司") if ct else f"采样 {n} 家博彩公司"

    bars = ('<div class="bar-title">共识概率（去水后各家均值）</div>'
            + _prob_bar(f"主胜 · {cn(home)}", avg("ih"), "#0fae57")
            + _prob_bar("平局", avg("id"), "#d97706")
            + _prob_bar(f"客胜 · {cn(away)}", avg("ia"), "#2563eb"))

    rows = []
    for b in books:
        rows.append(
            f'<tr><td class="l">{_h(b["title"])}</td>'
            f'<td>{b["ph"]:.2f}</td><td>{b["pd"]:.2f}</td><td>{b["pa"]:.2f}</td>'
            f'<td>{b["ih"]:.0f}</td><td>{b["ia"]:.0f}</td></tr>'
        )
    rows.append(
        f'<tr class="consensus"><td class="l">共识（均值）</td>'
        f'<td>{avg("ph"):.2f}</td><td>{avg("pd"):.2f}</td><td>{avg("pa"):.2f}</td>'
        f'<td>{avg("ih"):.0f}</td><td>{avg("ia"):.0f}</td></tr>'
    )
    table = ('<table class="grid compact"><thead><tr>'
             '<th class="l">博彩公司</th><th>主胜</th><th>平</th><th>客胜</th><th>主%</th><th>客%</th>'
             '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>')

    fav_home = avg("ih") >= avg("ia")
    fav = home if fav_home else away
    fav_pct = max(avg("ih"), avg("ia"))
    return (
        '<div class="match">'
        f'<div class="match-head">{_team_html(home)} <span class="vs">VS</span> {_team_html(away)}</div>'
        f'<div class="match-sub muted">{_h(sub)}　·　主胜/平/客胜 = 常规 90 分钟</div>'
        f'<div class="bars">{bars}</div>'
        f'<div class="kv">🏅 最看好 <b>{_team_html(fav)}</b>（共识 {fav_pct:.0f}%）</div>'
        f'{table}'
        '</div>'
    )


def render_html_outright(pools) -> str:
    cons = _outright_consensus(pools)
    if not cons:
        return '<p class="muted">未取到 outright 冠军赔率。</p>'
    by_mkt = defaultdict(list)
    for p in pools:
        by_mkt[p["market_key"]].append(p)
    chosen = by_mkt[max(by_mkt, key=lambda k: len(by_mkt[k]))]
    n_books = len(chosen)

    colors = ["#f0a500", "#9aa0a6", "#c8ccd1"]
    bars = '<div class="bar-title">市场隐含捧杯概率（去水后跨家均值）</div>'
    for i, (t, p) in enumerate(cons):
        bars += _prob_bar(f"{cn(t)}", p, colors[i % len(colors)])

    rows = []
    for p in sorted(chosen, key=lambda x: x["book"]):
        ranked = sorted(p["teams"].items(), key=lambda kv: kv[1])
        t1, pr1 = ranked[0]
        t2, pr2 = ranked[1] if len(ranked) > 1 else ("", 0)
        rows.append(
            f'<tr><td class="l">{_h(p["book"])}</td>'
            f'<td class="l">{_team_html(t1)}</td><td>{pr1:.1f}</td>'
            f'<td class="l">{_team_html(t2) if t2 else "-"}</td><td>{pr2:.1f}</td></tr>'
        )
    table = ('<table class="grid"><thead><tr>'
             '<th class="l">博彩公司</th><th class="l">第一热门</th><th>赔率</th>'
             '<th class="l">第二热门</th><th>赔率</th>'
             '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>')
    return f'<p class="muted">采样 {n_books} 家，去水后跨家均值（含加时与点球）</p><div class="bars">{bars}</div>{table}'


def render_html_report_odds(now_str, h2h_status, quota, matches, pools, digest) -> str:
    """组装完整的微信 HTML 报告。"""
    P = []
    title_text = "⚽ 世界杯 · 赔率版报告"
    P.append('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">'
             f'<title>{_h(title_text)}</title>'
             f'<meta name="description" content="{_h(digest)}">'
             f'<style>{_HTML_CSS}</style></head><body><div class="wrap">')

    hdr = ['<div class="hdr"><h1>⚽ 世界杯 · 赔率版报告</h1>'
           f'<p>🕒 生成时间（北京）{_h(now_str)} ｜ The Odds API 市场赔率去水反推</p>']
    if h2h_status:
        hdr.append(f'<p>💰 {_h(h2h_status)}</p>')
    if quota:
        hdr.append(f'<p>📊 {_h(quota)}</p>')
    hdr.append('</div>')
    P.append("".join(hdr))

    body = "".join(render_html_h2h(mt) for mt in matches) if matches \
        else '<p class="muted">当前无 h2h 赔率--决赛可能已开赛/结束。</p>'
    P.append('<div class="card"><h2>🏟️ 一、决赛赔率（各家对比 · 去水反推）</h2>' + body + '</div>')

    P.append('<div class="card"><h2>🏆 二、捧杯概率（outright，含加时点球）</h2>'
             + render_html_outright(pools) + '</div>')

    P.append('<div class="card"><h2>📌 说明</h2><ul class="notes">'
             '<li>去水（de-vig）：赔率反推为隐含概率后归一，剔除博彩公司抽水，更接近「真实」市场概率。</li>'
             '<li>h2h = 常规 90 分钟结果；outright = 捧杯概率（含加时点球），替代模型蒙特卡洛。</li>'
             '<li>数据来自 The Odds API（eu/uk/us 三区聚合），瞬时快照。</li>'
             '</ul></div>')

    P.append('<div class="footer">- 赔率版报告 · odds-report workflow 生成 -</div>')
    P.append('</div></body></html>')
    return "\n".join(P)


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

    # 推送 HTML 版到微信（PushPlus html 模板，样式同每日报告）
    ppt = (os.environ.get("PUSHPLUS_TOKEN") or "").strip()
    cons = _outright_consensus(pools) if pools else []
    digest = ""
    if cons:
        digest = "捧杯 " + "、".join(f"{cn(t)} {p:.0f}%" for t, p in cons[:2])
    if matches:
        mt0 = matches[0]
        nb = len(mt0["books"])
        sh = sum(b["ih"] for b in mt0["books"]) / nb
        sa = sum(b["ia"] for b in mt0["books"]) / nb
        digest = f"决赛 {cn(mt0['home'])} vs {cn(mt0['away'])} 共识 {sh:.0f}%/{sa:.0f}%　｜　" + digest
    html_report = render_html_report_odds(now_str, h2h_status, quota, matches, pools, digest)
    # 本地存一份预览，方便在浏览器里看推送效果
    (Path(__file__).resolve().parent / "reports").mkdir(parents=True, exist_ok=True)
    (Path(__file__).resolve().parent / "reports" / "preview_odds.html").write_text(html_report, encoding="utf-8")
    push_msg = push_to_wechat(ppt, f"⚽ 世界杯赔率版报告 · {today.isoformat()}",
                              html_report, template="html", summary=digest)
    print(f"📲 {push_msg}")


def _write(today, report):
    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "赔率报告.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

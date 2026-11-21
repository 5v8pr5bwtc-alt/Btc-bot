#!/usr/bin/env python3
"""
BTC 虛擬資金自動交易機器人 (Paper Trading) v3 - GitHub Actions 單次執行版
- 虛擬本金 100 USDT,多空雙向,可設定槓桿 (LEVERAGE,最高 100 倍),含爆倉模擬
- 多指標共振評分:順勢(EMA20/50) 為必要條件,再看 6 項加分:
  EMA200 順勢、RSI 轉折、MACD 同向、斐波那契回撤區、量能放大、K 線確認
  分數 >= MIN_SCORE 才開單;>= HIGH_SCORE 視為高信心,分開統計勝率
- 三段停利 (TP1/TP2/TP3) + 保本移動停損;達標的那一段先落袋,不用等最終目標
- 每次開倉/平倉、每 6 小時回報(做了什麼、為什麼沒開單)、持倉每 3 小時回報浮動損益
- Telegram 傳 /status 查目前狀態 (下次排程執行時回覆)
- 回測:python btc_paper_bot.py --backtest 60  (用過去 60 天歷史行情驗證策略與槓桿)
- 測試通知:python btc_paper_bot.py --test
"""
import os, sys, time, csv, json
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd

# ================= 設定 (可自行調整) =================
START_CASH = 100.0
LEVERAGE = 20          # 槓桿倍數 1~100。越高越容易爆倉 (100 倍約逆行 0.5% 就爆)
MARGIN_PCT = 0.20      # 每單用多少比例的資金當保證金;倉位大小 = 保證金 x 槓桿
FEE = 0.0005           # 單邊手續費 0.05%,以倉位名目價值計算
MMR = 0.005            # 維持保證金率 0.5% (用來估算爆倉價)
ATR_MULT = 1.5         # 停損 = 1.5 倍 ATR
RR = 2.0               # 單一停利模式的風報比 1:2
TP_MODE = "multi"      # "multi" = 三段停利 TP1/TP2/TP3;"single" = 單一停利
TP_R = (1.0, 2.0, 3.0) # 三段停利距離 = 停損距離的幾倍 (1R / 2R / 3R)
TP_FRAC = (0.4, 0.3, 0.3)  # 三段各平掉多少比例 (最後一段會平光剩餘)
MOVE_STOP = True       # TP1 達標後停損移到保本,TP2 達標後移到 TP1
MIN_SCORE = 4          # 6 項加分條件至少符合幾項才開單
HIGH_SCORE = 5         # 幾分以上算「高信心」
REPORT_HOURS = 3       # 持倉期間每幾小時回報浮動損益
SUMMARY_HOURS = 6      # 每幾小時回報「做了什麼、為什麼沒開單」
MAX_LOOKBACK = 6       # 排程被延遲時,最多回頭補檢查幾根 K 線
# =====================================================
LEVERAGE = max(1, min(100, int(LEVERAGE)))
STATE_FILE, TRADE_LOG = "state.json", "trades.csv"
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
TZ = timezone(timedelta(hours=8))  # 台灣時間
MAX_SCORE = 6


def now():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M")


def fmt_t(ms):
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%m-%d %H:%M")


def notify(msg):
    print(f"[{now()}] {msg}\n")
    if TG_TOKEN and TG_CHAT_ID:
        try:
            r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                              data={"chat_id": TG_CHAT_ID, "text": msg[:4000]}, timeout=10)
            if r.status_code != 200:
                print("Telegram 失敗:", r.status_code, r.text)
        except Exception as e:
            print("Telegram 失敗:", e)


# ---------- 資料 (Binance 公開行情,失敗時改用 Kraken) ----------
def get_klines():
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": "BTCUSDT", "interval": "15m", "limit": 500}, timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json()).iloc[:, :6]
        df.columns = ["t", "open", "high", "low", "close", "vol"]
    except Exception as e:
        print("Binance 失敗,改用 Kraken:", e)
        r = requests.get("https://api.kraken.com/0/public/OHLC",
                         params={"pair": "XBTUSDT", "interval": 15}, timeout=10)
        r.raise_for_status()
        rows = next(v for k, v in r.json()["result"].items() if k != "last")
        df = pd.DataFrame(rows).iloc[:, :7]
        df.columns = ["t", "open", "high", "low", "close", "vwap", "vol"]
        df["t"] = df["t"].astype("int64") * 1000
        df = df.tail(500).reset_index(drop=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df["t"] = df["t"].astype("int64")
    return df


def add_indicators(df):
    c = df["close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    d = c.diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + g / l)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"] = df["macd"] - df["sig"]
    df["vol20"] = df["vol"].rolling(20).mean()
    return df


# ---------- 狀態 ----------
def new_window():
    return {"start": time.time(), "bars": 0, "flat": 0, "weak": 0, "held": 0, "stale": 0,
            "best": None, "opens": [], "closes": []}


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"cash": START_CASH, "position": None, "wins": 0, "losses": 0, "last_bar": 0}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)


def win(s):
    if "win" not in s:
        s["win"] = new_window()
    return s["win"]


def log_trade(row):
    if os.path.exists(TRADE_LOG):  # 欄位改版時,舊檔改名保留
        with open(TRADE_LOG, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if header != list(row.keys()):
            os.replace(TRADE_LOG, "trades_old.csv")
    new = not os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)


# ---------- 多指標共振評分 ----------
def evaluate(df, i=-2):
    return evaluate_at(df, len(df) + i)


def evaluate_at(df, pos):
    """評估第 pos 根已收盤 K 線。回傳 dict:side/score/hits/misses/trend/signal..."""
    cur, prev = df.iloc[pos], df.iloc[pos - 1]
    out = {"t": int(cur["t"]), "side": None, "score": 0, "hits": [], "misses": [],
           "atr": float(cur["atr"]), "close": float(cur["close"]), "trend": "盤整", "signal": False}
    if pos < 200 or cur[["ema200", "atr", "macd", "vol20"]].isna().any():
        out["trend"] = "資料不足"
        return out
    up = cur["ema20"] > cur["ema50"] and cur["close"] > cur["ema50"]
    dn = cur["ema20"] < cur["ema50"] and cur["close"] < cur["ema50"]
    if not (up or dn):
        return out  # 均線糾纏,沒有明確趨勢,不交易
    out["side"], out["trend"] = ("long", "多頭") if up else ("short", "空頭")
    prior = df.iloc[pos - 6:pos]
    seg = df.iloc[pos - 95:pos + 1]
    hi, lo = seg["high"].max(), seg["low"].min()
    rng = hi - lo
    ih, il = seg["high"].values.argmax(), seg["low"].values.argmin()
    fib = False
    if rng >= 2 * cur["atr"]:
        if up and il < ih:
            fib = 0.382 <= (hi - cur["close"]) / rng <= 0.618
        if dn and ih < il:
            fib = 0.382 <= (cur["close"] - lo) / rng <= 0.618
    if up:
        checks = [("EMA200 順勢", cur["close"] > cur["ema200"]),
                  ("RSI 回檔轉強", prior["rsi"].min() < 45 <= cur["rsi"] and cur["rsi"] > prev["rsi"]),
                  ("MACD 同向", cur["macd"] > cur["sig"] and cur["hist"] > prev["hist"]),
                  ("斐波那契回撤區", fib),
                  ("量能放大", cur["vol"] > cur["vol20"]),
                  ("K 線確認", cur["close"] > cur["open"] and cur["close"] > prev["close"])]
    else:
        checks = [("EMA200 順勢", cur["close"] < cur["ema200"]),
                  ("RSI 反彈轉弱", prior["rsi"].max() > 55 >= cur["rsi"] and cur["rsi"] < prev["rsi"]),
                  ("MACD 同向", cur["macd"] < cur["sig"] and cur["hist"] < prev["hist"]),
                  ("斐波那契回撤區", fib),
                  ("量能放大", cur["vol"] > cur["vol20"]),
                  ("K 線確認", cur["close"] < cur["open"] and cur["close"] < prev["close"])]
    out["hits"] = [n for n, ok in checks if ok]
    out["misses"] = [n for n, ok in checks if not ok]
    out["score"] = len(out["hits"])
    out["signal"] = out["score"] >= MIN_SCORE
    return out


def why_text(ev):
    nm = "做多" if ev["side"] == "long" else "做空"
    return (f"{ev['trend']}順勢{nm} {ev['score']}/{MAX_SCORE} 分|符合:{'、'.join(ev['hits']) or '無'}|"
            f"未符合:{'、'.join(ev['misses']) or '無'}")


# ---------- 開倉 / 平倉 ----------
def trade_levels(side, price, atr, leverage, mode=None):
    """回傳 (停損, [各段停利價], [各段比例], 爆倉價)"""
    mode = TP_MODE if mode is None else mode
    d = 1 if side == "long" else -1
    r = ATR_MULT * atr
    stop = price - d * r
    liq = price * (1 - d * (1 / leverage - MMR))
    if mode == "multi":
        tps = [price + d * k * r for k in TP_R]
        tot = sum(TP_FRAC)
        fracs = [f / tot for f in TP_FRAC]
    else:
        tps, fracs = [price + d * RR * r], [1.0]
    return stop, tps, fracs, liq


def size_position(cash, price, leverage, margin_pct):
    margin = cash * margin_pct
    notional = margin * leverage
    return margin, notional, notional / price


def part_close(side, entry, q, price, margin_part):
    """平掉 q 數量。回傳 (這段的已實現損益(已扣平倉手續費), 退回帳戶的金額)"""
    d = 1 if side == "long" else -1
    gross = (price - entry) * q * d
    fee = q * price * FEE
    return gross - fee, max(0.0, margin_part + gross - fee)


def stop_after_tp(side, entry, tps, hit):
    """TP1 達標 -> 保本 (含來回手續費);TP2 達標 -> 移到 TP1"""
    d = 1 if side == "long" else -1
    return entry * (1 + d * 2 * FEE) if hit == 1 else tps[hit - 2]


def tier_of(score):
    return "high" if score >= HIGH_SCORE else "mid"


def open_position(s, ev, price):
    side = ev["side"]
    stop, tps, fracs, liq = trade_levels(side, price, ev["atr"], LEVERAGE)
    margin, notional, qty = size_position(s["cash"], price, LEVERAGE, MARGIN_PCT)
    s["cash"] -= margin + notional * FEE
    reason = why_text(ev)
    s["position"] = {"side": side, "entry": price, "qty": qty, "rem": qty, "margin": margin,
                     "lev": LEVERAGE, "stop": stop, "tps": tps, "fracs": fracs, "target": tps[-1],
                     "tp_hit": 0, "realized": 0.0, "liq": liq, "score": ev["score"],
                     "time": now(), "ts": int(time.time() * 1000), "reason": reason}
    s["last_report"] = time.time()
    name = "做多" if side == "long" else "做空"
    win(s)["opens"].append(f"{now()[5:]} {name} @{price:,.0f}(信心 {ev['score']}/{MAX_SCORE})")
    save_state(s)
    icon = "🟢" if side == "long" else "🔴"
    tag = "高信心" if ev["score"] >= HIGH_SCORE else "一般信心"
    tp_lines = "\n".join(f"TP{k + 1}:{tp:,.2f}(平 {fracs[k] * 100:.0f}%)" for k, tp in enumerate(tps))
    notify(f"{icon} 【虛擬開倉】BTC {name} {LEVERAGE}x\n進場價:{price:,.2f}\n停損:{stop:,.2f}\n{tp_lines}\n"
           f"爆倉價:{liq:,.2f}\n保證金:{margin:.2f} USDT(倉位 {notional:.2f})\n"
           f"信心:{ev['score']}/{MAX_SCORE}({tag})\n依據:{reason}")


def take_tp(s, level):
    """第 tp_hit+1 段停利達標:平掉這一段,把停損往上(下)移"""
    p = s["position"]
    side = p.get("side", "long")
    tps, fracs = p.get("tps", [p["target"]]), p.get("fracs", [1.0])
    idx = p.get("tp_hit", 0)
    if idx == len(tps) - 1:  # 最後一段:平光剩餘
        return close_position(s, level, f"觸及停利 TP{idx + 1}")
    rem = p.get("rem", p["qty"])
    q = min(p["qty"] * fracs[idx], rem)
    r, back = part_close(side, p["entry"], q, level, p["margin"] * q / p["qty"])
    s["cash"] += back
    p["realized"] = p.get("realized", 0.0) + r
    p["rem"] = rem - q
    p["tp_hit"] = idx + 1
    if MOVE_STOP:
        new = stop_after_tp(side, p["entry"], tps, idx + 1)
        p["stop"] = max(p["stop"], new) if side == "long" else min(p["stop"], new)
    name = "做多" if side == "long" else "做空"
    win(s)["closes"].append(f"{now()[5:]} {name} TP{idx + 1} 落袋 {r:+.2f}U")
    save_state(s)
    notify(f"🎯 【TP{idx + 1} 達標】{name} {p.get('lev', 1)}x 平倉 {fracs[idx] * 100:.0f}%\n"
           f"出場價:{level:,.2f}\n這一段已落袋:{r:+.2f} USDT\n"
           f"剩餘倉位 {p['rem'] / p['qty'] * 100:.0f}%,停損移至 {p['stop']:,.2f}"
           f"{'(保本)' if idx == 0 and MOVE_STOP else ''}\n下一目標 TP{idx + 2}:{tps[idx + 1]:,.2f}")


def close_position(s, price, why, liquidated=False):
    """平掉剩餘倉位並結算整筆交易 (含先前已落袋的各段停利)"""
    p = s["position"]
    side = p.get("side", "long")
    d = 1 if side == "long" else -1
    lev, qty = p.get("lev", 1), p["qty"]
    rem = p.get("rem", qty)
    margin_rem = p.get("margin", qty * p["entry"] / lev) * rem / qty
    if liquidated:  # 爆倉:剩餘部位賠光保證金
        r, back = -margin_rem, 0.0
    else:
        r, back = part_close(side, p["entry"], rem, price, margin_rem)
    s["cash"] += back
    hit = p.get("tp_hit", 0)
    pnl = p.get("realized", 0.0) + r - qty * p["entry"] * FEE
    s["wins" if pnl > 0 else "losses"] += 1
    s.setdefault("tier", {"high": [0, 0], "mid": [0, 0]})[tier_of(p.get("score", 0))][0 if pnl > 0 else 1] += 1
    s["position"] = None
    name = "做多" if d == 1 else "做空"
    win(s)["closes"].append(f"{now()[5:]} {name} 整筆 {'賺' if pnl > 0 else '虧'} {pnl:+.2f}U({why})")
    save_state(s)
    total = s["wins"] + s["losses"]
    log_trade({"open_time": p["time"], "close_time": now(), "side": name, "leverage": lev,
               "score": p.get("score", ""), "tp_hit": hit, "entry": round(p["entry"], 2),
               "exit": round(price, 2), "pnl_usdt": round(pnl, 4), "result": "WIN" if pnl > 0 else "LOSS",
               "close_why": why, "reason": p["reason"].replace(",", ";")})
    extra = f"\n(已先落袋 {'TP1' if hit == 1 else f'TP1~TP{hit}'})" if hit else ""
    notify(f"{'✅ 賺錢' if pnl > 0 else '❌ 虧錢'} 【虛擬平倉】{name} {lev}x({why})\n"
           f"進場:{p['entry']:,.2f} → 最後出場:{price:,.2f}{extra}\n整筆損益:{pnl:+.2f} USDT\n"
           f"目前資金:{s['cash']:.2f} USDT(本金 {START_CASH:.0f})\n"
           f"累計:{s['wins']} 勝 / {s['losses']} 敗,勝率 {s['wins'] / total * 100:.1f}%\n{tier_text(s)}")


def process_bar(s, high, low):
    """用一根 K 線的最高/最低價判斷:先看停損/爆倉 (保守),再依序檢查 TP。回傳是否已全部平倉。"""
    p = s["position"]
    long_ = p.get("side", "long") == "long"
    liq, stop = p.get("liq"), p["stop"]
    if long_:
        adverse = max(stop, liq) if liq else stop
        bad = low <= adverse
    else:
        adverse = min(stop, liq) if liq else stop
        bad = high >= adverse
    if bad:
        is_liq = bool(liq) and liq != stop and adverse == liq
        hit = p.get("tp_hit", 0)
        why = "爆倉" if is_liq else ("觸及停損" if hit == 0 else ("保本出場" if hit == 1 else "移動停損出場"))
        close_position(s, adverse, why, liquidated=is_liq)
        return True
    tps = p.get("tps", [p["target"]])
    while p.get("tp_hit", 0) < len(tps):
        lvl = tps[p.get("tp_hit", 0)]
        if (long_ and high >= lvl) or (not long_ and low <= lvl):
            take_tp(s, lvl)
            if s["position"] is None:
                return True
        else:
            break
    return False


def check_exit(s, df):
    """逐根重播開倉後的 K 線 (排程延遲時也不會漏掉),最後再用最新價檢查一次。"""
    p = s["position"]
    for _, row in df[df["t"] >= p["ts"]].iterrows():
        if process_bar(s, row["high"], row["low"]):
            return
    if df.iloc[-1]["t"] < p["ts"]:  # 最新這根 K 線在開倉前就開始了,用最新價補檢查一次
        price = df.iloc[-1]["close"]
        process_bar(s, price, price)


# ---------- 損益 / 狀態文字 ----------
def unreal_gross(p, price):
    d = 1 if p.get("side", "long") == "long" else -1
    return (price - p["entry"]) * p.get("rem", p["qty"]) * d


def floating_pnl(p, price):
    """本單累計損益 = 已落袋 + 未平倉部分浮盈 - 開倉手續費 - 預估平倉手續費"""
    rem = p.get("rem", p["qty"])
    return p.get("realized", 0.0) + unreal_gross(p, price) - p["qty"] * p["entry"] * FEE - rem * price * FEE


def tier_text(s):
    parts = []
    for k, name in (("high", f"高信心(≥{HIGH_SCORE}分)"), ("mid", f"一般({MIN_SCORE}分)")):
        w, l = s.get("tier", {}).get(k, [0, 0])
        parts.append(f"{name} {w}勝{l}敗" + (f" 勝率 {w / (w + l) * 100:.0f}%" if w + l else ""))
    return "分級勝率:" + " | ".join(parts)


def position_text(s, price):
    p = s["position"]
    name = "做多" if p.get("side", "long") == "long" else "做空"
    pnl = floating_pnl(p, price)
    tps, hit = p.get("tps", [p["target"]]), p.get("tp_hit", 0)
    tp_line = " ".join(f"TP{k + 1}{'✔' if k < hit else ''}:{tp:,.0f}" for k, tp in enumerate(tps))
    liq = f" 爆倉:{p['liq']:,.0f}" if p.get("liq") else ""
    rem_pct = p.get("rem", p["qty"]) / p["qty"] * 100
    return (f"持倉:BTC {name} {p.get('lev', 1)}x(開倉 {p['time']},信心 {p.get('score', '?')}/{MAX_SCORE})\n"
            f"進場:{p['entry']:,.2f} 現價:{price:,.2f} 剩餘倉位 {rem_pct:.0f}%\n"
            f"停損:{p['stop']:,.2f}{liq}\n{tp_line}\n"
            f"本單累計損益:{pnl:+.2f} USDT(含已落袋){'📈' if pnl > 0 else '📉'}")


def equity(s, price):
    e = s["cash"]
    if s["position"]:
        p = s["position"]
        rem = p.get("rem", p["qty"])
        margin_rem = p.get("margin", p["qty"] * p["entry"] / p.get("lev", 1)) * rem / p["qty"]
        e += max(0.0, margin_rem + unreal_gross(p, price) - rem * price * FEE)
    return e


def status_text(s, price):
    total = s["wins"] + s["losses"]
    wr = f"{s['wins'] / total * 100:.1f}%" if total else "尚無完成交易"
    e = equity(s, price)
    body = position_text(s, price) if s["position"] else "目前空手,等待訊號"
    return (f"📊 【目前狀態】BTC 現價 {price:,.2f}(槓桿設定 {LEVERAGE}x)\n{body}\n"
            f"資產:約 {e:.2f} USDT(本金 {START_CASH:.0f},{e - START_CASH:+.2f})\n"
            f"累計:{s['wins']} 勝 / {s['losses']} 敗,勝率 {wr}\n{tier_text(s)}")


def summary_text(s, df, price):
    w = win(s)
    c = df.iloc[-2]
    ev = evaluate(df)
    L = [f"📋 【{SUMMARY_HOURS} 小時回報】{datetime.fromtimestamp(w['start'], TZ).strftime('%m-%d %H:%M')} ~ {now()[5:]}",
         f"判斷 K 線 {w['bars']} 根(持倉中略過 {w['held']} 根)"]
    L.append(f"開單 {len(w['opens'])} 筆" + ("" if not w["opens"] else ":\n" + "\n".join("・" + x for x in w["opens"])))
    L.append(f"平倉/停利 {len(w['closes'])} 筆" + ("" if not w["closes"] else ":\n" + "\n".join("・" + x for x in w["closes"])))
    if not w["opens"]:
        L.append("沒開單的原因:")
        L.append(f"・趨勢不明(均線糾纏):{w['flat']} 根")
        L.append(f"・有趨勢但分數不足 {MIN_SCORE}/{MAX_SCORE}:{w['weak']} 根")
        if w["stale"]:
            L.append(f"・訊號出現但價格已跑遠、不追:{w['stale']} 次")
        if w["held"] and not w["bars"]:
            L.append("・期間都在持倉中")
        b = w["best"]
        if b:
            nm = "做多" if b["side"] == "long" else "做空"
            L.append(f"最接近進場:{fmt_t(b['t'])} {b['trend']}{nm} {b['score']}/{MAX_SCORE} 分\n"
                     f"  符合:{'、'.join(b['hits']) or '無'}\n  缺少:{'、'.join(b['misses']) or '無'}")
    macd = "偏多" if c["macd"] > c["sig"] else "偏空"
    L.append(f"目前市場:{ev['trend']} | 現價 {price:,.0f} | EMA20 {c['ema20']:,.0f} / EMA50 {c['ema50']:,.0f}"
             f" | RSI {c['rsi']:.0f} | MACD {macd}")
    L.append(position_text(s, price) if s["position"] else "目前空手")
    total = s["wins"] + s["losses"]
    L.append(f"累計:{s['wins']} 勝 {s['losses']} 敗" + (f",勝率 {s['wins'] / total * 100:.1f}%" if total else ""))
    L.append(tier_text(s))
    return "\n".join(L)


# ---------- Telegram 指令 / 定時回報 ----------
def get_tg_updates(offset):
    r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                     params={"offset": offset, "timeout": 0}, timeout=10)
    return r.json().get("result", []) if r.status_code == 200 else []


def handle_commands(s, price):
    if not (TG_TOKEN and TG_CHAT_ID):
        return
    try:
        wanted = False
        for u in get_tg_updates(s.get("tg_offset", 0)):
            s["tg_offset"] = u["update_id"] + 1
            m = u.get("message") or {}
            if str(m.get("chat", {}).get("id")) == str(TG_CHAT_ID) and \
                    str(m.get("text", "")).strip().lower().startswith("/status"):
                wanted = True
        save_state(s)
        if wanted:
            notify(status_text(s, price))
    except Exception as e:
        print("讀取指令失敗:", e)


def maybe_report(s, price):
    if s["position"] and time.time() - s.get("last_report", 0) >= REPORT_HOURS * 3600:
        s["last_report"] = time.time()
        save_state(s)
        notify("⏱ 【持倉回報】\n" + position_text(s, price))


def log_line(s, df, price):
    c = df.iloc[-2]
    ev = evaluate(df)
    pos = "持倉" if s["position"] else "空手"
    line = (f"{now()} | BTC {price:,.0f} | EMA20 {c['ema20']:,.0f} EMA50 {c['ema50']:,.0f} | RSI {c['rsi']:.1f} | "
            f"{ev['trend']} {ev['score']}/{MAX_SCORE} | {pos} | {s['wins']}勝{s['losses']}敗")
    print(line)
    f = os.getenv("GITHUB_STEP_SUMMARY")
    if f:
        with open(f, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ---------- 主流程 ----------
def run_once():
    s = load_state()
    w = win(s)
    df = add_indicators(get_klines())
    price = df.iloc[-1]["close"]
    if s["position"]:
        check_exit(s, df)
    handle_commands(s, price)
    if s["position"]:
        maybe_report(s, price)

    bar, last = int(df.iloc[-2]["t"]), s.get("last_bar", 0)
    if bar != last:
        pending = [i for i in range(-2, -2 - MAX_LOOKBACK, -1) if int(df.iloc[i]["t"]) > last][::-1]
        s["last_bar"] = bar
        found = None
        for i in pending:  # 由舊到新評估每一根還沒看過的已收盤 K 線
            if s["position"]:
                w["held"] += 1
                continue
            ev = evaluate(df, i)
            w["bars"] += 1
            if ev["side"] is None:
                w["flat"] += 1
                continue
            if not ev["signal"]:
                w["weak"] += 1
            if w["best"] is None or ev["score"] > w["best"]["score"]:
                w["best"] = {k: ev[k] for k in ("t", "side", "score", "hits", "misses", "trend")}
            if ev["signal"]:
                found = ev
        if found and not s["position"] and s["cash"] >= 1:
            if abs(price - found["close"]) <= found["atr"]:  # 價格已跑太遠就不追
                open_position(s, found, price)
            else:
                w["stale"] += 1
                print("訊號已過期:價格偏離過大,略過")
        save_state(s)

    if time.time() - w["start"] >= SUMMARY_HOURS * 3600:
        notify(summary_text(s, df, price))
        s["win"] = new_window()
    save_state(s)
    log_line(s, df, price)


# ---------- 回測 ----------
def get_history(days):
    """抓過去 days 天的 15 分鐘 K 線 (多抓 300 根給 EMA200 暖機)。Binance 失敗改用 Coinbase。"""
    need = days * 96 + 300
    end_ms = int(time.time() * 1000)
    try:
        rows, st = [], end_ms - need * 900_000
        while st < end_ms:
            r = requests.get("https://data-api.binance.vision/api/v3/klines",
                             params={"symbol": "BTCUSDT", "interval": "15m", "limit": 1000, "startTime": st},
                             timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            rows += [x[:6] for x in data]
            st = data[-1][0] + 900_000
            if len(data) < 1000:
                break
            time.sleep(0.2)
    except Exception as e:
        print("Binance 歷史資料失敗,改用 Coinbase (BTC-USD):", e)
        rows, end = [], datetime.now(timezone.utc)
        for k in range(-(-need // 300)):
            e_ = end - timedelta(seconds=k * 300 * 900)
            s_ = e_ - timedelta(seconds=300 * 900)
            r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                             params={"granularity": 900, "start": s_.isoformat(), "end": e_.isoformat()},
                             headers={"User-Agent": "btc-paper-bot"}, timeout=15)
            r.raise_for_status()
            rows += [[x[0] * 1000, x[3], x[2], x[1], x[4], x[5]] for x in r.json()]
            time.sleep(0.4)
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "vol"])
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df["t"] = df["t"].astype("int64")
    df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    if len(df) and df.iloc[-1]["t"] + 900_000 > end_ms:  # 去掉還沒收盤的最後一根
        df = df.iloc[:-1].reset_index(drop=True)
    return df


def simulate(df, evs, leverage, min_score, margin_pct=None, tp_mode=None, move_stop=None):
    """逐根 K 線模擬:訊號在 K 線收盤後,下一根開盤價進場。
    同一根 K 線先看停損/爆倉 (保守),再依序檢查 TP;停損移動從下一根才生效。"""
    margin_pct = MARGIN_PCT if margin_pct is None else margin_pct
    move_stop = MOVE_STOP if move_stop is None else move_stop
    o, h, lo, c = (df[k].values for k in ("open", "high", "low", "close"))
    cash, pos, pending, trades, curve = START_CASH, None, None, [], [START_CASH]

    def close_rest(price, liqd, why):
        nonlocal cash, pos
        q = pos["rem"]
        mrem = pos["margin"] * q / pos["qty"]
        r, back = (-mrem, 0.0) if liqd else part_close(pos["side"], pos["entry"], q, price, mrem)
        cash += back
        trades.append({"pnl": pos["realized"] + r - pos["qty"] * pos["entry"] * FEE, "side": pos["side"],
                       "score": pos["score"], "liq": liqd, "why": why, "hit": pos["hit"]})
        curve.append(cash)
        pos = None

    for j in range(200, len(df)):
        if pos is None and pending is not None and cash >= 1:
            price = o[j]
            stop, tps, fracs, liq = trade_levels(pending["side"], price, pending["atr"], leverage, tp_mode)
            margin, notional, qty = size_position(cash, price, leverage, margin_pct)
            cash -= margin + notional * FEE
            pos = {"side": pending["side"], "entry": price, "qty": qty, "rem": qty, "margin": margin,
                   "stop": stop, "tps": tps, "fracs": fracs, "liq": liq, "score": pending["score"],
                   "hit": 0, "realized": 0.0}
        pending = None
        if pos:
            long_ = pos["side"] == "long"
            adverse = max(pos["stop"], pos["liq"]) if long_ else min(pos["stop"], pos["liq"])
            bad = lo[j] <= adverse if long_ else h[j] >= adverse
            if bad:
                is_liq = adverse == pos["liq"] and pos["liq"] != pos["stop"]
                why = "爆倉" if is_liq else ("停損" if pos["hit"] == 0 else "保本/移動停損")
                close_rest(adverse, is_liq, why)
            else:
                while pos and pos["hit"] < len(pos["tps"]):
                    lvl = pos["tps"][pos["hit"]]
                    if not (h[j] >= lvl if long_ else lo[j] <= lvl):
                        break
                    idx = pos["hit"]
                    if idx == len(pos["tps"]) - 1:
                        close_rest(lvl, False, f"停利 TP{idx + 1}")
                    else:
                        q = min(pos["qty"] * pos["fracs"][idx], pos["rem"])
                        r, back = part_close(pos["side"], pos["entry"], q, lvl, pos["margin"] * q / pos["qty"])
                        cash += back
                        pos["realized"] += r
                        pos["rem"] -= q
                        pos["hit"] = idx + 1
                        if move_stop:
                            new = stop_after_tp(pos["side"], pos["entry"], pos["tps"], idx + 1)
                            pos["stop"] = max(pos["stop"], new) if long_ else min(pos["stop"], new)
        if pos is None:
            ev = evs[j]
            if ev["side"] and ev["score"] >= min_score:
                pending = ev
    if pos:
        close_rest(c[-1], False, "回測結束")
    return trades, cash, curve


def bt_stats(trades, cash, curve):
    n = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    gp = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    peak, dd = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak if peak > 0 else 0)
    return {"n": n, "w": len(wins), "wr": len(wins) / n * 100 if n else 0.0, "cash": cash,
            "ret": (cash / START_CASH - 1) * 100, "dd": dd * 100,
            "pf": (gp / gl if gl > 0 else float("inf")) if n else 0.0,
            "liq": sum(1 for t in trades if t["liq"])}


def fr(v):
    return f"{v:+.1f}%" if abs(v) < 1000 else (">+999%" if v > 0 else "<-999%")


def rate(ts):
    n = len(ts)
    w = sum(1 for t in ts if t["pnl"] > 0)
    return f"{n} 筆 勝率 {w / n * 100:.0f}%" if n else "0 筆"


def tp_desc(mode=None):
    mode = TP_MODE if mode is None else mode
    if mode != "multi":
        return f"單一停利 1:{RR:g}"
    return "三段停利 " + "/".join(f"{k:g}R" for k in TP_R) + " 各平 " + "/".join(f"{f * 100:.0f}" for f in TP_FRAC) + "%" + \
           ("、TP1 後保本" if MOVE_STOP else "")


def run_backtest(days=60):
    df = add_indicators(get_history(days))
    if len(df) < 400:
        notify(f"回測失敗:只抓到 {len(df)} 根 K 線,資料不足。")
        return
    evs = {j: evaluate_at(df, j) for j in range(200, len(df))}
    trades, cash, curve = simulate(df, evs, LEVERAGE, MIN_SCORE)
    st = bt_stats(trades, cash, curve)
    real_days = (len(df) - 200) / 96
    chg = (df.iloc[-1]["close"] / df.iloc[200]["close"] - 1) * 100
    pf = "∞" if st["pf"] == float("inf") else f"{st['pf']:.2f}"
    L = ["📈 【回測報告】BTC 15 分K",
         f"期間 {fmt_t(df.iloc[200]['t'])} ~ {fmt_t(df.iloc[-1]['t'])}({real_days:.0f} 天,同期 BTC {chg:+.1f}%)",
         f"設定:槓桿 {LEVERAGE}x|保證金 {MARGIN_PCT * 100:.0f}%|分數 ≥{MIN_SCORE}/{MAX_SCORE}|停損 {ATR_MULT}ATR",
         f"出場:{tp_desc()}", ""]
    if not trades:
        L.append("期間內沒有任何交易。可以試試把 MIN_SCORE 調低。")
    else:
        L += [f"交易 {st['n']} 筆(平均每天 {st['n'] / real_days:.1f} 筆)",
              f"勝率 {st['wr']:.1f}%({st['w']} 勝 {st['n'] - st['w']} 敗)",
              f"最終資金 {st['cash']:.2f} USDT(報酬 {fr(st['ret'])})",
              f"最大回撤 {st['dd']:.1f}%|獲利因子 {pf}|爆倉 {st['liq']} 次",
              f"多單 {rate([t for t in trades if t['side'] == 'long'])}|空單 {rate([t for t in trades if t['side'] == 'short'])}",
              "分數分級:" + " | ".join(f"{sc}分 {rate([t for t in trades if t['score'] == sc])}"
                                     for sc in range(MIN_SCORE, MAX_SCORE + 1))]
        if TP_MODE == "multi":
            L.append("到達各段停利的比例:" + " | ".join(
                f"TP{k}: {sum(1 for t in trades if t['hit'] >= k) / st['n'] * 100:.0f}%" for k in range(1, len(TP_R))) +
                f" | TP{len(TP_R)}: {sum(1 for t in trades if t['why'].startswith('停利 TP' + str(len(TP_R)))) / st['n'] * 100:.0f}%")
    L += ["", f"【停利方式比較】(槓桿 {LEVERAGE}x)"]
    for label, mode, mv in (("單一停利", "single", False), ("三段(不保本)", "multi", False), ("三段+保本", "multi", True)):
        t2, c2, cv2 = simulate(df, evs, LEVERAGE, MIN_SCORE, tp_mode=mode, move_stop=mv)
        x = bt_stats(t2, c2, cv2)
        pf2 = "∞" if x["pf"] == float("inf") else f"{x['pf']:.2f}"
        L.append(f"{label} 勝率 {x['wr']:.0f}%|報酬 {fr(x['ret'])}|回撤 {x['dd']:.0f}%|獲利因子 {pf2}")
    for label, mode in (("單一停利", "single"), ("三段+保本", "multi")):
        L += ["", f"【槓桿比較-{label}】"]
        for lev in (1, 5, 10, 20, 50, 100):
            t2, c2, cv2 = simulate(df, evs, lev, MIN_SCORE, tp_mode=mode, move_stop=(mode == "multi"))
            x = bt_stats(t2, c2, cv2)
            L.append(f"{lev:>3}x 勝率 {x['wr']:.0f}%|報酬 {fr(x['ret'])}|回撤 {x['dd']:.0f}%|爆倉 {x['liq']}")
    L += ["", f"【分數門檻比較】(槓桿 {LEVERAGE}x,{tp_desc()})"]
    for ms in (3, 4, 5, 6):
        t2, c2, cv2 = simulate(df, evs, LEVERAGE, ms)
        x = bt_stats(t2, c2, cv2)
        L.append(f"≥{ms} 分 {x['n']:>3} 筆|勝率 {x['wr']:.0f}%|報酬 {fr(x['ret'])}|回撤 {x['dd']:.0f}%")
    L += ["", "⚠ 歷史不代表未來;未計滑價與資金費率;進場以訊號後下一根開盤價計;"
              "同根 K 線停損與停利都觸及時算停損。保本出場也算「贏」,所以三段停利的勝率會偏高,請同時看報酬與獲利因子。"]
    text = "\n".join(L)
    notify(text)
    f = os.getenv("GITHUB_STEP_SUMMARY")
    if f:
        with open(f, "a", encoding="utf-8") as fh:
            fh.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    if "--test" in sys.argv:
        notify("✅ 測試成功!BTC 虛擬交易機器人已連上 Telegram,之後每次下單都會通知你。")
    elif "--backtest" in sys.argv:
        k = sys.argv.index("--backtest") + 1
        run_backtest(int(sys.argv[k]) if k < len(sys.argv) and sys.argv[k].isdigit() else 60)
    elif "--loop" in sys.argv:
        while True:
            try:
                run_once()
            except Exception as e:
                print("錯誤:", e)
            time.sleep(60)
    else:
        run_once()


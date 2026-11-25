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
- 每單平倉後自動製作「AI 虛擬老師」複盤影片 (中文語音 + 字幕) 傳到 Telegram;傳 /demo 可先看示範影片
- 回測:python btc_paper_bot.py --backtest 60  (用過去 60 天歷史行情驗證策略與槓桿)
- 測試通知:python btc_paper_bot.py --test
"""
import os, sys, re, time, csv, json, shutil, subprocess, tempfile, wave
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


# ---------- 新手小教室 (每單附白話說明 + 複盤圖) ----------
_CTX = {"df": None}

EXPLAIN = {
    "EMA200 順勢": "EMA200 是「長期平均價」。價格在它上方=大方向偏多,下方=偏空。順著大方向開單,通常比逆著來容易贏。",
    "RSI 回檔轉強": "RSI 看最近漲跌的力道。跌到 45 以下是回檔、賣壓大;回到 45 以上代表賣壓退了、買盤回來,像彈簧壓到底後彈起。",
    "RSI 反彈轉弱": "RSI 反彈到 55 以上是漲得有點急;再跌回 55 以下代表買盤退了,容易繼續往下。",
    "MACD 同向": "MACD 看動能有沒有加速。MACD 線在訊號線的有利一側、柱子又在變長,代表你要的方向力道在增強。",
    "斐波那契回撤區": "價格走完一大段後,常會回頭走 38%~62% 再繼續,這區叫黃金回撤區,很多人會在這裡進場,所以常有支撐或壓力。",
    "量能放大": "成交量比平常大,代表有真金白銀在推,這根 K 線的方向比較可信。",
    "K 線確認": "最後一根 K 線收在你要的方向(做多收漲、做空收跌),代表當下買賣雙方真的朝你這邊走。",
}

TIPS = [
    "停損是「認錯價」,不是失敗。小虧是交易的成本,不讓小虧變大虧才是重點。",
    "勝率高不等於賺錢,要看「賺的時候賺多少、虧的時候虧多少」。",
    "上一單虧了,不要為了翻本把下一單放大,這是新手最常見的爆倉原因。",
    "順勢交易:趨勢是你的朋友。盤整、均線糾纏時,空手觀望也是一種交易。",
    "槓桿放大賺也放大賠。新手先用低倍數練習,看懂停損距離與爆倉距離再加。",
    "先落袋一部分獲利,心理壓力會小很多,也比較不會把賺到的又吐回去。",
    "每次開單前先問自己:停損在哪?虧了會虧多少?答不出來就不要開。",
    "單一指標會騙人,多個條件同時指向同一方向,勝算才比較高。",
]


def lesson_open_text(side, trend, hits, misses, price, stop, liq, lev, margin):
    long_ = side == "long"
    nm = "做多(賭會漲)" if long_ else "做空(賭會跌)"
    dist = abs(price - stop) / price * 100
    liq_dist = abs(price - liq) / price * 100
    loss = margin * dist / 100 * lev
    L = ["📚 【小教室】這一單為什麼開?", "",
         f"① 先看大方向:{trend}。EMA20(短期平均價)在 EMA50(中期平均價){'之上' if long_ else '之下'},"
         f"代表價格最近在{'走高' if long_ else '走低'},所以只考慮{nm}。順著趨勢開單,比逆勢容易贏。", "",
         f"② 再用 6 項確認來加分,這單符合 {len(hits)} 項:"]
    L += [f"✔ {h}:{EXPLAIN.get(h, '')}" for h in hits]
    if misses:
        L.append("✘ 沒符合:" + "、".join(misses))
    L += ["", "③ 開單前先想「會虧多少」:",
          f"停損離進場 {dist:.2f}%,{lev}x 槓桿下,碰到停損大約虧掉這單保證金的 {dist * lev:.0f}%(約 {loss:.2f} USDT)。"
          f"爆倉價離進場只有 {liq_dist:.2f}%" + ("——比停損還近,會先被爆倉!" if liq_dist < dist else "。"),
          "", "👉 新手重點:先決定停損,再決定倉位大小;不要先想賺多少。"]
    return "\n".join(L)


def close_what(p, why):
    lev = p.get("lev", 1)
    if why == "爆倉":
        liq = abs(p["entry"] - p.get("liq", p["entry"])) / p["entry"] * 100
        stp = abs(p["entry"] - p.get("stop0", p["stop"])) / p["entry"] * 100
        return (f"{lev} 倍槓桿的爆倉距離只有約 {liq:.2f}%,停損距離是 {stp:.2f}%。價格逆行一點點就先被清掉,"
                "保證金歸零。這是高槓桿最大的風險:停損還沒用到,就先被強制平倉了。")
    if why == "保本出場":
        return "TP1 先落袋後,價格回頭碰到保本停損,剩下的倉位沒賺沒賠,整筆仍小賺。這就是「先落袋、再移動停損」的好處。"
    if why == "移動停損出場":
        return "前面已經落袋 TP1、TP2,價格回頭後被移動停損收掉剩下的倉位,大部分獲利被留住了。"
    if why.startswith("觸及停利"):
        return "價格一路走到最遠的停利目標,趨勢延續得很漂亮。順勢加上分批停利,這是理想的結果。"
    return ("價格往反方向走到你事先設好的認錯價,系統自動出場。這筆虧損在計畫內,"
            "停損的用途就是不讓小虧變成大虧。")


def lesson_close_text(p, pnl, why, hit, s):
    what = close_what(p, why)
    total = s["wins"] + s["losses"]
    L = [f"📚 【複盤小教室】這單{'賺' if pnl > 0 else '虧'} {pnl:+.2f} USDT", "",
         f"發生什麼事:{what}"]
    if p.get("hits") or p.get("score"):
        L.append(f"開單時符合 {p.get('score', '?')}/{MAX_SCORE} 項確認。結果好壞不代表當初判斷對錯,"
                 "要累積很多筆才看得出這套條件有沒有優勢。")
    L += ["", "怎麼自己複習:打開下面的圖,看進場前價格是不是順著 EMA20/EMA50 走,"
              "以及價格是怎麼碰到停損或停利的。",
          "", f"💡 今日重點:{TIPS[(total - 1) % len(TIPS)]}"]
    return "\n".join(L)


def make_chart(p, exit_price, pnl, why, path="trade_chart.png"):
    """畫複盤圖 (K 線 + EMA + 進場/停損/停利/出場)。圖上用英文標示避免字型問題。"""
    df = _CTX.get("df")
    if df is None:
        return None
    try:
        try:
            import matplotlib
        except Exception:
            _pip("matplotlib")
            import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("沒有 matplotlib,略過複盤圖:", e)
        return None
    before = df.index[df["t"] <= p["ts"]]
    ent = int(before.max()) if len(before) else 0
    a = max(0, ent - 40)
    d = df.iloc[a:].reset_index(drop=True)
    ent -= a
    ex = len(d) - 1
    for k in range(ent, len(d)):
        if d.loc[k, "low"] <= exit_price <= d.loc[k, "high"]:
            ex = k
            break
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    for k, r in d.iterrows():
        col = "#d62728" if r["close"] >= r["open"] else "#2ca02c"
        ax.plot([k, k], [r["low"], r["high"]], color=col, lw=0.8)
        ax.add_patch(plt.Rectangle((k - 0.35, min(r["open"], r["close"])), 0.7,
                                   max(abs(r["close"] - r["open"]), 1e-9), color=col))
    ax.plot(d["ema20"], lw=1, color="#1f77b4", label="EMA20")
    ax.plot(d["ema50"], lw=1, color="#ff7f0e", label="EMA50")
    levels = [("Entry", p["entry"], "black", "-"), ("SL", p["stop"], "red", "--")]
    levels += [(f"TP{k + 1}", tp, "green", ":") for k, tp in enumerate(p.get("tps", [p.get("target", p["entry"])]))]
    for name, y, c, ls in levels:
        ax.axhline(y, color=c, ls=ls, lw=0.9)
        ax.text(len(d) - 0.5, y, f"{name} {y:,.0f} ", color=c, fontsize=7, va="bottom", ha="right")
    side = p.get("side", "long")
    ax.scatter([ent], [p["entry"]], marker="^" if side == "long" else "v", s=60, color="black", zorder=5)
    ax.scatter([ex], [exit_price], marker="X", s=60, color="purple", zorder=5)
    ys = [d["low"].min(), d["high"].max(), p["stop"]] + [y for _, y, _, _ in levels]
    pad = (max(ys) - min(ys)) * 0.05
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xticks([])
    ax.set_title(f"BTC 15m  {side.upper()} {p.get('lev', 1)}x   PnL {pnl:+.2f} USDT  ({'WIN' if pnl > 0 else 'LOSS'})",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def send_photo(path, caption=""):
    if not (path and TG_TOKEN and TG_CHAT_ID):
        return
    try:
        with open(path, "rb") as f:
            r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                              data={"chat_id": TG_CHAT_ID, "caption": caption[:1000]},
                              files={"photo": f}, timeout=20)
        if r.status_code != 200:
            print("圖片發送失敗:", r.status_code, r.text)
    except Exception as e:
        print("圖片發送失敗:", e)


def open_position(s, ev, price):
    side = ev["side"]
    stop, tps, fracs, liq = trade_levels(side, price, ev["atr"], LEVERAGE)
    margin, notional, qty = size_position(s["cash"], price, LEVERAGE, MARGIN_PCT)
    s["cash"] -= margin + notional * FEE
    reason = why_text(ev)
    s["position"] = {"side": side, "entry": price, "qty": qty, "rem": qty, "margin": margin,
                     "lev": LEVERAGE, "stop": stop, "stop0": stop, "tps": tps, "fracs": fracs, "target": tps[-1],
                     "tp_hit": 0, "realized": 0.0, "liq": liq, "score": ev["score"],
                     "hits": ev["hits"], "misses": ev["misses"], "trend": ev["trend"],
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
    try:
        notify(lesson_open_text(side, ev["trend"], ev["hits"], ev["misses"], price, stop, liq, LEVERAGE, margin))
    except Exception as e:
        print("小教室失敗:", e)


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
    try:
        notify(lesson_close_text(p, pnl, why, hit, s))
        chart = make_chart(p, price, pnl, why)
        send_photo(chart, "📈 複盤圖:Entry=進場 SL=停損 TP=停利 X=出場;紅K=收漲、綠K=收跌;細線是 EMA20/EMA50")
        send_lesson_video(trade_info(p, price, pnl, why, hit, s, chart))
    except Exception as e:
        print("小教室失敗:", e)


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
    """讀取 Telegram 訊息。/status 回報狀態;/demo 製作示範影片 (下次執行時處理)。只回應自己的 chat id。"""
    if not (TG_TOKEN and TG_CHAT_ID):
        return
    try:
        cmds = set()
        for u in get_tg_updates(s.get("tg_offset", 0)):
            s["tg_offset"] = u["update_id"] + 1
            m = u.get("message") or {}
            if str(m.get("chat", {}).get("id")) != str(TG_CHAT_ID):
                continue
            txt = str(m.get("text", "")).strip().lower()
            if txt.startswith("/status"):
                cmds.add("status")
            elif txt.startswith("/demo"):
                cmds.add("demo")
        if "demo" in cmds:
            s["demo_video"] = True
        save_state(s)
        if "status" in cmds:
            notify(status_text(s, price))
        if "demo" in cmds:
            notify("🎬 收到 /demo,示範影片製作中,約 1~3 分鐘後送到。")
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


# ---------- AI 虛擬老師複盤影片 ----------
LESSON_VIDEO = True     # 平倉後自動製作老師複盤影片 (首次會自動安裝 edge-tts / ffmpeg / 中文字型,約多 1~2 分鐘)
VIDEO_W, VIDEO_H, VIDEO_FPS = 1280, 720, 12
VOICES = {"A": "zh-TW-HsiaoChenNeural", "B": "zh-TW-HsiaoYuNeural"}
TEACHER_NAMES = {"A": "主講老師", "B": "助教老師"}
_TOOLS = {}
_VIDEO_ERR = [""]


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], capture_output=True)


def find_cjk_font():
    try:
        out = subprocess.run(["fc-list", ":lang=zh-tw", "file"], capture_output=True, text=True).stdout
        files = [l.split(":")[0].strip() for l in out.splitlines() if l.strip()]
        files = [f for f in files if os.path.exists(f)]
        reg = [f for f in files if "Bold" not in f and "Black" not in f]
        sans = [f for f in reg if "Sans" in f or "Hei" in f]
        return (sans or reg or files or [None])[0]
    except Exception:
        return None


def ensure_video_tools():
    """找 ffmpeg 與中文字型,缺的話自動安裝 (只在要做影片時才做)。"""
    if _TOOLS:
        return _TOOLS
    for mod, pkg in (("PIL", "pillow"), ("numpy", "numpy")):
        try:
            __import__(mod)
        except Exception:
            _pip(pkg)
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
        except Exception:
            _pip("imageio-ffmpeg")
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            print("找不到 ffmpeg:", e)
    font = find_cjk_font()
    if not font and shutil.which("sudo"):
        subprocess.run("sudo -n apt-get update -qq && sudo -n apt-get install -y -qq fonts-noto-cjk",
                       shell=True, capture_output=True)
        font = find_cjk_font()
    try:
        import edge_tts  # noqa: F401
    except Exception:
        _pip("edge-tts")
    try:
        import gtts  # noqa: F401
    except Exception:
        _pip("gTTS")
    _TOOLS.update({"ffmpeg": ff, "font": font})
    return _TOOLS


def load_font(path, size):
    from PIL import ImageFont
    for idx in (3, 0):  # NotoSansCJK 的 .ttc:3 = 繁體中文字形
        try:
            return ImageFont.truetype(path, size, index=idx)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap(text, font, maxw):
    lines, cur = [], ""
    for ch in text:
        if cur and font.getlength(cur + ch) > maxw:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def split_sentences(text, maxlen=34):
    parts = []
    for seg in re.split(r"(?<=[。!?!?;;])", text):
        seg = seg.strip()
        while len(seg) > maxlen:
            cut = max(seg.rfind(c, 0, maxlen) for c in (",", ",", "、"))
            if cut < 8:
                cut = maxlen - 1
            parts.append(seg[:cut + 1])
            seg = seg[cut + 1:].strip()
        if seg:
            parts.append(seg)
    return parts


def teacher_path(key):
    """找老師圖片:優先 teacher_a / teacher_b;找不到就用倉庫根目錄的圖片檔 (依檔名排序,第 1 張=主講、第 2 張=助教)。"""
    for d in (".", "teachers"):
        for ext in ("jpg", "jpeg", "png", "webp"):
            path = os.path.join(d, f"teacher_{key.lower()}.{ext}")
            if os.path.exists(path):
                return path
    imgs = sorted(f for f in os.listdir(".")
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                  and not f.lower().startswith(("trade_chart", "teacher_", "lesson")))
    idx = 0 if key == "A" else 1
    return imgs[idx] if len(imgs) > idx else None


def load_teacher(key, size):
    """讀取老師圖片 (cover 裁切,偏向保留上半部的臉);沒有圖片就畫簡單的預設頭像。"""
    from PIL import Image, ImageDraw
    w, h = size
    path = teacher_path(key)
    if path:
        try:
            im = Image.open(path).convert("RGB")
            sc = max(w / im.width, h / im.height)
            im = im.resize((int(im.width * sc) + 1, int(im.height * sc) + 1), Image.LANCZOS)
            x, y = (im.width - w) // 2, int((im.height - h) * 0.2)
            return im.crop((x, y, x + w, y + h))
        except Exception as e:
            print("老師圖片讀取失敗:", path, e)
    im = Image.new("RGB", size, "#3A4A66")
    dr = ImageDraw.Draw(im)
    dr.ellipse((w * 0.3, h * 0.16, w * 0.7, h * 0.52), fill="#C9D3E6")
    dr.ellipse((w * 0.1, h * 0.58, w * 0.9, h * 1.25), fill="#C9D3E6")
    return im


def tts_line(text, voice, out_path):
    """文字轉語音。edge-tts 失敗就用 gTTS,都失敗回傳 None (影片會改成無聲、只有字幕)。"""
    try:
        import asyncio
        import edge_tts
        asyncio.run(edge_tts.Communicate(text, voice).save(out_path))
        if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            return out_path
    except Exception as e:
        print("edge-tts 失敗:", e)
    try:
        from gtts import gTTS
        gTTS(text, lang="zh-TW").save(out_path)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
            return out_path
    except Exception as e:
        print("gTTS 失敗:", e)
    return None


def load_audio(path, ffmpeg, sr):
    import numpy as np
    wav = path + ".wav"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(sr), wav], check=True)
    with wave.open(wav) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def trade_info(p, price, pnl, why, hit, s, chart=None, demo=False):
    lev = p.get("lev", 1)
    dist = abs(p["entry"] - p.get("stop0", p["stop"])) / p["entry"] * 100
    liq_dist = abs(p["entry"] - p["liq"]) / p["entry"] * 100 if p.get("liq") else 100.0
    total = max(s["wins"] + s["losses"], 1)
    return {"side": p.get("side", "long"), "lev": lev, "entry": p["entry"], "exit": price, "pnl": pnl,
            "why": why, "hit": hit, "score": p.get("score", "?"), "trend": p.get("trend", "順勢"),
            "hits": p.get("hits", []), "misses": p.get("misses", []), "dist": dist, "liq_dist": liq_dist,
            "loss_pct": dist * lev, "what": close_what(p, why), "tip": TIPS[(total - 1) % len(TIPS)],
            "chart": chart, "demo": demo}


def build_script(t):
    nm = "做多" if t["side"] == "long" else "做空"
    res = "賺" if t["pnl"] > 0 else "虧"
    L = []

    def add(who, slide, text):
        L.extend((who, slide, part) for part in split_sentences(text))

    add("A", 0, "大家好,我是 AI 虛擬老師。" + ("這是一支示範影片,內容不是真實交易。" if t.get("demo") else "我們來複盤剛剛這一單。"))
    add("B", 0, f"這一單是{nm},{t['lev']} 倍槓桿,結果{res}了 {abs(t['pnl']):.2f} 個 USDT。")
    add("A", 1, "先看這張圖。三角形是進場,叉叉是出場,紅色虛線是停損,綠色虛線是停利。")
    add("B", 2, "那為什麼會開這一單呢?")
    hits = "、".join(t["hits"][:3]) or "沒有額外確認"
    add("A", 2, f"因為大方向是{t['trend']},六項確認裡符合了 {t['score']} 項,像是{hits}。")
    add("B", 3, "那風險有多大呢?")
    add("A", 3, f"停損離進場大約百分之 {t['dist']:.1f}。在 {t['lev']} 倍槓桿下,碰到停損,大約會虧掉保證金的百分之 {t['loss_pct']:.0f}。")
    if t["liq_dist"] < t["dist"]:
        add("B", 3, "而且爆倉價比停損還近,所以要特別小心。")
    add("A", 0, t["what"])
    add("B", 4, "今天的重點:" + t["tip"])
    add("A", 4, "記得,這是模擬練習,結果不代表未來。我們下次見!")
    return L


def render_scene(t, slide, text, who, F, chart_im):
    """畫一個鏡頭的靜態畫面 (不含會動的老師與音波)。"""
    from PIL import Image, ImageDraw
    W, H = VIDEO_W, VIDEO_H
    im = Image.new("RGB", (W, H), "#0F1724")
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((30, 28, 190, 64), 18, fill="#3D5AFE")
    d.text((110, 46), "AI 虛擬老師", font=F[26], fill="#FFFFFF", anchor="mm")
    for key, x in (("A", 30), ("B", 290)):
        d.text((x + 115, 392), TEACHER_NAMES[key], font=F[30], fill="#C9D3E6", anchor="mm")
    px, py, pw, ph = 560, 40, 680, 520
    d.rounded_rectangle((px, py, px + pw, py + ph), 18, fill="#1B2536")

    def title(s_):
        d.text((px + 30, py + 24), s_, font=F[44], fill="#FFD166")

    def row(y, s_, fill="#FFFFFF", f=34):
        d.text((px + 30, y), s_, font=F[f], fill=fill)

    pnl = t["pnl"]
    nm = "做多" if t["side"] == "long" else "做空"
    if slide == 0:
        title("複盤小教室" + ("(示範)" if t.get("demo") else ""))
        d.text((px + 30, py + 100), f"{'賺' if pnl > 0 else '虧'} {pnl:+.2f} USDT", font=F[64],
               fill="#FFD166" if pnl > 0 else "#9AA7BD")
        row(py + 210, f"{nm}  {t['lev']} 倍槓桿")
        row(py + 262, f"進場 {t['entry']:,.0f}  →  出場 {t['exit']:,.0f}")
        row(py + 314, f"結束原因:{t['why']}")
        if t["hit"]:
            row(py + 366, f"已先落袋 TP1" + (f"~TP{t['hit']}" if t["hit"] > 1 else ""), "#4ADE80")
    elif slide == 1:
        title("複盤圖")
        if chart_im is not None:
            c = chart_im.copy()
            c.thumbnail((pw - 40, ph - 100))
            im.paste(c, (px + (pw - c.width) // 2, py + 90 + (ph - 100 - c.height) // 2))
        else:
            row(py + 200, "(這單沒有複盤圖)", "#9AA7BD")
    elif slide == 2:
        title("為什麼開這單?")
        row(py + 92, f"大方向:{t['trend']}", "#FFFFFF", 32)
        names = list(t["hits"]) + list(t["misses"])
        for k, nme in enumerate(names[:6]):
            ok = k < len(t["hits"])
            row(py + 150 + k * 52, ("●  " if ok else "○  ") + nme, "#4ADE80" if ok else "#7C8AA5", 32)
    elif slide == 3:
        title("風險有多大?")
        mx = max(t["dist"], t["liq_dist"], 0.01) * 1.15
        for k, (lab, val, col) in enumerate((("停損距離", t["dist"], "#FFD166"), ("爆倉距離", t["liq_dist"], "#FF6B6B"))):
            y = py + 110 + k * 110
            row(y, f"{lab}  {val:.2f}%", "#FFFFFF", 32)
            d.rounded_rectangle((px + 30, y + 56, px + 30 + max(8, int(560 * val / mx)), y + 84), 8, fill=col)
        row(py + 350, f"碰到停損約虧保證金 {t['loss_pct']:.0f}%", "#FFFFFF", 34)
        if t["liq_dist"] < t["dist"]:
            row(py + 410, "爆倉價比停損還近,會先爆倉!", "#FF6B6B", 32)
    else:
        title("今日重點")
        for k, ln in enumerate(wrap(t["tip"], F[38], pw - 60)[:7]):
            row(py + 110 + k * 62, ln, "#FFFFFF", 38)
    im = im.convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.rectangle((0, 590, W, H), fill=(0, 0, 0, 185))
    im = Image.alpha_composite(im, ov).convert("RGB")
    d = ImageDraw.Draw(im)
    lines = wrap(text, F[34], W - 140)[:3]
    y0 = 590 + (H - 590 - len(lines) * 44) // 2
    for k, ln in enumerate(lines):
        d.text((W // 2, y0 + k * 44 + 22), ln, font=F[34], fill="#FFFFFF", anchor="mm")
    return im


def make_lesson_video(t, out="lesson.mp4"):
    """做出一支老師複盤影片 (MP4)。失敗回傳 None。"""
    tools = ensure_video_tools()
    if not tools.get("ffmpeg") or not tools.get("font"):
        _VIDEO_ERR[0] = f"缺少影片工具:ffmpeg {'有' if tools.get('ffmpeg') else '沒有'}、中文字型 {'有' if tools.get('font') else '沒有'}"
        print(_VIDEO_ERR[0])
        return None
    import math
    import numpy as np
    from PIL import Image, ImageDraw, ImageEnhance
    F = {sz: load_font(tools["font"], sz) for sz in (26, 30, 32, 34, 38, 44, 64)}
    chart_im = Image.open(t["chart"]).convert("RGB") if t.get("chart") and os.path.exists(t["chart"]) else None
    tmp = tempfile.mkdtemp()
    sr = 24000
    gap = np.zeros(int(0.25 * sr), dtype=np.int16)
    parts, spans, cursor = [], [], 0.0
    for k, (who, slide, text) in enumerate(build_script(t)):
        mp3 = tts_line(text, VOICES[who], os.path.join(tmp, f"l{k}.mp3"))
        audio = None
        if mp3:
            try:
                audio = load_audio(mp3, tools["ffmpeg"], sr)
            except Exception as e:
                print("音檔轉換失敗:", e)
        syn = audio is None or len(audio) < sr * 0.2
        if syn:  # 沒有語音:依字數估算停留時間,畫面仍有字幕
            audio = np.zeros(int(max(1.8, 0.30 * len(text)) * sr), dtype=np.int16)
        spans.append((cursor, cursor + len(audio) / sr, who, slide, text, syn))
        parts += [audio, gap]
        cursor += len(audio) / sr + 0.25
    parts.append(np.zeros(int(0.8 * sr), dtype=np.int16))
    pcm = np.concatenate(parts)
    wav_path = os.path.join(tmp, "all.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    nframes = int((len(pcm) / sr) * VIDEO_FPS) + 1
    win = int(sr / VIDEO_FPS)
    raw = np.array([np.sqrt(np.mean(pcm[i * win:(i + 1) * win].astype(np.float64) ** 2) + 1e-9)
                    if i * win < len(pcm) else 0.0 for i in range(nframes)])
    voiced = raw[raw > 50]
    ref = np.percentile(voiced, 90) if len(voiced) else 1.0
    env = np.clip(raw / max(ref, 1.0), 0, 1)
    scenes = {}
    cw, ch = 230, 276
    port = {k: load_teacher(k, (cw, ch)) for k in "AB"}
    dim = {k: ImageEnhance.Brightness(im).enhance(0.55) for k, im in port.items()}
    cmd = [tools["ffmpeg"], "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(VIDEO_FPS), "-i", "-", "-i", wav_path,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    si = 0
    try:
        for f in range(nframes):
            tm = f / VIDEO_FPS
            while si < len(spans) - 1 and tm >= spans[si][1] + 0.25:
                si += 1
            st, en, who, slide, text, syn = spans[si]
            speaking = st <= tm < en
            e = float(env[f]) if not syn else (0.55 + 0.45 * abs(math.sin(f * 1.9)) if speaking else 0.0)
            if not speaking and not syn:
                e = 0.0
            if si not in scenes:
                scenes[si] = render_scene(t, slide, text, who, F, chart_im)
            frame = scenes[si].copy()
            dr = ImageDraw.Draw(frame)
            for key, x in (("A", 30), ("B", 290)):
                active = (key == who)
                img = port[key] if (active or not speaking) else dim[key]
                yo = -int(e * 9) if (active and speaking) else 0
                frame.paste(img, (x, 100 + yo))
                if active and speaking:
                    wdt = 3 + int(e * 6)
                    dr.rectangle((x - wdt, 100 + yo - wdt, x + cw + wdt, 100 + yo + ch + wdt), outline="#FFD166", width=wdt)
            for i in range(20):
                hh = int(e * (0.3 + 0.7 * abs(math.sin(f * 0.8 + i * 0.9))) * 110) + 4
                dr.rounded_rectangle((30 + i * 25, 560 - hh, 30 + i * 25 + 16, 560), 4, fill="#FFD166" if speaking else "#33415C")
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait(timeout=300)
    except Exception as ex:
        _VIDEO_ERR[0] = f"影片合成失敗:{ex}"
        print(_VIDEO_ERR[0], proc.stderr.read().decode(errors="ignore")[-500:] if proc.stderr else "")
        return None
    if proc.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) < 1000:
        _VIDEO_ERR[0] = "ffmpeg 轉檔失敗:" + (proc.stderr.read().decode(errors="ignore")[-200:] if proc.stderr else "")
        print(_VIDEO_ERR[0])
        return None
    return out


def send_video(path, caption=""):
    if not (path and TG_TOKEN and TG_CHAT_ID):
        return
    try:
        with open(path, "rb") as f:
            r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendVideo",
                              data={"chat_id": TG_CHAT_ID, "caption": caption[:1000], "supports_streaming": "true",
                                    "width": VIDEO_W, "height": VIDEO_H},
                              files={"video": f}, timeout=180)
        if r.status_code != 200:
            print("影片發送失敗:", r.status_code, r.text)
            notify(f"⚠️ 影片做好了,但 Telegram 拒收:{r.status_code} {r.text[:150]}")
    except Exception as e:
        print("影片發送失敗:", e)
        notify(f"⚠️ 影片做好了,但傳送失敗:{type(e).__name__}: {str(e)[:150]}")


def send_lesson_video(t):
    if not (LESSON_VIDEO and TG_TOKEN and TG_CHAT_ID):
        return
    try:
        path = make_lesson_video(t)
        if path:
            send_video(path, "🎬 AI 虛擬老師複盤影片(模擬練習,非投資建議)")
            os.remove(path)
        else:
            notify("⚠️ 這次沒有產生老師影片。原因:" + (_VIDEO_ERR[0] or "未知,請看 Actions 的執行紀錄"))
    except Exception as e:
        print("老師影片失敗:", e)
        notify(f"⚠️ 老師影片製作失敗:{type(e).__name__}: {str(e)[:200]}")


def send_demo_video(df, price):
    """/demo:用最近的行情做一支示範影片,讓你先看看效果。"""
    c = df.iloc[-1]
    entry = float(df.iloc[-30]["close"])
    atr = float(c["atr"])
    stop = entry - ATR_MULT * atr
    tps = [entry + k * ATR_MULT * atr for k in TP_R]
    p = {"side": "long", "lev": LEVERAGE, "entry": entry, "stop": stop, "stop0": stop, "tps": tps,
         "liq": entry * (1 - (1 / LEVERAGE - MMR)), "score": 5, "trend": "多頭",
         "hits": ["EMA200 順勢", "MACD 同向", "斐波那契回撤區", "K 線確認", "量能放大"], "misses": ["RSI 回檔轉強"],
         "ts": int(df.iloc[-30]["t"]), "time": now(), "reason": "示範", "tp_hit": 3}
    exit_price = tps[-1]
    pnl = 3.21
    chart = make_chart(p, exit_price, pnl, "觸及停利 TP3")
    t = trade_info(p, exit_price, pnl, "觸及停利 TP3", 2, {"wins": 3, "losses": 1}, chart, demo=True)
    send_lesson_video(t)


# ---------- 主流程 ----------
def run_once():
    s = load_state()
    w = win(s)
    df = add_indicators(get_klines())
    _CTX["df"] = df
    price = df.iloc[-1]["close"]
    if s["position"]:
        check_exit(s, df)
    handle_commands(s, price)
    if s.pop("demo_video", None):
        save_state(s)
        send_demo_video(df, price)
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

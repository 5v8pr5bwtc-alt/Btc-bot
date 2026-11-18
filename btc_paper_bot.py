#!/usr/bin/env python3
"""
BTC 虛擬資金自動交易機器人 (Paper Trading) - 雲端單次執行版
由 GitHub Actions 每 5 分鐘執行一次:檢查停損停利 -> 判斷進場 -> Telegram 通知 -> 存檔
- 虛擬本金 100 USDT,多空雙向 (做多/做空,1 倍無槓桿),不會動用真錢
- 做多:EMA20>EMA50 且 RSI 回升穿越 45;做空:EMA20<EMA50 且 RSI 回落跌破 55
- 停損 1.5 倍 ATR,風報比 1:2;15 分鐘 K,24 小時運作 (加密貨幣全年無休)
- Telegram 傳 /status 可查目前持倉與勝率(下次排程執行時回覆);持倉每 3 小時自動回報浮動損益
- 手機/電腦本機常駐也可用:python btc_paper_bot.py --loop
- 測試通知:python btc_paper_bot.py --test
"""
import os, sys, time, csv, json
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd

START_CASH = 100.0
RISK_PER_TRADE = 0.02
FEE = 0.001
RR = 2.0
ATR_MULT = 1.5
STATE_FILE = "state.json"
TRADE_LOG = "trades.csv"
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "")
TZ = timezone(timedelta(hours=8))  # 台灣時間


def now():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M")


def notify(msg):
    print(f"[{now()}] {msg}\n")
    if TG_TOKEN and TG_CHAT_ID:
        try:
            r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                              data={"chat_id": TG_CHAT_ID, "text": msg}, timeout=10)
            if r.status_code != 200:
                print("Telegram 失敗:", r.status_code, r.text)
        except Exception as e:
            print("Telegram 失敗:", e)


# ---------- 資料 (Binance 公開行情,失敗時改用 Kraken) ----------
def get_klines():
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/klines",
                         params={"symbol": "BTCUSDT", "interval": "15m", "limit": 200}, timeout=10)
        r.raise_for_status()
        df = pd.DataFrame(r.json()).iloc[:, :6]
        df.columns = ["t", "open", "high", "low", "close", "vol"]
    except Exception as e:
        print("Binance 失敗,改用 Kraken:", e)
        r = requests.get("https://api.kraken.com/0/public/OHLC",
                         params={"pair": "XBTUSDT", "interval": 15}, timeout=10)
        r.raise_for_status()
        res = r.json()["result"]
        rows = next(v for k, v in res.items() if k != "last")
        df = pd.DataFrame(rows).iloc[:, :7]
        df.columns = ["t", "open", "high", "low", "close", "vwap", "vol"]
        df["t"] = df["t"].astype("int64") * 1000
        df = df.tail(200).reset_index(drop=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df["t"] = df["t"].astype("int64")
    return df


def add_indicators(df):
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    d = df["close"].diff()
    g = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + g / l)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


# ---------- 狀態 ----------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"cash": START_CASH, "position": None, "wins": 0, "losses": 0, "last_bar": 0}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)


def log_trade(row):
    new = not os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)


# ---------- 策略 (多空雙向) ----------
def check_signal(df):
    """回傳 (方向, 依據, ATR)。方向: 'long' / 'short' / None。只用已收盤 K 線。"""
    cur, prev = df.iloc[-2], df.iloc[-3]
    if cur["ema20"] > cur["ema50"] and cur["close"] > cur["ema50"] and prev["rsi"] < 45 <= cur["rsi"]:
        return "long", (f"上升趨勢(EMA20 {cur['ema20']:.0f} > EMA50 {cur['ema50']:.0f}),"
                        f"RSI 由 {prev['rsi']:.1f} 回升至 {cur['rsi']:.1f},回檔結束轉強"), cur["atr"]
    if cur["ema20"] < cur["ema50"] and cur["close"] < cur["ema50"] and prev["rsi"] > 55 >= cur["rsi"]:
        return "short", (f"下降趨勢(EMA20 {cur['ema20']:.0f} < EMA50 {cur['ema50']:.0f}),"
                         f"RSI 由 {prev['rsi']:.1f} 回落至 {cur['rsi']:.1f},反彈結束轉弱"), cur["atr"]
    return None, "", 0


def open_position(s, side, price, atr, reason):
    d = 1 if side == "long" else -1
    stop = price - d * ATR_MULT * atr
    target = price + d * RR * ATR_MULT * atr
    qty = min(s["cash"] * RISK_PER_TRADE / abs(price - stop), s["cash"] * 0.95 / price)
    s["cash"] -= qty * price * (1 + FEE)  # 1 倍、無槓桿:鎖定等值保證金 + 手續費
    s["position"] = {"side": side, "entry": price, "qty": qty, "stop": stop, "target": target,
                     "time": now(), "ts": int(time.time() * 1000), "reason": reason}
    s["last_report"] = time.time()
    save_state(s)
    icon, name = ("🟢", "做多") if side == "long" else ("🔴", "做空")
    notify(f"{icon} 【虛擬開倉】BTC {name}\n進場價:{price:,.2f}\n停損:{stop:,.2f}\n停利:{target:,.2f}\n"
           f"數量:{qty:.6f} BTC\n依據:{reason}")


def close_position(s, price, why):
    p = s["position"]
    d = 1 if p.get("side", "long") == "long" else -1
    gross = (price - p["entry"]) * p["qty"] * d
    fees = p["qty"] * p["entry"] * FEE + p["qty"] * price * FEE
    pnl = gross - fees
    s["cash"] += p["qty"] * p["entry"] + gross - p["qty"] * price * FEE
    s["wins" if pnl > 0 else "losses"] += 1
    s["position"] = None
    save_state(s)
    total = s["wins"] + s["losses"]
    name = "做多" if d == 1 else "做空"
    log_trade({"open_time": p["time"], "close_time": now(), "side": name, "entry": round(p["entry"], 2),
               "exit": round(price, 2), "pnl_usdt": round(pnl, 4),
               "result": "WIN" if pnl > 0 else "LOSS", "reason": p["reason"], "close_why": why})
    notify(f"{'✅ 賺錢' if pnl > 0 else '❌ 虧錢'} 【虛擬平倉】{name}({why})\n"
           f"進場:{p['entry']:,.2f} → 出場:{price:,.2f}\n本筆損益:{pnl:+.2f} USDT\n"
           f"目前資金:{s['cash']:.2f} USDT(本金 {START_CASH:.0f})\n"
           f"累計:{s['wins']} 勝 / {s['losses']} 敗,勝率 {s['wins'] / total * 100:.1f}%")


def check_exit(s, df):
    """用開倉後 K 線的最高/最低價判斷;同一根都觸及時保守算停損。"""
    p = s["position"]
    long_ = p.get("side", "long") == "long"
    for _, row in df[df["t"] >= p["ts"]].iterrows():
        hit_stop = row["low"] <= p["stop"] if long_ else row["high"] >= p["stop"]
        hit_tp = row["high"] >= p["target"] if long_ else row["low"] <= p["target"]
        if hit_stop:
            return close_position(s, p["stop"], "觸及停損")
        if hit_tp:
            return close_position(s, p["target"], "觸及停利")
    price = df.iloc[-1]["close"]
    if (long_ and price <= p["stop"]) or (not long_ and price >= p["stop"]):
        close_position(s, price, "觸及停損")
    elif (long_ and price >= p["target"]) or (not long_ and price <= p["target"]):
        close_position(s, price, "觸及停利")


REPORT_HOURS = 3  # 持倉期間每幾小時回報一次浮動盈虧


def floating_pnl(p, price):
    d = 1 if p.get("side", "long") == "long" else -1
    gross = (price - p["entry"]) * p["qty"] * d
    fees = p["qty"] * p["entry"] * FEE + p["qty"] * price * FEE
    return gross - fees


def position_text(s, price):
    p = s["position"]
    name = "做多" if p.get("side", "long") == "long" else "做空"
    pnl = floating_pnl(p, price)
    return (f"持倉:BTC {name}(開倉 {p['time']})\n進場:{p['entry']:,.2f} 現價:{price:,.2f}\n"
            f"停損:{p['stop']:,.2f} 停利:{p['target']:,.2f}\n"
            f"浮動損益:{pnl:+.2f} USDT {'📈' if pnl > 0 else '📉'}")


def status_text(s, price):
    total = s["wins"] + s["losses"]
    wr = f"{s['wins'] / total * 100:.1f}%" if total else "尚無完成交易"
    equity = s["cash"]
    if s["position"]:
        p = s["position"]
        equity += p["qty"] * p["entry"] + floating_pnl(p, price) + p["qty"] * p["entry"] * FEE
    body = position_text(s, price) if s["position"] else "目前空手,等待訊號"
    return (f"📊 【目前狀態】BTC 現價 {price:,.2f}\n{body}\n"
            f"資產:約 {equity:.2f} USDT(本金 {START_CASH:.0f},{equity - START_CASH:+.2f})\n"
            f"累計:{s['wins']} 勝 / {s['losses']} 敗,勝率 {wr}")


def get_tg_updates(offset):
    r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
                     params={"offset": offset, "timeout": 0}, timeout=10)
    return r.json().get("result", []) if r.status_code == 200 else []


def handle_commands(s, price):
    """讀取 Telegram 訊息,收到 /status 就回報。只回應自己的 chat id。"""
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


def run_once():
    s = load_state()
    df = add_indicators(get_klines())
    price = df.iloc[-1]["close"]
    if s["position"]:
        check_exit(s, df)
    handle_commands(s, price)
    if s["position"]:
        maybe_report(s, price)
    if not s["position"]:
        bar = int(df.iloc[-2]["t"])
        if bar != s.get("last_bar"):
            s["last_bar"] = bar
            save_state(s)
            side, reason, atr = check_signal(df)
            if side:
                open_position(s, side, df.iloc[-1]["close"], atr, reason)


if __name__ == "__main__":
    if "--test" in sys.argv:
        notify("✅ 測試成功!BTC 虛擬交易機器人已連上 Telegram,之後每次下單都會通知你。")
    elif "--loop" in sys.argv:
        while True:
            try:
                run_once()
            except Exception as e:
                print("錯誤:", e)
            time.sleep(60)
    else:
        run_once()

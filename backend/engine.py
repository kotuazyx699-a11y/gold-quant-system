import os
import re
import sys
import json
import time
import subprocess
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from openai import OpenAI

# ================= LOAD LOCAL .ENV IF PRESENT =================
def load_local_env():
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass

load_local_env()

# ================= CONFIGURATION =================
THAILLM_API_KEY = os.getenv("THAILLM_API_KEY", "").strip()
NTFY_TOPIC = "pansakorn888"
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
THAI_TZ = timezone(timedelta(hours=7))

MODEL_QUANT_AUDITOR = "pathumma-thaillm-qwen3-8b-think-3.0.0"
MODEL_MACRO_OFFICER = "typhoon-s-thaillm-8b-instruct"
MODEL_CONTRARIAN_CRITIC = "typhoon-s-thaillm-8b-instruct"
MODEL_CHIEF_STRATEGIST = "thalle-0.2-thaillm-8b-fa"

llm_client = OpenAI(
    base_url="https://thaillm.or.th/api/v1",
    api_key=THAILLM_API_KEY if THAILLM_API_KEY else "MISSING_KEY"
)

# ================= 0. KNOWLEDGE BASE & CLOSED-LOOP SELF-LEARNING =================
DEFAULT_KB_FILES = {
    "01_market_maker_liquidity.md": (
        "# Market Maker, Asian Range & Liquidity Sweep Rules\n"
        "- สภาพคล่อง (Liquidity Pools) สะสมอยู่เหนือ Previous 24H High (PDH) และใต้ Previous 24H Low (PDL)\n"
        "- London Breakout Setup: จับจังหวะราคาเบรกกรอบ Asian Session Range (07:00 - 12:45 น. เวลาไทย) ช่วงเปิดตลาดลอนดอน (13:00 - 16:30 น.) ตามเทรนด์ H1/M15\n"
        "- Liquidity Sweep Setup: เมื่อราคาพุ่งทะลุ PDH หรือ PDL เพื่อกวาด Stop Loss แต่แท่งเทียน M15 ปิดกลับเข้ามาในกรอบเดิมพร้อมไส้เทียนปฏิเสธราคา\n"
        "- RSI Exhaustion Guard: ห้ามไล่ Breakout ฝั่ง Short เมื่อ RSI14 < 28 และห้ามไล่ฝั่ง Long เมื่อ RSI14 > 72\n"
    ),
    "02_factorbench_investability.md": (
        "# Institutional R-Multiple, Daily Sharpe & Risk Rules\n"
        "- วัดผล Sharpe, Sortino และ Calmar จากผลตอบแทนรายวันจริง (Daily Calendar Aggregation, sqrt(252)) ไม่ใช้ per-trade inflation\n"
        "- Pessimistic Intrabar Execution: หากแท่งเทียน M15 เดียวกันแตะทั้ง SL และ TP ให้ถือว่าโดน SL ก่อนเสมอ\n"
        "- Walk-Forward 67% Train / 33% Out-of-Sample (OOS) พร้อมหัก Spread + Commission จริง ($0.60/oz Trend, $0.45/oz Scalp)\n"
    ),
    "03_macro_gold_regime.md": (
        "# Macro Gold (XAU/USD) & Multi-Timeframe Regime Rules\n"
        "- เมื่อ ADX(14) <= ADX Threshold ตลาดแกว่งออกข้าง ใช้กลยุทธ์ VALUE_AREA_FADE บริเวณขอบ VAH/VAL\n"
        "- เมื่อ ADX(14) > ADX Threshold ตลาดมีเทรนด์แข็งแกร่ง เน้น LONDON_BREAKOUT, MOMENTUM_BREAKOUT หรือ HTF_TREND_PULLBACK\n"
        "- ดัชนี DXY และ US10Y Yield ที่ปรับตัวขึ้นกดดันราคาทองคำ ต้องระวังความผันผวนช่วงข่าว USD สำคัญ\n"
    ),
    "04_vp_anatomy_scalp_500pt.md": (
        "# 4 กายวิภาค Volume Profile (VPAnatomies_EA) สำหรับ Scalp 500 จุด ($5.00)\n"
        "- 1. D-SHAPED (POC 34%-66%): ตลาดสมดุล ดัก BUY ที่ VAL และดัก SELL ที่ VAH เป้าหมายเข้าหา POC\n"
        "- 2. B-SHAPED (POC < 34%): โวลุ่มกองโซนล่าง เปิด SELL เมื่อกดใต้ POC หรือเปิด BUY เมื่อเบรกทะลุ VAH\n"
        "- 3. P-SHAPED (POC > 66%): โวลุ่มกองโซนบน เปิด BUY ตามแนวโน้มเมื่อย่อรับที่ POC/VAL หรือ SELL หากหลุด VAL\n"
        "- 4. DOUBLE ZONE (2nd Peak >= 60% ของ POC + มี Valley คั่น): ดัก BUY ที่ Lower Peak และดัก SELL ที่ Upper Peak\n"
    )
}

def evaluate_live_self_learning():
    """Closed-Loop Self-Learning: วิเคราะห์ผลแพ้/ชนะจริงจาก Live Journals เพื่อปรับสิทธิ์การใช้งานอัตโนมัติ"""
    os.makedirs("data", exist_ok=True)
    lesson_path = "knowledge_base/05_live_journal_lessons.md"
    weights_path = "data/learning_weights.json"

    all_closed = []
    for j_file, mode_lbl, risk_unit in [
        ("data/trade_journal.json", "TREND", 10.0),
        ("data/scalp_journal.json", "SCALP_500PT", 5.0)
    ]:
        if os.path.exists(j_file):
            try:
                with open(j_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
                for t in items:
                    if t.get("status") in ("WIN", "LOSS"):
                        tc = dict(t)
                        tc["mode"] = mode_lbl
                        tc["est_r"] = round(float(t.get("pnl", 0)) / max(abs(float(t.get("entry", 0)) - float(t.get("sl", 0))), risk_unit), 2)
                        all_closed.append(tc)
            except Exception:
                pass

    muted_trend_factors = []
    muted_scalp_shapes = []
    stats_by_key = {}

    for t in all_closed:
        key = f"{t['mode']}:{t.get('factor', 'SETUP')}"
        st = stats_by_key.setdefault(key, {"mode": t["mode"], "name": t.get("factor", "SETUP"), "w": 0, "l": 0, "pnl": 0.0, "net_r": 0.0, "recent": []})
        if t["status"] == "WIN":
            st["w"] += 1
        else:
            st["l"] += 1
        st["pnl"] = round(st["pnl"] + float(t.get("pnl", 0)), 2)
        st["net_r"] = round(st["net_r"] + float(t.get("est_r", 0)), 2)
        st["recent"].append(t["status"])

    for k, st in stats_by_key.items():
        tot = st["w"] + st["l"]
        consec_losses = 0
        for res in st["recent"]:
            if res == "LOSS":
                consec_losses += 1
            else:
                break
        st["consec_losses"] = consec_losses
        st["wr"] = round((st["w"] / tot) * 100.0, 1) if tot > 0 else 0.0

        # กฎ Self-Learning Guard: หากแพ้ติดกัน >= 2 ไม้ หรือขาดทุนสะสมเกิน -1.5R จาก >= 3 ไม้ -> สั่ง Mute ชั่วคราว
        if consec_losses >= 2 or (tot >= 3 and st["net_r"] <= -1.5):
            st["auto_muted"] = True
            if st["mode"] == "TREND":
                muted_trend_factors.append(st["name"])
            else:
                muted_scalp_shapes.append(st["name"])
        else:
            st["auto_muted"] = False

    learning_state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_closed": len(all_closed),
        "overall_wins": len([t for t in all_closed if t["status"] == "WIN"]),
        "overall_losses": len([t for t in all_closed if t["status"] == "LOSS"]),
        "overall_pnl": round(sum(float(t.get("pnl", 0)) for t in all_closed), 2),
        "muted_trend_factors": muted_trend_factors,
        "muted_scalp_shapes": muted_scalp_shapes,
        "breakdown": list(stats_by_key.values())
    }

    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(learning_state, f, ensure_ascii=False, indent=2)

    lines = ["# Live Signal Journal Post-Mortem (Closed-Loop Self-Learning Memory)"]
    if not all_closed:
        lines.append("- ยังไม่มีออเดอร์จริงที่ปิดสถานะ ระบบพร้อมบันทึกบทเรียนอัตโนมัติทันทีเมื่อชน TP หรือ SL")
    else:
        wr_all = round((learning_state["overall_wins"] / len(all_closed)) * 100.0, 1)
        lines.append(f"- สถิติสัญญาณจริงสะสม {len(all_closed)} ไม้: Win-Rate = {wr_all}% | Net PnL = ${learning_state['overall_pnl']}/oz")
        for st in stats_by_key.values():
            flag = " [LIVE AUTO-MUTED: แพ้ต่อเนื่อง]" if st["auto_muted"] else " [ACTIVE]"
            lines.append(f"- `{st['mode']}:{st['name']}`: {st['w']}W/{st['l']}L (WR {st['wr']}%, {st['net_r']}R){flag}")

    with open(lesson_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return learning_state

def ensure_and_load_knowledge_base():
    kb_dir = "knowledge_base"
    os.makedirs(kb_dir, exist_ok=True)
    for fname, content in DEFAULT_KB_FILES.items():
        fpath = os.path.join(kb_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)

    learning_state = evaluate_live_self_learning()

    loaded_texts = []
    loaded_names = []
    for fname in sorted(os.listdir(kb_dir)):
        if fname.endswith(".md"):
            fpath = os.path.join(kb_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                loaded_texts.append(f.read().strip())
                loaded_names.append(fname)

    return "\n\n".join(loaded_texts), loaded_names, learning_state

def clean_yf_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def call_thaillm_agent(model_name, prompt, fallback_text="", temperature=0.1):
    if not THAILLM_API_KEY or THAILLM_API_KEY in ("YOUR_THAILLM_API_KEY", "MISSING_KEY"):
        return fallback_text, "quant-local-engine"

    models_to_try = [model_name]
    if model_name != MODEL_CHIEF_STRATEGIST:
        models_to_try.append(MODEL_CHIEF_STRATEGIST)

    for m in models_to_try:
        try:
            response = llm_client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=35
            )
            raw = response.choices[0].message.content.strip()
            cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            return cleaned, m
        except Exception as e:
            print(f"[AI] Model '{m}' error: {e} -> trying fallback...")
    return fallback_text, "quant-fallback"

# ================= 1. INSTITUTIONAL REAL-TIME XAU/USD FEED =================
def fetch_true_realtime_spot():
    """ดึงราคา Spot XAU/USD ระดับสถาบันแบบอัตโนมัติ (Binance XAUUSDT TradFi + Swissquote BBO + Gold-API)"""
    spot_mid, spot_bid, spot_ask = None, None, None
    feed_source = "Swissquote Institutional BBO"
    xau_perp_mid = None

    # 1. Binance Futures TradFi XAUUSDT BookTicker (Real-Time LBMA/Spot Index Contract)
    try:
        res_bn = requests.get(
            "https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol=XAUUSDT",
            timeout=6
        ).json()
        if "bidPrice" in res_bn and "askPrice" in res_bn:
            b_bid = round(float(res_bn["bidPrice"]), 2)
            b_ask = round(float(res_bn["askPrice"]), 2)
            xau_perp_mid = round((b_bid + b_ask) / 2.0, 2)
            spot_bid, spot_ask, spot_mid = b_bid, b_ask, xau_perp_mid
            feed_source = "Binance XAUUSDT Institutional"
    except Exception:
        pass

    # 2. Swissquote Institutional Spot XAU/USD BBO (Cross-check / Primary OTC Spot)
    try:
        res_sq = requests.get(
            "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD",
            timeout=6
        ).json()
        prices = res_sq[0]["spreadProfilePrices"][0]
        sq_bid = round(float(prices["bid"]), 2)
        sq_ask = round(float(prices["ask"]), 2)
        sq_mid = round((sq_bid + sq_ask) / 2.0, 2)
        if spot_mid is None:
            spot_bid, spot_ask, spot_mid = sq_bid, sq_ask, sq_mid
            feed_source = "Swissquote XAU/USD Spot"
        else:
            # เฉลี่ยถ่วงน้ำหนักให้ตรงกับราคาตลาด Spot โบรกเกอร์มาตรฐาน
            spot_mid = round((spot_mid + sq_mid) / 2.0, 2)
            half_s = max(0.10, round((sq_ask - sq_bid) / 2.0, 2))
            spot_bid = round(spot_mid - half_s, 2)
            spot_ask = round(spot_mid + half_s, 2)
            feed_source = "XAU/USD Institutional Composite"
    except Exception:
        pass

    # 3. Gold-API Fallback
    if spot_mid is None:
        try:
            res = requests.get("https://api.gold-api.com/price/XAU", timeout=6).json()
            if "price" in res and res["price"]:
                spot_mid = round(float(res["price"]), 2)
                spot_bid = round(spot_mid - 0.12, 2)
                spot_ask = round(spot_mid + 0.12, 2)
                feed_source = "Gold-API Spot XAU"
        except Exception:
            pass

    return spot_mid, spot_bid, spot_ask, feed_source

def sync_hydra_m15_vault(df_fresh):
    os.makedirs("data", exist_ok=True)
    vault_path = "data/gold_m15_vault.csv"

    df_new = df_fresh[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    if df_new.index.tz is None:
        df_new.index = df_new.index.tz_localize("UTC")
    else:
        df_new.index = df_new.index.tz_convert("UTC")

    if os.path.exists(vault_path):
        try:
            df_old = pd.read_csv(vault_path, index_col=0, parse_dates=True)
            if df_old.index.tz is None:
                df_old.index = df_old.index.tz_localize("UTC")
            else:
                df_old.index = df_old.index.tz_convert("UTC")
            df_merged = pd.concat([df_old, df_new])
            df_merged = df_merged[~df_merged.index.duplicated(keep='last')].sort_index()
        except Exception:
            df_merged = df_new.sort_index()
    else:
        df_merged = df_new.sort_index()

    df_merged = df_merged.tail(18000)
    try:
        df_merged.to_csv(vault_path)
    except Exception:
        pass

    return df_merged

def fetch_market_data():
    df_fresh_60d = clean_yf_df(yf.download("GC=F", period="60d", interval="15m", progress=False))
    df_intraday_vault = sync_hydra_m15_vault(df_fresh_60d)

    df_daily = clean_yf_df(yf.download("GC=F", period="120d", interval="1d", progress=False))
    df_dxy = clean_yf_df(yf.download("DX-Y.NYB", period="60d", interval="1d", progress=False))
    df_us10y = clean_yf_df(yf.download("^TNX", period="60d", interval="1d", progress=False))
    
    spot_price, bid, ask, feed_source = fetch_true_realtime_spot()

    try:
        gc_info = yf.Ticker("GC=F").info
        open_interest = gc_info.get("openInterest", None)
    except Exception:
        open_interest = None
        
    return df_intraday_vault, df_daily, df_dxy, df_us10y, spot_price, bid, ask, feed_source, open_interest

def compute_mtf_indicators(df_intraday, spot_offset=0.0):
    df_s = pd.DataFrame({
        'Open': df_intraday['Open'] + spot_offset,
        'High': df_intraday['High'] + spot_offset,
        'Low': df_intraday['Low'] + spot_offset,
        'Close': df_intraday['Close'] + spot_offset,
        'Volume': df_intraday['Volume']
    }, index=df_intraday.index)

    df_h1 = df_s.resample('1h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    h1_ema50 = df_h1['Close'].ewm(span=50, adjust=False).mean().shift(1)
    h1_ema200 = df_h1['Close'].ewm(span=200, adjust=False).mean().shift(1)
    h1_ema50_m15 = h1_ema50.reindex(df_s.index, method='ffill').bfill()
    h1_ema200_m15 = h1_ema200.reindex(df_s.index, method='ffill').bfill()

    ema21 = df_s['Close'].ewm(span=21, adjust=False).mean()
    ema50 = df_s['Close'].ewm(span=50, adjust=False).mean()
    ema200 = df_s['Close'].ewm(span=200, adjust=False).mean()

    tr1 = df_s['High'] - df_s['Low']
    tr2 = (df_s['High'] - df_s['Close'].shift(1)).abs()
    tr3 = (df_s['Low'] - df_s['Close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    delta = df_s['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-6)
    rs = gain / loss
    rsi14 = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    up_move = df_s['High'].diff()
    down_move = -df_s['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_safe = atr14.replace(0, 1e-6)
    plus_di = 100.0 * pd.Series(plus_dm, index=df_s.index).rolling(14).mean() / atr_safe
    minus_di = 100.0 * pd.Series(minus_dm, index=df_s.index).rolling(14).mean() / atr_safe
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1.0)
    adx14 = dx.rolling(14).mean().fillna(20.0)

    pdh = df_s['High'].rolling(92).max().shift(1)
    pdl = df_s['Low'].rolling(92).min().shift(1)

    utc_idx = df_s.index.tz_convert("UTC")
    dates_only = pd.Series(utc_idx.date, index=df_s.index)
    asian_mask = (utc_idx.hour >= 0) & (utc_idx.hour <= 5)
    asian_df = df_s[asian_mask]
    if not asian_df.empty:
        daily_asian = asian_df.groupby(asian_df.index.tz_convert("UTC").date).agg(
            asian_high=('High', 'max'),
            asian_low=('Low', 'min')
        )
        asian_high = dates_only.map(daily_asian['asian_high']).ffill().bfill()
        asian_low = dates_only.map(daily_asian['asian_low']).ffill().bfill()
    else:
        asian_high = pdh
        asian_low = pdl

    return {
        "ema21": ema21, "ema50": ema50, "ema200": ema200,
        "h1_ema50": h1_ema50_m15, "h1_ema200": h1_ema200_m15,
        "atr14": atr14, "rsi14": rsi14, "adx14": adx14,
        "pdh": pdh, "pdl": pdl, "asian_high": asian_high, "asian_low": asian_low
    }

def calculate_volume_profile(df_intraday, spot_offset=0.0, bins=36):
    df = df_intraday.tail(460).copy()
    df_vol = df[df['Volume'] > 0].copy()
    if df_vol.empty:
        df_vol = df.copy()
        df_vol['Volume'] = 1

    price_min = float(df_vol['Low'].min()) + spot_offset
    price_max = float(df_vol['High'].max()) + spot_offset
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    typical_price = ((df_vol['High'] + df_vol['Low'] + df_vol['Close']) / 3) + spot_offset
    vol_profile, _ = np.histogram(typical_price, bins=bin_edges, weights=df_vol['Volume'])
    
    poc_idx = int(np.argmax(vol_profile))
    poc_price = float(bin_centers[poc_idx])
    
    total_vol = np.sum(vol_profile)
    target_vol = total_vol * 0.70
    current_vol = vol_profile[poc_idx]
    low_idx, high_idx = poc_idx, poc_idx
    
    while current_vol < target_vol and (low_idx > 0 or high_idx < bins - 1):
        vol_down = vol_profile[low_idx - 1] if low_idx > 0 else 0
        vol_up = vol_profile[high_idx + 1] if high_idx < bins - 1 else 0
        if vol_up >= vol_down and high_idx < bins - 1:
            high_idx += 1
            current_vol += vol_up
        elif low_idx > 0:
            low_idx -= 1
            current_vol += vol_down
        else:
            break

    histogram = [
        {"price": round(float(p), 1), "volume": int(v)}
        for p, v in reversed(list(zip(bin_centers, vol_profile)))
    ]
    
    return {
        "poc": round(poc_price, 2),
        "vah": round(float(bin_edges[high_idx + 1]), 2),
        "val": round(float(bin_edges[low_idx]), 2),
        "histogram": histogram
    }

# ================= 1B. 4-ANATOMY VOLUME PROFILE ANALYZER =================
ANATOMY_META = {
    "D_SHAPED": {
        "title_th": "① D-SHAPED (ทรงตัว D - ตลาดสมดุล)",
        "desc_th": "POC อยู่กึ่งกลางกรอบ (34%–66%) โวลุ่มสมมาตรคล้ายระฆังคว่ำ",
        "rule_th": "ดัก BUY ที่แนวรับ VAL เด้งกลับ และดัก SELL ที่แนวต้าน VAH เป้าหมายเข้าหา POC"
    },
    "B_SHAPED": {
        "title_th": "② b-SHAPED (ทรงตัว b - โวลุ่มกองโซนล่าง)",
        "desc_th": "POC อยู่โซนล่างต่ำกว่า 34% จากแรงเทขายหรือสะสมพลังฐานล่าง",
        "rule_th": "เปิด SELL เมื่อโมเมนตัมลงและกดใต้ POC หรือเปิด BUY เมื่อเบรกทะลุ VAH ขึ้นไป"
    },
    "P_SHAPED": {
        "title_th": "③ P-SHAPED (ทรงตัว P - โวลุ่มกองโซนบน)",
        "desc_th": "POC อยู่โซนบนสูงกว่า 66% โมเมนตัมขาขึ้นแข็งแกร่ง / Short Covering",
        "rule_th": "เปิด BUY ตามแนวโน้มเมื่อย่อรับที่ POC/VAL และเปิด SELL หากหลุดกรอบล่าง VAL"
    },
    "DOUBLE_ZONE": {
        "title_th": "④ DOUBLE ZONE (ทรง 2 โซน / 2 ยอดโวลุ่ม)",
        "desc_th": "มียอดโวลุ่มที่ 2 (>= 60% ของ POC) และมีหุบเขาโวลุ่มต่ำ (Valley) คั่นกลาง",
        "rule_th": "ดัก BUY เมื่อเด้งขึ้นจากยอดโวลุ่มล่าง (Lower Peak) และดัก SELL ที่ยอดโวลุ่มบน (Upper Peak)"
    }
}

def classify_vp_anatomy(low_win, high_win, close_win, vol_win, bins=32):
    w_min = float(low_win.min())
    w_max = float(high_win.max())
    if w_max - w_min < 2.0:
        return None
    bin_edges = np.linspace(w_min, w_max, bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    typical = (high_win + low_win + close_win) / 3.0
    weights = vol_win if vol_win.sum() > 0 else np.ones_like(vol_win)
    hist, _ = np.histogram(typical, bins=bin_edges, weights=weights)

    kernel = np.array([0.2, 0.6, 0.2])
    smooth_hist = np.convolve(hist, kernel, mode='same')

    poc_idx = int(np.argmax(smooth_hist))
    poc_price = float(centers[poc_idx])
    poc_vol = float(smooth_hist[poc_idx])
    if poc_vol <= 0:
        return None

    tot_vol = hist.sum()
    cur_vol = hist[poc_idx]
    lo_i, hi_i = poc_idx, poc_idx
    while cur_vol < tot_vol * 0.70 and (lo_i > 0 or hi_i < bins - 1):
        vd = hist[lo_i - 1] if lo_i > 0 else 0
        vu = hist[hi_i + 1] if hi_i < bins - 1 else 0
        if vu >= vd and hi_i < bins - 1:
            hi_i += 1
            cur_vol += vu
        elif lo_i > 0:
            lo_i -= 1
            cur_vol += vd
        else:
            break
    val_price = float(bin_edges[lo_i])
    vah_price = float(bin_edges[hi_i + 1])

    peaks = []
    for k in range(1, bins - 1):
        if smooth_hist[k] >= smooth_hist[k - 1] and smooth_hist[k] >= smooth_hist[k + 1]:
            peaks.append((k, float(smooth_hist[k]), float(centers[k])))
    if smooth_hist[0] > smooth_hist[1]:
        peaks.append((0, float(smooth_hist[0]), float(centers[0])))
    if smooth_hist[-1] > smooth_hist[-2]:
        peaks.append((bins - 1, float(smooth_hist[-1]), float(centers[-1])))
    peaks.sort(key=lambda x: x[1], reverse=True)

    second_peak = None
    valley_price = None
    for idx_p, v_p, p_price in peaks:
        if abs(idx_p - poc_idx) >= 4 and v_p >= 0.60 * poc_vol:
            lo_b, hi_b = min(poc_idx, idx_p), max(poc_idx, idx_p)
            valley_slice = smooth_hist[lo_b + 1:hi_b]
            if len(valley_slice) > 0:
                min_v = float(valley_slice.min())
                if min_v <= 0.78 * v_p:
                    v_rel_idx = int(np.argmin(valley_slice))
                    valley_price = float(centers[lo_b + 1 + v_rel_idx])
                    second_peak = (idx_p, v_p, p_price)
                    break

    poc_pos_ratio = (poc_price - w_min) / (w_max - w_min)

    if second_peak is not None:
        shape = "DOUBLE_ZONE"
        lower_peak = round(min(poc_price, second_peak[2]), 2)
        upper_peak = round(max(poc_price, second_peak[2]), 2)
    elif poc_pos_ratio < 0.34:
        shape = "B_SHAPED"
        lower_peak = round(poc_price, 2)
        upper_peak = round(vah_price, 2)
    elif poc_pos_ratio > 0.66:
        shape = "P_SHAPED"
        lower_peak = round(val_price, 2)
        upper_peak = round(poc_price, 2)
    else:
        shape = "D_SHAPED"
        lower_peak = round(val_price, 2)
        upper_peak = round(vah_price, 2)

    meta = ANATOMY_META[shape]
    return {
        "shape": shape,
        "title_th": meta["title_th"],
        "desc_th": meta["desc_th"],
        "rule_th": meta["rule_th"],
        "poc": round(poc_price, 2),
        "vah": round(vah_price, 2),
        "val": round(val_price, 2),
        "poc_pos_pct": round(poc_pos_ratio * 100.0, 1),
        "second_peak": round(second_peak[2], 2) if second_peak else None,
        "valley": round(valley_price, 2) if valley_price else None,
        "lower_peak": lower_peak,
        "upper_peak": upper_peak
    }

def fetch_usd_news():
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        res = requests.get(url, timeout=10).json()
        now_utc = datetime.now(timezone.utc)
        upcoming_high = []
        upcoming_med = []
        recent_past = []
        
        for item in res:
            if item.get("country") == "USD" and item.get("impact") in ("High", "Medium"):
                event_dt_utc = datetime.fromisoformat(item["date"]).astimezone(timezone.utc)
                event_dt_th = event_dt_utc.astimezone(THAI_TZ)
                diff_hours = round((event_dt_utc - now_utc).total_seconds() / 3600, 1)
                
                entry = {
                    "time_th": event_dt_th.strftime("%d/%m %H:%M น."),
                    "title": item.get("title", ""),
                    "impact": item.get("impact", "High"),
                    "forecast": item.get("forecast") or "-",
                    "previous": item.get("previous") or "-",
                    "actual": item.get("actual") or "-",
                    "hours_away": diff_hours,
                    "is_upcoming": diff_hours >= 0,
                    "status": f"อีก {diff_hours} ชม." if diff_hours >= 0 else f"ออกแล้ว ({item.get('actual') or 'ดูผล'})"
                }
                if diff_hours >= 0:
                    if item.get("impact") == "High":
                        upcoming_high.append(entry)
                    else:
                        upcoming_med.append(entry)
                else:
                    if item.get("impact") == "High":
                        recent_past.append(entry)
                        
        combined_upcoming = sorted(upcoming_high + upcoming_med[:3], key=lambda x: x["hours_away"])
        if len(combined_upcoming) >= 3:
            return combined_upcoming[:6]
        return (combined_upcoming + recent_past[-3:])[:6]
    except Exception:
        return []

# ================= INSTITUTIONAL DAILY-AGGREGATED RISK METRICS =================
def compute_institutional_risk_metrics(df_records, all_dates):
    r_arr = df_records["r"].values
    dates_col = df_records["date"].values

    unique_days = pd.Index(sorted(set(all_dates)))
    daily_r = pd.Series(0.0, index=unique_days)
    for d, r_val in zip(dates_col, r_arr):
        if d in daily_r.index:
            daily_r[d] += float(r_val)

    d_vals = daily_r.values
    mean_d = float(np.mean(d_vals)) if len(d_vals) > 0 else 0.0
    std_d = float(np.std(d_vals, ddof=1)) if len(d_vals) > 1 else 1e-4
    downside_d = float(np.sqrt(np.mean(np.minimum(0.0, d_vals) ** 2)))

    ann_factor = np.sqrt(252.0)
    sharpe = round(float(np.clip((mean_d / max(std_d, 1e-4)) * ann_factor, -5.0, 6.0)), 2)
    sortino = round(float(np.clip((mean_d / max(downside_d, 1e-4)) * ann_factor, -5.0, 9.0)), 2)

    cum_r = np.cumsum(r_arr)
    max_dd_r = float((np.maximum.accumulate(cum_r) - cum_r).max()) if len(cum_r) > 0 else 0.0
    ann_return_r = mean_d * 252.0
    calmar = round(float(np.clip(ann_return_r / max(max_dd_r, 1.0), -5.0, 12.0)), 2)

    expectancy_r = round(float(np.mean(r_arr)), 2)
    var_95_r = round(float(np.percentile(r_arr, 5)), 2)

    max_consec_loss = 0
    cur_l = 0
    for r_v in r_arr:
        if r_v < 0:
            cur_l += 1
            max_consec_loss = max(max_consec_loss, cur_l)
        else:
            cur_l = 0

    rng = np.random.default_rng(42)
    mc_sims = rng.choice(r_arr, size=(2000, len(r_arr)), replace=True)
    mc_cum = np.cumsum(mc_sims, axis=1)
    mc_peaks = np.maximum.accumulate(mc_cum, axis=1)
    mc_dds = np.max(mc_peaks - mc_cum, axis=1)
    mc_dd_95_r = round(float(np.percentile(mc_dds, 95)), 2)
    mc_prob_profit = round(float(np.mean(mc_cum[:, -1] > 0) * 100.0), 1)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "expectancy_r": expectancy_r,
        "max_consec_loss": int(max_consec_loss),
        "var_95_r": var_95_r,
        "mc_dd_95_r": mc_dd_95_r,
        "mc_prob_profit": mc_prob_profit,
        "max_dd_r": round(max_dd_r, 2)
    }

# ================= 2A. TREND WALK-FORWARD ENGINE (PESSIMISTIC FILL + AUDIT TRAIL) =================
def run_walkforward_backtest(df_intraday, ind, spot_offset=0.0):
    df = df_intraday.copy()
    open_arr = df['Open'].values + spot_offset
    high_arr = df['High'].values + spot_offset
    low_arr = df['Low'].values + spot_offset
    close_arr = df['Close'].values + spot_offset
    vol_arr = df['Volume'].values

    ema21_arr = ind["ema21"].values
    ema50_arr = ind["ema50"].values
    ema200_arr = ind["ema200"].values
    h1_ema50_arr = ind["h1_ema50"].values
    h1_ema200_arr = ind["h1_ema200"].values
    atr_arr = ind["atr14"].values
    rsi_arr = ind["rsi14"].values
    adx_arr = ind["adx14"].values
    pdh_arr = ind["pdh"].values
    pdl_arr = ind["pdl"].values
    asian_h_arr = ind["asian_high"].values
    asian_l_arr = ind["asian_low"].values
    n_bars = len(df)

    dt_index_utc = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    dt_index_th = dt_index_utc.tz_convert(THAI_TZ)
    utc_hours = dt_index_utc.hour.values
    date_ordinals = dt_index_utc.date

    bars_per_day = 92
    lookback_5d = 5 * bars_per_day
    split_idx = lookback_5d + int((n_bars - lookback_5d) * 0.67)
    spread_cost = 0.60  # $0.60/oz = 60 points (Spread + ECN Commission)

    all_factors = ["VALUE_AREA_FADE", "LIQUIDITY_SWEEP", "LONDON_BREAKOUT", "MOMENTUM_BREAKOUT", "HTF_TREND_PULLBACK"]

    default_best = {
        "atr_sl_mult": 1.6, "rr_mult": 2.2, "adx_thresh": 25,
        "enabled_factors": all_factors, "trades": 0, "win_rate": 0.0,
        "profit_factor": 0.0, "r_profit_factor": 0.0, "net_pnl": 0.0, "net_r": 0.0,
        "expectancy_r": 0.0, "max_consec_loss": 0,
        "max_dd": 0.0, "max_dd_r": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "calmar": 0.0, "var_95_r": -1.0, "mc_dd_95_r": -4.0, "mc_prob_profit": 50.0,
        "total_fee_drag": 0.0, "oos_trades": 0, "oos_win_rate": 0.0,
        "oos_profit_factor": 0.0, "oos_net_r": 0.0, "oos_net_pnl": 0.0,
        "is_profitable_edge": False, "equity_curve_r": [0], "trade_r_list": [], "recent_sim_trades": []
    }
    if n_bars < lookback_5d + 120:
        return {"best": default_best, "factor_breakdown": [], "top_5": [default_best], "split_bar": split_idx}

    vp_val = np.full(n_bars, np.nan)
    vp_poc = np.full(n_bars, np.nan)
    vp_vah = np.full(n_bars, np.nan)
    typical = (high_arr + low_arr + close_arr) / 3.0

    for i in range(lookback_5d, n_bars, 16):
        w_low = low_arr[i - lookback_5d:i].min()
        w_high = high_arr[i - lookback_5d:i].max()
        if w_high <= w_low:
            continue
        bins = np.linspace(w_low, w_high, 31)
        centers = (bins[:-1] + bins[1:]) / 2.0
        weights = vol_arr[i - lookback_5d:i]
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        hist, _ = np.histogram(typical[i - lookback_5d:i], bins=bins, weights=weights)
        poc_idx = int(np.argmax(hist))
        tot = hist.sum()
        cur = hist[poc_idx]
        lo_i, hi_i = poc_idx, poc_idx
        while cur < tot * 0.70 and (lo_i > 0 or hi_i < 29):
            vd = hist[lo_i - 1] if lo_i > 0 else 0
            vu = hist[hi_i + 1] if hi_i < 29 else 0
            if vu >= vd and hi_i < 29:
                hi_i += 1; cur += vu
            elif lo_i > 0:
                lo_i -= 1; cur += vd
            else:
                break
        end_idx = min(i + 16, n_bars)
        vp_val[i:end_idx] = bins[lo_i]
        vp_poc[i:end_idx] = centers[poc_idx]
        vp_vah[i:end_idx] = bins[hi_i + 1]

    def simulate_raw_trades(atr_sl_mult, rr_mult, adx_thresh, enabled_set):
        in_pos = 0
        active_factor = ""
        entry_p = sl_p = tp_p = init_risk = 0.0
        entry_time_str = ""
        be_moved = False
        trail_locked = False
        trade_records = []
        cooldown = 0

        for i in range(lookback_5d + 1, n_bars):
            if np.isnan(vp_val[i]) or np.isnan(atr_arr[i]) or atr_arr[i] <= 0 or np.isnan(pdh_arr[i]):
                continue

            # Pessimistic Intrabar Execution: ตรวจ SL ก่อน TP เสมอในแท่งเดียวกัน
            if in_pos == 1:
                if low_arr[i] <= sl_p:
                    pnl = (sl_p - entry_p) - spread_cost
                    r_val = pnl / init_risk
                    trade_records.append({
                        "idx": i, "engine": "TREND", "factor": active_factor, "side": "LONG",
                        "entry_time": entry_time_str, "exit_time": dt_index_th[i].strftime("%d/%m %H:%M"),
                        "entry": round(entry_p, 2), "exit": round(sl_p, 2),
                        "pnl": round(pnl, 2), "r": round(r_val, 2), "date": date_ordinals[i]
                    })
                    in_pos = 0; cooldown = 3; continue
                if high_arr[i] >= tp_p:
                    pnl = (tp_p - entry_p) - spread_cost
                    r_val = pnl / init_risk
                    trade_records.append({
                        "idx": i, "engine": "TREND", "factor": active_factor, "side": "LONG",
                        "entry_time": entry_time_str, "exit_time": dt_index_th[i].strftime("%d/%m %H:%M"),
                        "entry": round(entry_p, 2), "exit": round(tp_p, 2),
                        "pnl": round(pnl, 2), "r": round(r_val, 2), "date": date_ordinals[i]
                    })
                    in_pos = 0; cooldown = 2; continue
                if not be_moved and high_arr[i] >= entry_p + init_risk * 1.0:
                    sl_p = entry_p + (atr_arr[i] * 0.15); be_moved = True
                if not trail_locked and high_arr[i] >= entry_p + init_risk * 1.8:
                    sl_p = max(sl_p, entry_p + init_risk * 1.0); trail_locked = True
                continue

            elif in_pos == -1:
                if high_arr[i] >= sl_p:
                    pnl = (entry_p - sl_p) - spread_cost
                    r_val = pnl / init_risk
                    trade_records.append({
                        "idx": i, "engine": "TREND", "factor": active_factor, "side": "SHORT",
                        "entry_time": entry_time_str, "exit_time": dt_index_th[i].strftime("%d/%m %H:%M"),
                        "entry": round(entry_p, 2), "exit": round(sl_p, 2),
                        "pnl": round(pnl, 2), "r": round(r_val, 2), "date": date_ordinals[i]
                    })
                    in_pos = 0; cooldown = 3; continue
                if low_arr[i] <= tp_p:
                    pnl = (entry_p - tp_p) - spread_cost
                    r_val = pnl / init_risk
                    trade_records.append({
                        "idx": i, "engine": "TREND", "factor": active_factor, "side": "SHORT",
                        "entry_time": entry_time_str, "exit_time": dt_index_th[i].strftime("%d/%m %H:%M"),
                        "entry": round(entry_p, 2), "exit": round(tp_p, 2),
                        "pnl": round(pnl, 2), "r": round(r_val, 2), "date": date_ordinals[i]
                    })
                    in_pos = 0; cooldown = 2; continue
                if not be_moved and low_arr[i] <= entry_p - init_risk * 1.0:
                    sl_p = entry_p - (atr_arr[i] * 0.15); be_moved = True
                if not trail_locked and low_arr[i] <= entry_p - init_risk * 1.8:
                    sl_p = min(sl_p, entry_p - init_risk * 1.0); trail_locked = True
                continue

            if cooldown > 0:
                cooldown -= 1; continue

            hr = utc_hours[i]
            if not (6 <= hr <= 17):
                continue

            atr = atr_arr[i]; adx = adx_arr[i]; rsi = rsi_arr[i]
            e21 = ema21_arr[i]; e50 = ema50_arr[i]; e200 = ema200_arr[i]
            h1_e50 = h1_ema50_arr[i]; h1_e200 = h1_ema200_arr[i]
            pdh = pdh_arr[i]; pdl = pdl_arr[i]
            as_h = asian_h_arr[i]; as_l = asian_l_arr[i]
            risk = atr * atr_sl_mult
            reward = risk * rr_mult

            c_range = max(high_arr[i] - low_arr[i], 1e-6)
            body_pct = abs(close_arr[i] - open_arr[i]) / c_range
            lower_wick_pct = (min(open_arr[i], close_arr[i]) - low_arr[i]) / c_range
            upper_wick_pct = (high_arr[i] - max(open_arr[i], close_arr[i])) / c_range

            bull_candle = close_arr[i] > open_arr[i] and close_arr[i] > close_arr[i - 1]
            bear_candle = close_arr[i] < open_arr[i] and close_arr[i] < close_arr[i - 1]
            t_str = dt_index_th[i].strftime("%d/%m %H:%M")

            if "VALUE_AREA_FADE" in enabled_set and adx <= adx_thresh:
                if low_arr[i] <= vp_val[i] + atr * 0.35 and close_arr[i] > vp_val[i] - atr * 0.25 and bull_candle and rsi <= 45:
                    in_pos = 1; active_factor = "VALUE_AREA_FADE"; be_moved = False; trail_locked = False; entry_time_str = t_str
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p - risk; tp_p = entry_p + reward; continue
                elif high_arr[i] >= vp_vah[i] - atr * 0.35 and close_arr[i] < vp_vah[i] + atr * 0.25 and bear_candle and rsi >= 55:
                    in_pos = -1; active_factor = "VALUE_AREA_FADE"; be_moved = False; trail_locked = False; entry_time_str = t_str
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p + risk; tp_p = entry_p - reward; continue

            swept_pdl = (low_arr[i] < pdl or low_arr[i - 1] < pdl) and close_arr[i] > pdl and bull_candle and (lower_wick_pct >= 0.22 or rsi <= 45)
            swept_pdh = (high_arr[i] > pdh or high_arr[i - 1] > pdh) and close_arr[i] < pdh and bear_candle and (upper_wick_pct >= 0.22 or rsi >= 55)

            if "LIQUIDITY_SWEEP" in enabled_set and swept_pdl:
                in_pos = 1; active_factor = "LIQUIDITY_SWEEP"; be_moved = False; trail_locked = False; entry_time_str = t_str
                entry_p = close_arr[i]
                swing_low = min(low_arr[i], low_arr[i - 1])
                init_risk = max(entry_p - (swing_low - atr * 0.25), risk * 0.85)
                sl_p = entry_p - init_risk; tp_p = entry_p + (init_risk * rr_mult); continue
            elif "LIQUIDITY_SWEEP" in enabled_set and swept_pdh:
                in_pos = -1; active_factor = "LIQUIDITY_SWEEP"; be_moved = False; trail_locked = False; entry_time_str = t_str
                entry_p = close_arr[i]
                swing_high = max(high_arr[i], high_arr[i - 1])
                init_risk = max((swing_high + atr * 0.25) - entry_p, risk * 0.85)
                sl_p = entry_p + init_risk; tp_p = entry_p - (init_risk * rr_mult); continue

            if "LONDON_BREAKOUT" in enabled_set and (6 <= hr <= 9) and not np.isnan(as_h) and body_pct >= 0.45:
                htf_bear = close_arr[i] < h1_e200 and close_arr[i] < e200
                htf_bull = close_arr[i] > h1_e200 and close_arr[i] > e200
                if htf_bear and close_arr[i] < as_l and close_arr[i - 1] >= as_l and bear_candle and (28 <= rsi <= 52):
                    in_pos = -1; active_factor = "LONDON_BREAKOUT"; be_moved = False; trail_locked = False; entry_time_str = t_str
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p + risk; tp_p = entry_p - reward; continue
                elif htf_bull and close_arr[i] > as_h and close_arr[i - 1] <= as_h and bull_candle and (48 <= rsi <= 72):
                    in_pos = 1; active_factor = "LONDON_BREAKOUT"; be_moved = False; trail_locked = False; entry_time_str = t_str
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p - risk; tp_p = entry_p + reward; continue

            if "MOMENTUM_BREAKOUT" in enabled_set and adx > adx_thresh and body_pct >= 0.45:
                htf_bear = close_arr[i] < h1_e200 and h1_e50 < h1_e200 and close_arr[i] < e200
                htf_bull = close_arr[i] > h1_e200 and h1_e50 > h1_e200 and close_arr[i] > e200
                broke_down = close_arr[i] < vp_val[i] and close_arr[i - 1] >= vp_val[i] and bear_candle and (28 <= rsi <= 48)
                broke_up = close_arr[i] > vp_vah[i] and close_arr[i - 1] <= vp_vah[i] and bull_candle and (52 <= rsi <= 72)

                if htf_bear and broke_down:
                    in_pos = -1; active_factor = "MOMENTUM_BREAKOUT"; be_moved = False; trail_locked = False; entry_time_str = t_str
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p + risk; tp_p = entry_p - reward; continue
                elif htf_bull and broke_up:
                    in_pos = 1; active_factor = "MOMENTUM_BREAKOUT"; be_moved = False; trail_locked = False; entry_time_str = t_str
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p - risk; tp_p = entry_p + reward; continue

            if "HTF_TREND_PULLBACK" in enabled_set and adx > adx_thresh:
                htf_bear = close_arr[i] < h1_e200 and h1_e50 < h1_e200 and close_arr[i] < e200 and e50 < e200
                htf_bull = close_arr[i] > h1_e200 and h1_e50 > h1_e200 and close_arr[i] > e200 and e50 > e200
                if htf_bear and high_arr[i] >= e21 and close_arr[i] < e21 and bear_candle and rsi >= 48:
                    in_pos = -1; active_factor = "HTF_TREND_PULLBACK"; be_moved = False; trail_locked = False; entry_time_str = t_str
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p + risk; tp_p = entry_p - reward
                elif htf_bull and low_arr[i] <= e21 and close_arr[i] > e21 and bull_candle and rsi <= 52:
                    in_pos = 1; active_factor = "HTF_TREND_PULLBACK"; be_moved = False; trail_locked = False; entry_time_str = t_str
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p - risk; tp_p = entry_p + reward

        return trade_records

    def summarize_records(records, raw_records, sl_m, rr_m, adx_t, enabled_list):
        if len(records) < 5:
            return None
        df_tr = pd.DataFrame(records)
        df_raw_all = pd.DataFrame(raw_records) if raw_records else df_tr
        arr = df_tr["pnl"].values
        r_arr = df_tr["r"].values

        wins_r = r_arr[r_arr > 0]
        losses_r = r_arr[r_arr < 0]
        wr = (len(wins_r) / len(r_arr)) * 100.0

        gross_r_win = float(wins_r.sum()) if len(wins_r) > 0 else 0.0
        gross_r_loss = float(abs(losses_r.sum())) if len(losses_r) > 0 else 1e-6
        r_pf = round(min(gross_r_win / gross_r_loss, 9.99), 2)

        wins_usd = arr[arr > 0]
        losses_usd = arr[arr < 0]
        usd_pf = round(min((float(wins_usd.sum()) / float(abs(losses_usd.sum()))) if len(losses_usd) > 0 else 2.0, 9.99), 2)

        net_pnl = float(arr.sum())
        net_r = float(r_arr.sum())

        cum_usd = np.cumsum(arr)
        max_dd_usd = float((np.maximum.accumulate(cum_usd) - cum_usd).max()) if len(cum_usd) > 0 else 0.0
        cum_r = np.cumsum(r_arr)
        total_fee_drag = round(len(arr) * spread_cost, 2)

        inst = compute_institutional_risk_metrics(df_tr, date_ordinals[lookback_5d:])

        oos_df = df_tr[df_tr["idx"] >= split_idx]
        if len(oos_df) > 0:
            o_r = oos_df["r"].values
            o_pnl = oos_df["pnl"].values
            o_w = o_r[o_r > 0]; o_l = o_r[o_r < 0]
            oos_wr = round((len(o_w) / len(o_r)) * 100.0, 1)
            oos_r_pf = round(min((float(o_w.sum()) / float(abs(o_l.sum()))) if len(o_l) > 0 else (2.0 if len(o_w) > 0 else 0.0), 9.99), 2)
            oos_net_r = round(float(o_r.sum()), 2)
            oos_pnl = round(float(o_pnl.sum()), 2)
            oos_trades = int(len(oos_df))
        else:
            oos_wr, oos_r_pf, oos_net_r, oos_pnl, oos_trades = 0.0, 0.0, 0.0, 0.0, 0

        is_df = df_tr[df_tr["idx"] < split_idx]
        if len(is_df) >= 3:
            is_r = is_df["r"].values
            is_w = is_r[is_r > 0]; is_l = is_r[is_r < 0]
            is_r_pf = min((float(is_w.sum()) / float(abs(is_l.sum()))) if len(is_l) > 0 else 2.0, 9.99)
            is_net_r = float(is_r.sum())
            is_cum_r = np.cumsum(is_r)
            is_dd_r = float((np.maximum.accumulate(is_cum_r) - is_cum_r).max())
            train_score = (is_r_pf * 2.0 + is_net_r * 0.25) * np.sqrt(len(is_r)) / (1.0 + is_dd_r * 0.2)
        else:
            train_score = r_pf * np.sqrt(len(r_arr))

        breakdown = []
        for f_name in all_factors:
            is_enabled = f_name in enabled_list
            src_df = df_tr if is_enabled else df_raw_all
            sub = src_df[src_df["factor"] == f_name]
            if len(sub) > 0:
                s_r = sub["r"].values; s_pnl = sub["pnl"].values
                s_w = s_r[s_r > 0]; s_l = s_r[s_r < 0]
                s_r_pf = min((float(s_w.sum()) / float(abs(s_l.sum()))) if len(s_l) > 0 else (2.0 if len(s_w) > 0 else 0.0), 9.99)
                breakdown.append({
                    "factor": f_name, "enabled": is_enabled, "trades": int(len(sub)),
                    "win_rate": round((len(s_w) / len(sub)) * 100.0, 1),
                    "r_profit_factor": round(s_r_pf, 2),
                    "net_pnl": round(float(s_pnl.sum()), 2),
                    "net_r": round(float(s_r.sum()), 2)
                })
            else:
                breakdown.append({
                    "factor": f_name, "enabled": is_enabled, "trades": 0, "win_rate": 0.0, "r_profit_factor": 0.0, "net_pnl": 0.0, "net_r": 0.0
                })

        recent_sim = [
            {k: v for k, v in rec.items() if k not in ("idx", "date")}
            for rec in reversed(records[-12:])
        ]

        return {
            "atr_sl_mult": sl_m, "rr_mult": rr_m, "adx_thresh": adx_t, "enabled_factors": enabled_list,
            "trades": int(len(r_arr)), "win_rate": round(wr, 1), "profit_factor": usd_pf,
            "r_profit_factor": r_pf, "net_pnl": round(net_pnl, 2), "net_r": round(net_r, 2),
            "expectancy_r": inst["expectancy_r"], "max_consec_loss": inst["max_consec_loss"],
            "max_dd": round(max_dd_usd, 2), "max_dd_r": inst["max_dd_r"],
            "sharpe": inst["sharpe"], "sortino": inst["sortino"], "calmar": inst["calmar"],
            "var_95_r": inst["var_95_r"], "mc_dd_95_r": inst["mc_dd_95_r"],
            "mc_prob_profit": inst["mc_prob_profit"], "total_fee_drag": total_fee_drag,
            "oos_trades": oos_trades, "oos_win_rate": oos_wr, "oos_profit_factor": oos_r_pf,
            "oos_net_r": oos_net_r, "oos_net_pnl": oos_pnl,
            "is_profitable_edge": bool(r_pf >= 1.10 and net_r > 0 and len(r_arr) >= 8),
            "score": round(float(train_score), 3),
            "equity_curve_r": [round(float(x), 2) for x in np.insert(cum_r, 0, 0.0)],
            "trade_r_list": [round(float(x), 2) for x in r_arr],
            "factor_breakdown": breakdown,
            "recent_sim_trades": recent_sim
        }

    candidates = []
    seen_sigs = set()
    for sl_m in [1.5, 2.0, 2.5]:
        for rr_m in [2.0, 2.5, 3.2]:
            for adx_t in [22, 25, 28]:
                raw_all = simulate_raw_trades(sl_m, rr_m, adx_t, set(all_factors))
                if len(raw_all) < 6:
                    continue
                df_all = pd.DataFrame(raw_all)
                df_train = df_all[df_all["idx"] < split_idx]

                surviving_factors = []
                for f_name in all_factors:
                    sub_tr = df_train[df_train["factor"] == f_name]
                    if len(sub_tr) == 0 or sub_tr["r"].sum() > 0:
                        surviving_factors.append(f_name)
                if not surviving_factors:
                    surviving_factors = ["VALUE_AREA_FADE", "LIQUIDITY_SWEEP"]

                filtered_records = simulate_raw_trades(sl_m, rr_m, adx_t, set(surviving_factors))
                summary = summarize_records(filtered_records, raw_all, sl_m, rr_m, adx_t, surviving_factors)
                if summary:
                    sig = (summary["trades"], summary["win_rate"], summary["net_r"], summary["oos_net_r"])
                    if sig not in seen_sigs:
                        seen_sigs.add(sig)
                        candidates.append(summary)

    if not candidates:
        return {"best": default_best, "factor_breakdown": [], "top_5": [default_best], "split_bar": split_idx}

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    best_cfg = ranked[0]
    breakdown = best_cfg.pop("factor_breakdown", [])
    top_5_clean = [
        {k: v for k, v in item.items() if k not in ("equity_curve_r", "trade_r_list", "factor_breakdown", "recent_sim_trades")}
        for item in ranked[:5]
    ]

    return {
        "best": best_cfg,
        "factor_breakdown": breakdown,
        "top_5": top_5_clean,
        "split_bar": split_idx
    }

# ================= 2B. VP-ANATOMY 500-POINT ($5.00) SCALP ENGINE (DEDUPLICATED & AUDITABLE) =================
def run_vp_anatomy_scalp_engine(df_intraday, ind, spot_price, spot_offset=0.0, live_muted_shapes=None):
    live_muted_set = set(live_muted_shapes or [])
    open_arr = df_intraday['Open'].values + spot_offset
    high_arr = df_intraday['High'].values + spot_offset
    low_arr = df_intraday['Low'].values + spot_offset
    close_arr = df_intraday['Close'].values + spot_offset
    vol_arr = df_intraday['Volume'].values
    rsi_arr = ind["rsi14"].values
    ema21_arr = ind["ema21"].values
    ema50_arr = ind["ema50"].values
    n_bars = len(df_intraday)

    dt_index_utc = df_intraday.index.tz_localize("UTC") if df_intraday.index.tz is None else df_intraday.index.tz_convert("UTC")
    dt_index_th = dt_index_utc.tz_convert(THAI_TZ)
    utc_hours = dt_index_utc.hour.values
    date_ordinals = dt_index_utc.date

    curr_anatomy = classify_vp_anatomy(
        low_arr[-92:], high_arr[-92:], close_arr[-92:], vol_arr[-92:], bins=32
    )
    if curr_anatomy is None:
        curr_anatomy = {
            "shape": "D_SHAPED",
            "title_th": ANATOMY_META["D_SHAPED"]["title_th"],
            "desc_th": ANATOMY_META["D_SHAPED"]["desc_th"],
            "rule_th": ANATOMY_META["D_SHAPED"]["rule_th"],
            "poc": round(spot_price, 2), "vah": round(spot_price + 8.0, 2), "val": round(spot_price - 8.0, 2),
            "poc_pos_pct": 50.0, "second_peak": None, "valley": None,
            "lower_peak": round(spot_price - 8.0, 2), "upper_peak": round(spot_price + 8.0, 2)
        }

    win_24h = 92
    split_idx = win_24h + int((n_bars - win_24h) * 0.67)
    an_shape = np.empty(n_bars, dtype=object)
    an_poc = np.full(n_bars, np.nan)
    an_vah = np.full(n_bars, np.nan)
    an_val = np.full(n_bars, np.nan)
    an_lo_pk = np.full(n_bars, np.nan)
    an_hi_pk = np.full(n_bars, np.nan)

    for i in range(win_24h, n_bars, 4):
        res_a = classify_vp_anatomy(
            low_arr[i - win_24h:i], high_arr[i - win_24h:i],
            close_arr[i - win_24h:i], vol_arr[i - win_24h:i], bins=28
        )
        end_i = min(i + 4, n_bars)
        if res_a:
            an_shape[i:end_i] = res_a["shape"]
            an_poc[i:end_i] = res_a["poc"]
            an_vah[i:end_i] = res_a["vah"]
            an_val[i:end_i] = res_a["val"]
            an_lo_pk[i:end_i] = res_a["lower_peak"]
            an_hi_pk[i:end_i] = res_a["upper_peak"]

    all_shapes = ["D_SHAPED", "B_SHAPED", "P_SHAPED", "DOUBLE_ZONE"]
    sl_usd = 5.00       # 500 จุด
    be_trigger = 2.50   # +250 จุด เลื่อนบังทุน
    spread_cost = 0.45  # 45 จุด (ECN Raw Spread + Commission)

    def sim_scalp(tp_usd, tol_usd, wick_min, enabled_shapes):
        in_pos = 0
        active_sh = ""
        entry_p = sl_p = tp_p = 0.0
        entry_time_str = ""
        be_moved = False
        cooldown = 0
        recs = []

        for i in range(win_24h + 1, n_bars):
            sh = an_shape[i]
            if sh is None or np.isnan(an_poc[i]):
                continue

            # Pessimistic Fill: ตรวจ SL ก่อน TP เสมอ
            if in_pos == 1:
                if low_arr[i] <= sl_p:
                    pnl = (sl_p - entry_p) - spread_cost
                    recs.append({
                        "idx": i, "engine": "SCALP", "shape": active_sh, "side": "LONG",
                        "entry_time": entry_time_str, "exit_time": dt_index_th[i].strftime("%d/%m %H:%M"),
                        "entry": round(entry_p, 2), "exit": round(sl_p, 2),
                        "pnl": round(pnl, 2), "r": round(pnl / sl_usd, 2), "date": date_ordinals[i]
                    })
                    in_pos = 0; cooldown = 1; continue
                if high_arr[i] >= tp_p:
                    pnl = (tp_p - entry_p) - spread_cost
                    recs.append({
                        "idx": i, "engine": "SCALP", "shape": active_sh, "side": "LONG",
                        "entry_time": entry_time_str, "exit_time": dt_index_th[i].strftime("%d/%m %H:%M"),
                        "entry": round(entry_p, 2), "exit": round(tp_p, 2),
                        "pnl": round(pnl, 2), "r": round(pnl / sl_usd, 2), "date": date_ordinals[i]
                    })
                    in_pos = 0; cooldown = 1; continue
                if not be_moved and high_arr[i] >= entry_p + be_trigger:
                    sl_p = entry_p + 0.45
                    be_moved = True
                continue
            elif in_pos == -1:
                if high_arr[i] >= sl_p:
                    pnl = (entry_p - sl_p) - spread_cost
                    recs.append({
                        "idx": i, "engine": "SCALP", "shape": active_sh, "side": "SHORT",
                        "entry_time": entry_time_str, "exit_time": dt_index_th[i].strftime("%d/%m %H:%M"),
                        "entry": round(entry_p, 2), "exit": round(sl_p, 2),
                        "pnl": round(pnl, 2), "r": round(pnl / sl_usd, 2), "date": date_ordinals[i]
                    })
                    in_pos = 0; cooldown = 1; continue
                if low_arr[i] <= tp_p:
                    pnl = (entry_p - tp_p) - spread_cost
                    recs.append({
                        "idx": i, "engine": "SCALP", "shape": active_sh, "side": "SHORT",
                        "entry_time": entry_time_str, "exit_time": dt_index_th[i].strftime("%d/%m %H:%M"),
                        "entry": round(entry_p, 2), "exit": round(tp_p, 2),
                        "pnl": round(pnl, 2), "r": round(pnl / sl_usd, 2), "date": date_ordinals[i]
                    })
                    in_pos = 0; cooldown = 1; continue
                if not be_moved and low_arr[i] <= entry_p - be_trigger:
                    sl_p = entry_p - 0.45
                    be_moved = True
                continue

            if cooldown > 0:
                cooldown -= 1; continue

            hr = utc_hours[i]
            if not (3 <= hr <= 17):
                continue

            if sh not in enabled_shapes:
                continue

            c_open = open_arr[i]; c_high = high_arr[i]; c_low = low_arr[i]; c_close = close_arr[i]
            c_rng = max(c_high - c_low, 1e-6)
            lo_wick = (min(c_open, c_close) - c_low) / c_rng
            up_wick = (c_high - max(c_open, c_close)) / c_rng
            bull_c = c_close > c_open and (lo_wick >= wick_min or c_close > close_arr[i - 1])
            bear_c = c_close < c_open and (up_wick >= wick_min or c_close < close_arr[i - 1])
            rsi = rsi_arr[i]
            poc = an_poc[i]; vah = an_vah[i]; val = an_val[i]
            lo_pk = an_lo_pk[i]; hi_pk = an_hi_pk[i]
            t_str = dt_index_th[i].strftime("%d/%m %H:%M")

            if sh == "D_SHAPED":
                if c_low <= val + tol_usd and c_close >= val - 1.5 and bull_c and rsi <= 52:
                    in_pos = 1; active_sh = "D_SHAPED"; be_moved = False; entry_time_str = t_str
                    entry_p = c_close; sl_p = entry_p - sl_usd; tp_p = entry_p + tp_usd
                elif c_high >= vah - tol_usd and c_close <= vah + 1.5 and bear_c and rsi >= 48:
                    in_pos = -1; active_sh = "D_SHAPED"; be_moved = False; entry_time_str = t_str
                    entry_p = c_close; sl_p = entry_p + sl_usd; tp_p = entry_p - tp_usd

            elif sh == "B_SHAPED":
                if c_high >= poc - tol_usd and c_close < poc and bear_c and c_close < ema21_arr[i] and rsi >= 30:
                    in_pos = -1; active_sh = "B_SHAPED"; be_moved = False; entry_time_str = t_str
                    entry_p = c_close; sl_p = entry_p + sl_usd; tp_p = entry_p - tp_usd
                elif c_close > vah and close_arr[i - 1] <= vah and bull_c and rsi <= 70:
                    in_pos = 1; active_sh = "B_SHAPED"; be_moved = False; entry_time_str = t_str
                    entry_p = c_close; sl_p = entry_p - sl_usd; tp_p = entry_p + tp_usd

            elif sh == "P_SHAPED":
                if ((c_low <= poc + tol_usd and c_close >= poc - 1.0) or (c_low <= val + tol_usd and c_close >= val - 1.0)) and bull_c and c_close > ema50_arr[i] and rsi <= 68:
                    in_pos = 1; active_sh = "P_SHAPED"; be_moved = False; entry_time_str = t_str
                    entry_p = c_close; sl_p = entry_p - sl_usd; tp_p = entry_p + tp_usd
                elif c_close < val and close_arr[i - 1] >= val and bear_c and rsi >= 32:
                    in_pos = -1; active_sh = "P_SHAPED"; be_moved = False; entry_time_str = t_str
                    entry_p = c_close; sl_p = entry_p + sl_usd; tp_p = entry_p - tp_usd

            elif sh == "DOUBLE_ZONE":
                if c_low <= lo_pk + tol_usd and c_close >= lo_pk - 1.5 and bull_c and rsi <= 55:
                    in_pos = 1; active_sh = "DOUBLE_ZONE"; be_moved = False; entry_time_str = t_str
                    entry_p = c_close; sl_p = entry_p - sl_usd; tp_p = entry_p + tp_usd
                elif c_high >= hi_pk - tol_usd and c_close <= hi_pk + 1.5 and bear_c and rsi >= 45:
                    in_pos = -1; active_sh = "DOUBLE_ZONE"; be_moved = False; entry_time_str = t_str
                    entry_p = c_close; sl_p = entry_p + sl_usd; tp_p = entry_p - tp_usd

        return recs

    def build_scalp_candidate(tp_cand, tol_cand, wick_cand, raw_s, surv, strict_mode=True):
        df_s = pd.DataFrame(raw_s)
        filt_s = sim_scalp(tp_cand, tol_cand, wick_cand, set(surv))
        if len(filt_s) < 5:
            return None
        df_f = pd.DataFrame(filt_s)
        r_arr = df_f["r"].values
        pnl_arr = df_f["pnl"].values
        w_r = r_arr[r_arr > 0]; l_r = r_arr[r_arr < 0]
        wr = round((len(w_r) / len(r_arr)) * 100.0, 1)
        r_pf = round(min((float(w_r.sum()) / float(abs(l_r.sum()))) if len(l_r) > 0 else 2.0, 9.99), 2)
        net_r = round(float(r_arr.sum()), 2)
        net_pnl = round(float(pnl_arr.sum()), 2)
        cum_r = np.cumsum(r_arr)

        inst = compute_institutional_risk_metrics(df_f, date_ordinals[win_24h:])

        oos_df = df_f[df_f["idx"] >= split_idx]
        if len(oos_df) > 0:
            o_r = oos_df["r"].values
            o_w = o_r[o_r > 0]; o_l = o_r[o_r < 0]
            oos_wr = round((len(o_w) / len(o_r)) * 100.0, 1)
            oos_pf = round(min((float(o_w.sum()) / float(abs(o_l.sum()))) if len(o_l) > 0 else (2.0 if len(o_w) > 0 else 0.0), 9.99), 2)
            oos_net_r = round(float(o_r.sum()), 2)
            oos_trades = int(len(oos_df))
        else:
            oos_wr, oos_pf, oos_net_r, oos_trades = 0.0, 0.0, 0.0, 0

        df_is = df_f[df_f["idx"] < split_idx]
        is_r = df_is["r"].values if len(df_is) > 0 else r_arr
        sc_score = (float(is_r.sum()) + 5.0) * (r_pf ** 0.5) / (1.0 + inst["max_dd_r"] * 0.15)
        if not strict_mode:
            sc_score -= 100.0

        anatomy_breakdown = []
        for s_name in all_shapes:
            is_on = (s_name in surv) and (s_name not in live_muted_set)
            src_df = df_f if (s_name in surv) else df_s
            sub = src_df[src_df["shape"] == s_name]
            if len(sub) > 0:
                sr = sub["r"].values
                sw = sr[sr > 0]
                anatomy_breakdown.append({
                    "shape": s_name,
                    "enabled": is_on,
                    "live_learned_mute": s_name in live_muted_set,
                    "trades": int(len(sub)),
                    "win_rate": round((len(sw) / len(sub)) * 100.0, 1),
                    "net_r": round(float(sr.sum()), 2)
                })
            else:
                anatomy_breakdown.append({
                    "shape": s_name, "enabled": is_on, "live_learned_mute": s_name in live_muted_set,
                    "trades": 0, "win_rate": 0.0, "net_r": 0.0
                })

        recent_sim = [
            {k: v for k, v in rec.items() if k not in ("idx", "date")}
            for rec in reversed(filt_s[-12:])
        ]

        return {
            "sl_usd": sl_usd, "sl_points": 500,
            "tp_usd": tp_cand, "tp_points": int(round(tp_cand * 100)),
            "tol_points": int(round(tol_cand * 100)),
            "wick_mode": "STRICT" if wick_cand >= 0.25 else "NORMAL",
            "be_usd": be_trigger, "be_points": 250,
            "enabled_shapes": [s for s in surv if s not in live_muted_set],
            "trades": int(len(r_arr)), "win_rate": wr,
            "r_profit_factor": r_pf, "net_r": net_r, "net_pnl": net_pnl,
            "expectancy_r": inst["expectancy_r"], "max_consec_loss": inst["max_consec_loss"],
            "max_dd_r": inst["max_dd_r"], "sharpe": inst["sharpe"], "sortino": inst["sortino"],
            "mc_dd_95_r": inst["mc_dd_95_r"], "mc_prob_profit": inst["mc_prob_profit"],
            "oos_trades": oos_trades, "oos_win_rate": oos_wr,
            "oos_r_pf": oos_pf, "oos_net_r": oos_net_r,
            "has_edge": bool(r_pf >= 1.10 and net_r > 0 and len(r_arr) >= 8),
            "score": round(sc_score, 3),
            "equity_curve_r": [round(float(x), 2) for x in np.insert(cum_r, 0, 0.0)],
            "anatomy_breakdown": anatomy_breakdown,
            "recent_sim_trades": recent_sim
        }

    scalp_candidates = []
    fallback_candidates = []
    seen_scalp_sigs = set()

    for tp_cand in [4.00, 5.00, 6.00, 7.50]:
        for tol_cand in [1.20, 2.20, 3.20]:
            for wick_cand in [0.15, 0.28]:
                raw_s = sim_scalp(tp_cand, tol_cand, wick_cand, set(all_shapes))
                if len(raw_s) < 6:
                    continue
                df_s = pd.DataFrame(raw_s)
                df_tr = df_s[df_s["idx"] < split_idx]

                surv = []
                shape_scores = []
                for s in all_shapes:
                    sub_tr = df_tr[df_tr["shape"] == s]
                    tr_r = float(sub_tr["r"].sum()) if len(sub_tr) > 0 else -99.0
                    shape_scores.append((s, tr_r, len(sub_tr)))
                    if len(sub_tr) > 0 and tr_r > 0:
                        surv.append(s)

                if surv:
                    cand = build_scalp_candidate(tp_cand, tol_cand, wick_cand, raw_s, surv, strict_mode=True)
                    if cand:
                        sig = (cand["tp_points"], cand["trades"], cand["win_rate"], cand["net_r"])
                        if sig not in seen_scalp_sigs:
                            seen_scalp_sigs.add(sig)
                            scalp_candidates.append(cand)
                else:
                    shape_scores.sort(key=lambda x: x[1], reverse=True)
                    cand_fb = build_scalp_candidate(tp_cand, tol_cand, wick_cand, raw_s, [shape_scores[0][0]], strict_mode=False)
                    if cand_fb:
                        sig = (cand_fb["tp_points"], cand_fb["trades"], cand_fb["win_rate"], cand_fb["net_r"])
                        if sig not in seen_scalp_sigs:
                            seen_scalp_sigs.add(sig)
                            fallback_candidates.append(cand_fb)

    chosen_pool = scalp_candidates if scalp_candidates else fallback_candidates
    if chosen_pool:
        scalp_ranked = sorted(chosen_pool, key=lambda x: x["score"], reverse=True)
        scalp_bt = scalp_ranked[0]
        scalp_bt["top_5"] = [
            {k: v for k, v in item.items() if k not in ("equity_curve_r", "anatomy_breakdown", "recent_sim_trades")}
            for item in scalp_ranked[:5]
        ]
    else:
        scalp_bt = {
            "sl_usd": 5.0, "sl_points": 500, "tp_usd": 5.0, "tp_points": 500,
            "tol_points": 200, "wick_mode": "NORMAL",
            "be_usd": 2.5, "be_points": 250, "enabled_shapes": [],
            "trades": 0, "win_rate": 0.0, "expectancy_r": 0.0, "max_consec_loss": 0,
            "r_profit_factor": 0.0, "net_r": 0.0, "net_pnl": 0.0, "max_dd_r": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "mc_dd_95_r": 0.0, "mc_prob_profit": 0.0,
            "oos_trades": 0, "oos_win_rate": 0.0, "oos_r_pf": 0.0, "oos_net_r": 0.0,
            "has_edge": False, "equity_curve_r": [0], "anatomy_breakdown": [], "top_5": [], "recent_sim_trades": []
        }

    sh = curr_anatomy["shape"]
    poc = curr_anatomy["poc"]
    vah = curr_anatomy["vah"]
    val = curr_anatomy["val"]
    lo_pk = curr_anatomy["lower_peak"]
    hi_pk = curr_anatomy["upper_peak"]
    tp_usd = scalp_bt["tp_usd"]
    tp_pts = scalp_bt["tp_points"]
    enabled_shapes = set(scalp_bt.get("enabled_shapes", []))
    shape_is_active = (sh in enabled_shapes) and (sh not in live_muted_set)

    if sh == "D_SHAPED":
        if abs(spot_price - val) <= abs(spot_price - vah):
            s_dir = "LONG"; s_entry = val
            s_bias = f"D-SHAPE VAL BOUNCE BUY (ดักเด้งแนวรับ VAL ${val} เป้าหมาย POC ${poc})"
        else:
            s_dir = "SHORT"; s_entry = vah
            s_bias = f"D-SHAPE VAH REJECT SELL (ดักขายแนวต้าน VAH ${vah} เป้าหมาย POC ${poc})"
    elif sh == "B_SHAPED":
        if spot_price <= poc + 2.0:
            s_dir = "SHORT"; s_entry = poc
            s_bias = f"b-SHAPE POC BREAKDOWN SELL (กดขายใต้ฐานโวลุ่ม POC ${poc})"
        else:
            s_dir = "LONG"; s_entry = vah
            s_bias = f"b-SHAPE VAH BREAKOUT BUY (สะสมฐานล่างเสร็จ เบรก VAH ${vah})"
    elif sh == "P_SHAPED":
        if spot_price >= val - 1.5:
            s_dir = "LONG"; s_entry = poc if spot_price >= poc - 1.5 else val
            s_bias = f"P-SHAPE PULLBACK BUY (ย่อรับตามโมเมนตัมขาขึ้นที่ ${s_entry})"
        else:
            s_dir = "SHORT"; s_entry = val
            s_bias = f"P-SHAPE VAL BREAKDOWN SELL (เสียทรงหลุดกรอบล่าง VAL ${val})"
    else:
        if abs(spot_price - lo_pk) <= abs(spot_price - hi_pk):
            s_dir = "LONG"; s_entry = lo_pk
            s_bias = f"DOUBLE-ZONE LOWER PEAK BUY (ดักเด้งยอดโวลุ่มล่าง ${lo_pk})"
        else:
            s_dir = "SHORT"; s_entry = hi_pk
            s_bias = f"DOUBLE-ZONE UPPER PEAK SELL (ดักขายยอดโวลุ่มบน ${hi_pk})"

    s_sl = round(s_entry - sl_usd, 2) if s_dir == "LONG" else round(s_entry + sl_usd, 2)
    s_be = round(s_entry + be_trigger, 2) if s_dir == "LONG" else round(s_entry - be_trigger, 2)
    s_tp = round(s_entry + tp_usd, 2) if s_dir == "LONG" else round(s_entry - tp_usd, 2)
    dist_usd = round(abs(spot_price - s_entry), 2)
    dist_pts = int(round(dist_usd * 100))

    now_th = datetime.now(THAI_TZ)
    in_scalp_session = (10 <= now_th.hour <= 23) or (now_th.hour == 0 and now_th.minute <= 30)
    gate_session = bool(in_scalp_session)
    gate_edge = bool(scalp_bt["has_edge"] and shape_is_active)
    gate_prox = bool(dist_usd <= 3.00)
    scalp_ready = gate_session and gate_edge and gate_prox

    if not gate_session:
        s_verdict = "NO (อยู่นอกเวลา Scalp 10:00–00:30 น.)"
    elif sh in live_muted_set:
        s_verdict = f"NO (Self-Learning Guard สั่ง MUTE ทรง {sh} ชั่วคราวหลังแพ้ต่อเนื่อง)"
    elif not shape_is_active:
        active_str = ", ".join(enabled_shapes) if enabled_shapes else "ไม่มี"
        s_verdict = f"NO (ทรง {sh} ถูก MUTE จาก Backtest — รอทรง {active_str})"
    elif not scalp_bt["has_edge"]:
        s_verdict = "NO (สถิติ Scalp R-PF ต่ำกว่า 1.10 — รอจังหวะถัดไป)"
    elif not gate_prox:
        s_verdict = f"NO (ห่างจุดเข้า ${dist_usd:.2f} / {dist_pts} จุด — รอเข้าใกล้ <= 300 จุด)"
    else:
        s_verdict = f"YES (พร้อมยิง SCALP 500 จุด! ทรง {sh} ผ่านเกณฑ์และเข้าใกล้ {dist_pts} จุด)"

    return {
        "anatomy": curr_anatomy,
        "backtest": scalp_bt,
        "setup": {
            "shape": sh,
            "shape_active": shape_is_active,
            "bias": s_bias,
            "direction": s_dir,
            "entry": round(s_entry, 2),
            "sl": s_sl,
            "sl_usd": sl_usd,
            "sl_points": 500,
            "be_target": s_be,
            "be_points": 250,
            "tp": s_tp,
            "tp_usd": tp_usd,
            "tp_points": tp_pts,
            "rr": round(tp_usd / sl_usd, 2),
            "dist_usd": dist_usd,
            "dist_points": dist_pts
        },
        "gates": {
            "session_ok": gate_session,
            "shape_ok": shape_is_active,
            "edge_ok": gate_edge,
            "proximity_ok": gate_prox,
            "all_passed": scalp_ready,
            "verdict_text": s_verdict
        }
    }

# ================= 3. SESSION-AWARE SPOT TREND ENGINE =================
def calculate_quant_and_setup(df_intraday, df_daily, df_dxy, df_us10y, vp_data, ind, spot_price, bid, ask, feed_source, spot_offset, open_interest, tuned_params, live_muted_factors=None):
    live_muted_f_set = set(live_muted_factors or [])
    futures_price = float(df_intraday['Close'].iloc[-1])
    active_spot = round(spot_price, 2) if spot_price else round(futures_price, 2)
    
    mean_20_spot = float(df_daily['Close'].rolling(20).mean().iloc[-1]) + spot_offset
    std_20 = float(df_daily['Close'].rolling(20).std().iloc[-1])
    z_price = float((active_spot - mean_20_spot) / std_20) if std_20 > 0 else 0.0
    
    completed_vol = df_daily['Volume'].iloc[:-1]
    vol_mean_20 = completed_vol.rolling(20).mean().iloc[-1]
    vol_std_20 = completed_vol.rolling(20).std().iloc[-1]
    last_completed_vol = float(completed_vol.iloc[-1])
    z_vol = float((last_completed_vol - vol_mean_20) / vol_std_20) if vol_std_20 > 0 else 0.0
    
    log_hl = np.log(df_daily['High'] / df_daily['Low']) ** 2
    parkinson_vol = float(np.sqrt(log_hl.tail(14).mean() / (4 * np.log(2))))
    expected_move = round(float(active_spot * parkinson_vol), 2)
    
    recent = df_intraday.tail(24).copy()
    spread = (recent['High'] - recent['Low']).replace(0, 1e-6)
    delta = recent['Volume'] * ((recent['Close'] - recent['Open']) / spread)
    cvd_6h = float(delta.sum())
    
    combined = pd.DataFrame({
        "gold": df_daily['Close'].pct_change(),
        "dxy": df_dxy['Close'].pct_change()
    }).dropna().tail(20)
    corr_dxy = float(combined['gold'].corr(combined['dxy'])) if len(combined) > 5 else 0.0
    dxy_latest = round(float(df_dxy['Close'].iloc[-1]), 2) if not df_dxy.empty else 0
    us10y_latest = round(float(df_us10y['Close'].iloc[-1]), 2) if not df_us10y.empty else 0

    poc, vah, val = vp_data['poc'], vp_data['vah'], vp_data['val']
    curr_ema21 = round(float(ind["ema21"].iloc[-1]), 2)
    curr_ema50 = round(float(ind["ema50"].iloc[-1]), 2)
    curr_ema200 = round(float(ind["ema200"].iloc[-1]), 2)
    curr_h1_ema50 = round(float(ind["h1_ema50"].iloc[-1]), 2)
    curr_h1_ema200 = round(float(ind["h1_ema200"].iloc[-1]), 2)
    curr_atr14 = round(float(ind["atr14"].iloc[-1]), 2)
    curr_rsi14 = round(float(ind["rsi14"].iloc[-1]), 1)
    curr_adx14 = round(float(ind["adx14"].iloc[-1]), 1)
    curr_pdh = round(float(ind["pdh"].iloc[-1]), 2)
    curr_pdl = round(float(ind["pdl"].iloc[-1]), 2)
    curr_asian_h = round(float(ind["asian_high"].iloc[-1]), 2)
    curr_asian_l = round(float(ind["asian_low"].iloc[-1]), 2)

    atr_sl_mult = float(tuned_params.get("atr_sl_mult", 1.6))
    rr_mult = float(tuned_params.get("rr_mult", 2.2))
    adx_thresh = float(tuned_params.get("adx_thresh", 25))
    has_edge = bool(tuned_params.get("is_profitable_edge", False))
    enabled_factors = set(tuned_params.get("enabled_factors", ["VALUE_AREA_FADE", "LIQUIDITY_SWEEP", "LONDON_BREAKOUT"])) - live_muted_f_set

    risk_unit = round(curr_atr14 * atr_sl_mult, 2)
    be_trigger_dist = round(risk_unit * 1.0, 2)
    reward_unit = round(risk_unit * rr_mult, 2)

    now_th = datetime.now(THAI_TZ)
    in_killzone = (13 <= now_th.hour <= 23) or (now_th.hour == 0 and now_th.minute <= 30)
    in_london_window = (13 <= now_th.hour <= 16)
    session_name = "LONDON / NY KILLZONE ACTIVE" if in_killzone else "NO TRADE ZONE (รอ 13:00 น.)"

    rsi_oversold_exhausted = curr_rsi14 < 28.0
    rsi_overbought_exhausted = curr_rsi14 > 72.0

    last_low = float(df_intraday['Low'].iloc[-1]) + spot_offset
    last_high = float(df_intraday['High'].iloc[-1]) + spot_offset
    near_pdl_sweep = (last_low <= curr_pdl + curr_atr14 * 0.25) and (active_spot >= curr_pdl)
    near_pdh_sweep = (last_high >= curr_pdh - curr_atr14 * 0.25) and (active_spot <= curr_pdh)

    if "LIQUIDITY_SWEEP" in enabled_factors and near_pdl_sweep:
        active_factor = "LIQUIDITY_SWEEP"
        regime = f"LIQUIDITY SWEEP AT PDL (${curr_pdl}) | RSI {curr_rsi14}"
        bias = "LIQUIDITY SWEEP LONG (กวาดสภาพคล่องใต้ PDL แล้วดึงกลับ)"
        direction = "LONG"
        entry = round(max(active_spot, curr_pdl), 2)
        sl = round(entry - risk_unit, 2)
        be_target = round(entry + be_trigger_dist, 2)
        tp = round(entry + reward_unit, 2)
    elif "LIQUIDITY_SWEEP" in enabled_factors and near_pdh_sweep:
        active_factor = "LIQUIDITY_SWEEP"
        regime = f"LIQUIDITY SWEEP AT PDH (${curr_pdh}) | RSI {curr_rsi14}"
        bias = "LIQUIDITY SWEEP SHORT (กวาดสภาพคล่องเหนือ PDH แล้วกดกลับ)"
        direction = "SHORT"
        entry = round(min(active_spot, curr_pdh), 2)
        sl = round(entry + risk_unit, 2)
        be_target = round(entry - be_trigger_dist, 2)
        tp = round(entry - reward_unit, 2)
    elif curr_adx14 > adx_thresh and active_spot < curr_h1_ema200 and active_spot < curr_ema200:
        if rsi_oversold_exhausted:
            active_factor = "ANTI_CHASE_RETEST"
            regime = f"OVERSOLD EXHAUSTION (ADX {curr_adx14} | RSI {curr_rsi14} < 28)"
            bias = f"ANTI-CHASE RETEST SHORT (ห้ามไล่ Short ใต้ PDL — รอเด้งชน PDL ${curr_pdl} / EMA21 ${curr_ema21})"
            direction = "SHORT"
            retest_lvl = curr_pdl if curr_pdl > active_spot else curr_ema21; retest_lbl = f"PDL ${curr_pdl}" if curr_pdl > active_spot else f"EMA21 ${curr_ema21}"; bias = f"ANTI-CHASE RETEST SHORT (ห้ามไล่ Short ใต้ PDL — รอเด้งชนแนวต้าน {retest_lbl})"; entry = round(retest_lvl, 2)
            sl = round(entry + risk_unit, 2)
            be_target = round(entry - be_trigger_dist, 2)
            tp = round(entry - reward_unit, 2)
        elif "LONDON_BREAKOUT" in enabled_factors and in_london_window and active_spot >= curr_asian_l:
            active_factor = "LONDON_BREAKOUT"
            regime = f"LONDON BREAKOUT BEAR (ADX {curr_adx14} | Asian Low ${curr_asian_l})"
            bias = f"LONDON BREAKDOWN SHORT (รอเบรกหลุด Asian Low ${curr_asian_l})"
            direction = "SHORT"
            entry = round(curr_asian_l - 0.5, 2)
            sl = round(entry + risk_unit, 2)
            be_target = round(entry - be_trigger_dist, 2)
            tp = round(entry - reward_unit, 2)
        else:
            active_factor = "MOMENTUM_BREAKOUT" if "MOMENTUM_BREAKOUT" in enabled_factors else "LIQUIDITY_SWEEP"
            regime = f"STRONG BEARISH EXPANSION (ADX {curr_adx14} | H1+M15 ใต้ EMA200)"
            if active_spot > curr_pdl:
                bias = f"MOMENTUM BREAKDOWN SHORT (รอเบรกหลุด PDL ${curr_pdl} ตามน้ำ)"
                entry = round(curr_pdl - 0.5, 2)
            else:
                bias = f"MOMENTUM RETEST SHORT (รอเด้งทดสอบแนวต้าน PDL ${curr_pdl} / EMA21 ${curr_ema21})"
                entry = round(curr_pdl, 2)
            direction = "SHORT"
            sl = round(entry + risk_unit, 2)
            be_target = round(entry - be_trigger_dist, 2)
            tp = round(entry - reward_unit, 2)
    elif curr_adx14 > adx_thresh and active_spot > curr_h1_ema200 and active_spot > curr_ema200:
        if rsi_overbought_exhausted:
            active_factor = "ANTI_CHASE_RETEST"
            regime = f"OVERBOUGHT EXHAUSTION (ADX {curr_adx14} | RSI {curr_rsi14} > 72)"
            bias = f"ANTI-CHASE RETEST LONG (ห้ามไล่ Buy เหนือ PDH — รอย่อแตะ PDH ${curr_pdh} / EMA21 ${curr_ema21})"
            direction = "LONG"
            retest_lvl = curr_pdh if curr_pdh < active_spot else curr_ema21
            entry = round(retest_lvl, 2)
            sl = round(entry - risk_unit, 2)
            be_target = round(entry + be_trigger_dist, 2)
            tp = round(entry + reward_unit, 2)
        elif "LONDON_BREAKOUT" in enabled_factors and in_london_window and active_spot <= curr_asian_h:
            active_factor = "LONDON_BREAKOUT"
            regime = f"LONDON BREAKOUT BULL (ADX {curr_adx14} | Asian High ${curr_asian_h})"
            bias = f"LONDON BREAKOUT LONG (รอเบรกทะลุ Asian High ${curr_asian_h})"
            direction = "LONG"
            entry = round(curr_asian_h + 0.5, 2)
            sl = round(entry - risk_unit, 2)
            be_target = round(entry + be_trigger_dist, 2)
            tp = round(entry + reward_unit, 2)
        else:
            active_factor = "MOMENTUM_BREAKOUT" if "MOMENTUM_BREAKOUT" in enabled_factors else "LIQUIDITY_SWEEP"
            regime = f"STRONG BULLISH EXPANSION (ADX {curr_adx14} | H1+M15 เหนือ EMA200)"
            if active_spot < curr_pdh:
                bias = f"MOMENTUM BREAKOUT LONG (รอเบรกทะลุ PDH ${curr_pdh} ตามน้ำ)"
                entry = round(curr_pdh + 0.5, 2)
            else:
                bias = f"MOMENTUM RETEST LONG (รอย่อทดสอบแนวรับ PDH ${curr_pdh} / EMA21 ${curr_ema21})"
                entry = round(curr_pdh, 2)
            direction = "LONG"
            sl = round(entry - risk_unit, 2)
            be_target = round(entry + be_trigger_dist, 2)
            tp = round(entry + reward_unit, 2)
    else:
        if "VALUE_AREA_FADE" in enabled_factors and curr_adx14 <= adx_thresh:
            active_factor = "VALUE_AREA_FADE"
            regime = f"BALANCED RANGE (ADX {curr_adx14} <= {adx_thresh})"
            if active_spot >= poc:
                bias = "RANGE FADE SHORT NEAR VAH (ขายกรอบบน เป้าหมาย POC)"
                direction = "SHORT"
                entry = vah
                sl = round(vah + risk_unit, 2)
                be_target = round(vah - be_trigger_dist, 2)
                tp = round(vah - reward_unit, 2)
            else:
                bias = "RANGE FADE LONG NEAR VAL (ซื้อกรอบล่าง เป้าหมาย POC)"
                direction = "LONG"
                entry = val
                sl = round(val - risk_unit, 2)
                be_target = round(val + be_trigger_dist, 2)
                tp = round(val + reward_unit, 2)
        else:
            active_factor = "LIQUIDITY_SWEEP"
            regime = f"MTF DIVERGENCE / LIQUIDITY WATCH (PDH ${curr_pdh} / PDL ${curr_pdl})"
            if abs(active_spot - curr_pdh) < abs(active_spot - curr_pdl):
                bias = f"WATCH PDH SWEEP SHORT (ดักกวาดสภาพคล่องเหนือ PDH ${curr_pdh})"
                direction = "SHORT"
                entry = curr_pdh
                sl = round(entry + risk_unit, 2)
                be_target = round(entry - be_trigger_dist, 2)
                tp = round(entry - reward_unit, 2)
            else:
                bias = f"WATCH PDL SWEEP LONG (ดักกวาดสภาพคล่องใต้ PDL ${curr_pdl})"
                direction = "LONG"
                entry = curr_pdl
                sl = round(entry - risk_unit, 2)
                be_target = round(entry + be_trigger_dist, 2)
                tp = round(entry + reward_unit, 2)

    sl_dist = round(abs(entry - sl), 2)
    tp_dist = round(abs(tp - entry), 2)
    rr_ratio = round(tp_dist / max(sl_dist, 0.1), 2)
    dist_to_entry = round(abs(active_spot - entry), 2)

    factor_live_muted = active_factor in live_muted_f_set
    gate_session = bool(in_killzone)
    gate_edge = bool(has_edge and not factor_live_muted)
    gate_proximity = bool(dist_to_entry <= 8.0)
    all_gates_passed = gate_session and gate_edge and gate_proximity

    if not gate_session:
        gate_verdict = "NO (อยู่นอกเวลา Killzone — รอปลดล็อก 13:00 น.)"
    elif factor_live_muted:
        gate_verdict = f"NO (Self-Learning Guard สั่ง MUTE {active_factor} ชั่วคราวหลังแพ้ต่อเนื่อง)"
    elif not gate_edge:
        gate_verdict = f"NO (สถิติ Walk-Forward R-PF ต่ำกว่าเกณฑ์ 1.10)"
    elif not gate_proximity:
        gate_verdict = f"NO (ราคา Spot ปัจจุบันอยู่ห่าง Entry ${dist_to_entry:.2f} > $8.00)"
    else:
        gate_verdict = f"YES (ครบทั้ง 3 เงื่อนไข: Killzone + Quant Edge + เข้าใกล้ Entry ${dist_to_entry:.2f})"

    start_dt_str = df_intraday.index[0].astimezone(THAI_TZ).strftime("%d/%m/%Y")
    end_dt_str = df_intraday.index[-1].astimezone(THAI_TZ).strftime("%d/%m/%Y %H:%M")
    vault_days = round(len(df_intraday) / 92.0, 1)

    return {
        "spot_price": active_spot,
        "bid": bid if bid else active_spot,
        "ask": ask if ask else active_spot,
        "feed_source": feed_source,
        "futures_price": round(futures_price, 2),
        "spot_offset": round(spot_offset, 2),
        "data_audit": {
            "source": f"Hydra Local Vault + {feed_source}",
            "total_bars": int(len(df_intraday)),
            "vault_days": vault_days,
            "date_range": f"{start_dt_str} - {end_dt_str}",
            "spot_offset": round(spot_offset, 2)
        },
        "mean_20_spot": round(mean_20_spot, 2),
        "std_20": round(std_20, 2),
        "cme_open_interest": int(open_interest) if open_interest else "N/A",
        "z_score_price": round(z_price, 2),
        "z_score_volume": round(z_vol, 2),
        "parkinson_vol_pct": round(parkinson_vol * 100, 2),
        "expected_move_usd": expected_move,
        "cvd_6h_proxy": round(cvd_6h, 0),
        "dxy_price": dxy_latest,
        "us10y_yield": us10y_latest,
        "corr_gold_dxy_20d": round(corr_dxy, 2),
        "technicals": {
            "ema21": curr_ema21, "ema50": curr_ema50, "ema200": curr_ema200,
            "h1_ema50": curr_h1_ema50, "h1_ema200": curr_h1_ema200,
            "atr14": curr_atr14, "rsi14": curr_rsi14, "adx14": curr_adx14,
            "pdh": curr_pdh, "pdl": curr_pdl,
            "asian_high": curr_asian_h, "asian_low": curr_asian_l,
            "rsi_exhausted": bool(rsi_oversold_exhausted or rsi_overbought_exhausted),
            "in_killzone": in_killzone, "session_name": session_name
        },
        "tuned_params": {
            "atr_sl_mult": atr_sl_mult, "rr_mult": rr_mult, "adx_thresh": adx_thresh,
            "enabled_factors": list(enabled_factors), "is_profitable_edge": has_edge
        },
        "gates": {
            "session_ok": gate_session, "edge_ok": gate_edge, "proximity_ok": gate_proximity,
            "all_passed": all_gates_passed, "dist_to_entry": dist_to_entry, "verdict_text": gate_verdict
        },
        "setup": {
            "active_factor": active_factor, "regime": regime, "bias": bias, "direction": direction,
            "entry": entry, "sl": sl, "be_target": be_target, "tp": tp,
            "sl_dist": sl_dist, "tp_dist": tp_dist, "rr": rr_ratio,
            "has_edge": has_edge, "in_killzone": in_killzone, "session_name": session_name
        }
    }

# ================= 4. LIVE LIFECYCLE JOURNALS + INSTANT NTFY OUTCOME ALERTS =================
def _process_single_journal(journal_path, mode_label, setup_dict, is_triggered, current_spot, df_intraday, spot_offset, spread_cost=0.60, dup_tol=8.0, factor_key="active_factor"):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(journal_path):
        try:
            with open(journal_path, "r", encoding="utf-8") as f:
                journal = json.load(f)
        except Exception:
            journal = []
    else:
        journal = []

    now_utc = datetime.now(timezone.utc)
    now_th_str = now_utc.astimezone(THAI_TZ).strftime("%d/%m %H:%M")
    lifecycle_events = []

    df_spot = df_intraday.tail(96).copy()
    bar_times = df_spot.index.tz_localize("UTC") if df_spot.index.tz is None else df_spot.index.tz_convert("UTC")
    bar_highs = (df_spot["High"] + spot_offset).values
    bar_lows = (df_spot["Low"] + spot_offset).values

    for trade in journal:
        if trade["status"] != "OPEN":
            continue
        opened_dt = datetime.fromisoformat(trade["opened_at"])
        direction = trade["direction"]
        entry_p = float(trade["entry"])
        sl_p = float(trade["sl"])
        tp_p = float(trade["tp"])
        be_p = float(trade.get("be_target", (entry_p + (tp_p - entry_p) * 0.45)))

        sub_highs = [current_spot]
        sub_lows = [current_spot]
        for idx, b_dt in enumerate(bar_times):
            if b_dt >= opened_dt:
                sub_highs.append(float(bar_highs[idx]))
                sub_lows.append(float(bar_lows[idx]))

        max_h = max(sub_highs)
        min_l = min(sub_lows)

        if direction == "LONG":
            # เช็ก SL ก่อน TP (Pessimistic Fill)
            if min_l <= sl_p:
                trade["status"] = "LOSS"
                trade["pnl"] = round((sl_p - entry_p) - spread_cost, 2)
                trade["closed_at_th"] = now_th_str
                lifecycle_events.append({"event": "SL_LOSS", "mode": mode_label, "trade": dict(trade)})
            elif max_h >= tp_p:
                trade["status"] = "WIN"
                trade["pnl"] = round((tp_p - entry_p) - spread_cost, 2)
                trade["closed_at_th"] = now_th_str
                lifecycle_events.append({"event": "TP_WIN", "mode": mode_label, "trade": dict(trade)})
            elif not trade.get("be_hit") and max_h >= be_p:
                trade["be_hit"] = True
                lifecycle_events.append({"event": "BE_HIT", "mode": mode_label, "trade": dict(trade)})

        elif direction == "SHORT":
            if max_h >= sl_p:
                trade["status"] = "LOSS"
                trade["pnl"] = round((entry_p - sl_p) - spread_cost, 2)
                trade["closed_at_th"] = now_th_str
                lifecycle_events.append({"event": "SL_LOSS", "mode": mode_label, "trade": dict(trade)})
            elif min_l <= tp_p:
                trade["status"] = "WIN"
                trade["pnl"] = round((entry_p - tp_p) - spread_cost, 2)
                trade["closed_at_th"] = now_th_str
                lifecycle_events.append({"event": "TP_WIN", "mode": mode_label, "trade": dict(trade)})
            elif not trade.get("be_hit") and min_l <= be_p:
                trade["be_hit"] = True
                lifecycle_events.append({"event": "BE_HIT", "mode": mode_label, "trade": dict(trade)})

    if is_triggered:
        has_duplicate = any(
            t["status"] == "OPEN"
            and t["direction"] == setup_dict["direction"]
            and abs(float(t["entry"]) - float(setup_dict["entry"])) < dup_tol
            for t in journal
        )
        if not has_duplicate:
            new_sig = {
                "id": f"SIG-{now_utc.strftime('%m%d%H%M')}",
                "opened_at": now_utc.isoformat(),
                "opened_at_th": now_th_str,
                "factor": setup_dict.get(factor_key, "SETUP"),
                "direction": setup_dict["direction"],
                "bias": setup_dict["bias"],
                "entry": setup_dict["entry"],
                "sl": setup_dict["sl"],
                "be_target": setup_dict.get("be_target", setup_dict["entry"]),
                "tp": setup_dict["tp"],
                "rr": setup_dict["rr"],
                "be_hit": False,
                "status": "OPEN",
                "pnl": 0.0,
                "closed_at_th": "-"
            }
            journal.insert(0, new_sig)
            lifecycle_events.append({"event": "NEW_ENTRY", "mode": mode_label, "trade": dict(new_sig)})

    journal = journal[:40]
    with open(journal_path, "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)

    closed = [t for t in journal if t["status"] in ("WIN", "LOSS")]
    wins = [t for t in closed if t["status"] == "WIN"]
    live_wr = round((len(wins) / len(closed)) * 100.0, 1) if closed else 0.0
    net_live_pnl = round(sum(float(t["pnl"]) for t in closed), 2)

    return {
        "total_signals": len(journal),
        "closed_trades": len(closed),
        "open_trades": len([t for t in journal if t["status"] == "OPEN"]),
        "wins": len(wins),
        "losses": len(closed) - len(wins),
        "live_win_rate": live_wr,
        "net_live_pnl": net_live_pnl,
        "recent_trades": journal[:10]
    }, lifecycle_events

def update_all_journals(quant_data, scalp_data, df_intraday, spot_offset, trend_triggered, scalp_triggered):
    trend_j, trend_events = _process_single_journal(
        "data/trade_journal.json", "TREND", quant_data["setup"], trend_triggered,
        quant_data["spot_price"], df_intraday, spot_offset,
        spread_cost=0.60, dup_tol=8.0, factor_key="active_factor"
    )
    scalp_j, scalp_events = _process_single_journal(
        "data/scalp_journal.json", "SCALP 500PT", scalp_data["setup"], scalp_triggered,
        quant_data["spot_price"], df_intraday, spot_offset,
        spread_cost=0.45, dup_tol=3.0, factor_key="shape"
    )
    return trend_j, scalp_j, (trend_events + scalp_events)

# ================= 5. 4-AGENT SWARM DEBATE COUNCIL =================
def run_multi_agent_thaillm(quant_data, vp_data, backtest_data, scalp_data, news_list, learning_state, kb_files):
    setup = quant_data["setup"]
    tech = quant_data["technicals"]
    gates = quant_data["gates"]
    bt = backtest_data["best"]
    sc_an = scalp_data["anatomy"]
    sc_bt = scalp_data["backtest"]
    sc_set = scalp_data["setup"]
    sc_gates = scalp_data["gates"]

    upcoming = [n for n in news_list if n["is_upcoming"]]
    now_th_str = datetime.now(THAI_TZ).strftime("%H:%M น.")

    if upcoming:
        news_lines = "\n".join([f"- {n['time_th']} ({n['status']}): {n['title']} [Impact: {n['impact']}]" for n in upcoming])
        imminent = [n['title'] for n in upcoming if n['hours_away'] <= 2.0]
        news_status_text = f"ระวังข่าว USD กำลังจะออกใน 2 ชม.: {', '.join(imminent)}\n{news_lines}" if imminent else f"ข่าว USD ที่กำลังจะออก:\n{news_lines}"
    else:
        news_status_text = "ไม่มีข่าวสำคัญ USD คงเหลือในช่วงใกล้ๆ นี้"

    muted_all = learning_state.get("muted_trend_factors", []) + learning_state.get("muted_scalp_shapes", [])
    learning_note = (
        f"Self-Learning ปิดกั้นชั่วคราว: {', '.join(muted_all)} (สถิติจริงสะสม {learning_state['overall_wins']}W/{learning_state['overall_losses']}L)"
        if muted_all else
        f"Self-Learning ปกติ (สถิติจริงสะสม {learning_state['overall_wins']}W/{learning_state['overall_losses']}L | Net ${learning_state['overall_pnl']})"
    )

    fb_a1 = (
        f"ผลทดสอบ Walk-Forward Trend ({bt['trades']} ไม้): WR {bt['win_rate']}%, R-PF {bt['r_profit_factor']}, Net {bt['net_r']}R, "
        f"Daily Sharpe {bt['sharpe']} | ระบบ Scalp 500 จุด ({sc_bt['trades']} ไม้): WR {sc_bt['win_rate']}%, R-PF {sc_bt['r_profit_factor']}, "
        f"Net {sc_bt['net_r']}R ({learning_note})"
    )
    agent1_prompt = (
        "คุณคือ Quantitative R-Multiple & Scalp Auditor ประเมินผลทดสอบทั้งระบบรันเทรนด์และระบบ Scalp 500 จุด "
        "(ตอบเป็นภาษาไทยสั้นๆ กระชับ 2-3 ประโยค ห้ามเปลี่ยนตัวเลข)\n\n"
        f"- Trend: {bt['trades']} ไม้ | WR = {bt['win_rate']}% | R-PF = {bt['r_profit_factor']} | Net = {bt['net_r']}R | Daily Sharpe = {bt['sharpe']} | Expectancy = {bt.get('expectancy_r', 0)}R/ไม้\n"
        f"- Scalp 500 จุด: {sc_bt['trades']} ไม้ | WR = {sc_bt['win_rate']}% | R-PF = {sc_bt['r_profit_factor']} | Net = {sc_bt['net_r']}R | ทรงที่เปิดใช้งาน: {', '.join(sc_bt.get('enabled_shapes', []))}\n"
        f"- {learning_note}\n"
        "จงสรุปความคุ้มเสี่ยงและความพร้อมใช้งาน"
    )
    quant_audit_text, used_m1 = call_thaillm_agent(MODEL_QUANT_AUDITOR, agent1_prompt, fallback_text=fb_a1, temperature=0.1)

    locked_time_fact = (
        f"ขณะนี้เวลา {now_th_str} (เวลาไทย) ซึ่ง **อยู่ในช่วง KILLZONE ACTIVE**"
        if setup["in_killzone"] else
        f"ขณะนี้เวลา {now_th_str} (เวลาไทย) ซึ่ง **อยู่ในช่วง NO TRADE ZONE (ช่วงเช้า)**"
    )

    fb_a2 = (
        f"Spot XAU/USD ปัจจุบันอยู่ที่ ${quant_data['spot_price']} โครงสร้าง 24H Volume Profile เป็นทรง {sc_an['title_th']} "
        f"(POC ตำแหน่ง {sc_an['poc_pos_pct']}% @ ${sc_an['poc']}) ขณะที่ DXY อยู่ที่ {quant_data['dxy_price']} และ US10Y {quant_data['us10y_yield']}%"
    )
    agent2_prompt = (
        "คุณคือ Macro & Volume Profile Anatomy Officer วิเคราะห์โครงสร้างตลาด Spot XAU/USD "
        "(ตอบเป็นภาษาไทยสั้นๆ กระชับ 2-3 ประโยค)\n\n"
        f"- เวลา: {locked_time_fact} | Spot = ${quant_data['spot_price']}\n"
        f"- กายวิภาค Volume Profile 24H: {sc_an['title_th']} (สถานะ: {'ON' if sc_set['shape_active'] else 'MUTED'} | POC {sc_an['poc_pos_pct']}% | VAH ${sc_an['vah']} / POC ${sc_an['poc']} / VAL ${sc_an['val']})\n"
        f"- DXY = {quant_data['dxy_price']} | US10Y = {quant_data['us10y_yield']}%\n"
        f"- ข่าว USD:\n{news_status_text}\n"
        "จงอธิบายโครงสร้างโวลุ่มและแรงกดดันมหภาค"
    )
    macro_audit_text, used_m2 = call_thaillm_agent(MODEL_MACRO_OFFICER, agent2_prompt, fallback_text=fb_a2, temperature=0.1)

    fb_a3 = (
        f"จุดระวังหลักคือ RSI14 อยู่ที่ {tech['rsi14']} และ ADX14 อยู่ที่ {tech['adx14']} "
        f"หากเข้าออเดอร์ก่อนราคาแตะโซน Entry จริง (Trend ห่าง ${gates['dist_to_entry']} / Scalp ห่าง {sc_set['dist_points']} จุด) อาจโดนไส้เทียนสะบัดกิน SL"
    )
    agent3_critic_prompt = (
        "คุณคือ Contrarian Devil's Advocate จับผิดความเสี่ยงทั้งแผนรันเทรนด์และแผน Scalp 500 จุด "
        "(ตอบเป็นภาษาไทยสั้นๆ กระชับ 2-3 ประโยค)\n\n"
        f"- แผน Trend: {setup['bias']} @ ${setup['entry']} (ห่าง ${gates['dist_to_entry']})\n"
        f"- แผน Scalp 500 จุด ({sc_an['shape']} - {'ON' if sc_set['shape_active'] else 'MUTED'}): {sc_set['bias']} @ ${sc_set['entry']} (ห่าง {sc_set['dist_points']} จุด)\n"
        f"- RSI14 = {tech['rsi14']} | ADX14 = {tech['adx14']}\n"
        "จงเตือนจุดเสี่ยงที่อาจทำให้โดน SL"
    )
    contrarian_audit_text, used_m3_critic = call_thaillm_agent(MODEL_CONTRARIAN_CRITIC, agent3_critic_prompt, fallback_text=fb_a3, temperature=0.15)

    fb_final = (
        f"1. **Daily Outlook & VP Anatomy**: ราคา Spot XAU/USD อยู่ที่ ${quant_data['spot_price']} โครงสร้าง 24H Volume Profile เป็นทรง {sc_an['title_th']} ({sc_an['desc_th']}) แนวรับ-ต้านหลักอยู่ที่ VAL ${sc_an['val']} / POC ${sc_an['poc']} / VAH ${sc_an['vah']}\n"
        f"2. **Dual-Engine Edge & Self-Learning**: ระบบ Trend ทำผลงาน {bt['net_r']}R (WR {bt['win_rate']}%, Daily Sharpe {bt['sharpe']}) และระบบ Scalp 500 จุดทำผลงาน {sc_bt['net_r']}R (WR {sc_bt['win_rate']}%, R-PF {sc_bt['r_profit_factor']}) | {learning_note}\n"
        f"3. **Trend Execution Plan**: {setup['bias']} | Entry ${setup['entry']} | SL ${setup['sl']} | BE (+1.0R) ${setup['be_target']} | TP ${setup['tp']} (R:R 1:{setup['rr']})\n"
        f"4. **VP-Anatomy Scalp Plan (500 จุด)**: {sc_set['bias']} | Entry ${sc_set['entry']} | SL ${sc_set['sl']} (500 จุด) | BE +250 จุด (${sc_set['be_target']}) | TP ${sc_set['tp']} ({sc_set['tp_points']} จุด)\n"
        f"5. **Alert Trigger**: Trend: {gates['verdict_text']} | Scalp 500pt: {sc_gates['verdict_text']}"
    )

    agent4_prompt = (
        "คุณคือ Chief Execution Strategist สำหรับ Spot XAU/USD (Dual-Engine: Trend + VP-Anatomy Scalp 500 จุด) "
        "จงสังเคราะห์ความเห็นจากทั้ง 3 Agents ร่วมกับ [ข้อเท็จจริงที่ถูกล็อกไว้] ห้ามเปลี่ยนคำตัดสินในข้อ 5 เด็ดขาด!\n\n"
        "[ข้อเท็จจริงที่ถูกล็อกไว้ 100%]\n"
        f"- เวลา: {locked_time_fact} | Spot = ${quant_data['spot_price']}\n"
        f"- กายวิภาค Volume Profile 24H: {sc_an['title_th']} ({sc_an['desc_th']})\n"
        f"- สถานะ Self-Learning: {learning_note}\n"
        f"- แผน Trend หลัก: {setup['bias']} | Entry ${setup['entry']} | SL ${setup['sl']} | BE ${setup['be_target']} | TP ${setup['tp']} (R:R 1:{setup['rr']})\n"
        f"- แผน Scalp 500 จุด: {sc_set['bias']} | Entry ${sc_set['entry']} | SL ${sc_set['sl']} (500 จุด) | BE +250 จุด (${sc_set['be_target']}) | TP ${sc_set['tp']} ({sc_set['tp_points']} จุด)\n"
        f"- คำตัดสิน Alert Trigger ที่บังคับตอบในข้อ 5: Trend: {gates['verdict_text']} | Scalp 500pt: {sc_gates['verdict_text']}\n\n"
        f"[Agent 1]: {quant_audit_text}\n[Agent 2]: {macro_audit_text}\n[Agent 3]: {contrarian_audit_text}\n\n"
        "จงเขียนรายงาน 7 หัวข้อตามรูปแบบนี้เป๊ะๆ:\n"
        "1. **Daily Outlook & VP Anatomy**: สรุปโครงสร้างราคาและรูปทรง Volume Profile\n"
        f"2. **Dual-Engine Edge & Self-Learning**: สรุปผลทดสอบ Trend, Scalp 500 จุด และบทเรียนจาก Journal จริง\n"
        f"3. **Trend Execution Plan**: แผนรันเทรนด์ Entry ${setup['entry']}, SL ${setup['sl']}, BE ${setup['be_target']}, TP ${setup['tp']}\n"
        f"4. **VP-Anatomy Scalp Plan (500 จุด)**: แผนสั้น Entry ${sc_set['entry']}, SL ${sc_set['sl']}, BE ${sc_set['be_target']}, TP ${sc_set['tp']}\n"
        f"5. **Alert Trigger**: คัดลอกคำตัดสินนี้ไปวางทันที -> Trend: {gates['verdict_text']} | Scalp 500pt: {sc_gates['verdict_text']}\n"
        "6. **Active Recall Question**: ตั้งคำถามสั้นๆ 1 ข้อเกี่ยวกับกายวิภาค Volume Profile หรือระบบ Self-Learning\n"
        "7. **Active Recall Answer**: เฉลยคำตอบสั้นๆ 1-2 ประโยค"
    )
    final_raw, used_m4 = call_thaillm_agent(MODEL_CHIEF_STRATEGIST, agent4_prompt, fallback_text=fb_final, temperature=0.08)

    q_match = re.search(r'6\.\s*\*?\*?Active Recall Question\*?\*?:\s*(.*?)(?=\n7\.|\Z)', final_raw, re.DOTALL)
    a_match = re.search(r'7\.\s*\*?\*?Active Recall Answer\*?\*?:\s*(.*)', final_raw, re.DOTALL)
    recall_q = q_match.group(1).strip() if q_match else "เหตุใดระบบจึงใช้ Daily Aggregated Returns ในการคำนวณ Sharpe/Sortino และใช้ Self-Learning Guard ปิดทรงที่แพ้ต่อเนื่อง?"
    recall_a = a_match.group(1).strip() if a_match else "เพื่อป้องกันตัวเลข Sharpe เฟ้อเกินจริงจากจำนวนไม้ที่น้อย และหยุดการขาดทุนซ้ำซากทันทีเมื่อพฤติกรรมราคาจริงเปลี่ยนไปจากผล Backtest"

    main_match = re.search(r'(1\.\s*\*?\*?Daily Outlook.*?)(?=\n6\.|\Z)', final_raw, re.DOTALL)
    clean_final = main_match.group(1).strip() if main_match else final_raw

    return {
        "final_report": clean_final,
        "quant_audit": quant_audit_text,
        "macro_audit": macro_audit_text,
        "contrarian_audit": contrarian_audit_text,
        "active_recall_q": recall_q,
        "active_recall_a": recall_a,
        "kb_files_loaded": kb_files,
        "models_used": {
            "quant_auditor": used_m1,
            "macro_officer": used_m2,
            "contrarian_critic": used_m3_critic,
            "chief_strategist": used_m4
        }
    }

# ================= 6. FULL-LIFECYCLE NTFY DISPATCHER =================
def send_ntfy_alert(title, message, priority=4, tags=None):
    url = "https://ntfy.sh"
    payload = {
        "topic": NTFY_TOPIC,
        "title": title,
        "message": message,
        "priority": priority,
        "tags": tags or ["chart_with_upwards_trend", "coin"],
        "markdown": True
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print(f"[NTFY] ส่งแจ้งเตือน '{title}' เข้า '{NTFY_TOPIC}' สำเร็จ")
            return True
        else:
            print(f"[NTFY] ส่งไม่สำเร็จ HTTP {res.status_code}: {res.text}")
            return False
    except Exception as e:
        print(f"[NTFY] เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")
        return False

def dispatch_lifecycle_notifications(lifecycle_events, updated_learning_state):
    """ยิงแจ้งเตือนทันทีเมื่อเกิดเหตุการณ์: NEW_ENTRY, BE_HIT, TP_WIN, SL_LOSS"""
    for ev in lifecycle_events:
        etype = ev["event"]
        mode = ev["mode"]
        tr = ev["trade"]
        factor = tr.get("factor", "SETUP")
        direction = tr.get("direction", "-")
        entry_p = tr.get("entry", 0)
        sl_p = tr.get("sl", 0)
        tp_p = tr.get("tp", 0)
        be_p = tr.get("be_target", entry_p)
        pnl = tr.get("pnl", 0)

        if etype == "NEW_ENTRY":
            send_ntfy_alert(
                f"NEW SIGNAL [{mode}] {direction} @ ${entry_p}",
                f"**กลยุทธ์:** {factor}\n**Entry:** ${entry_p} | **SL:** ${sl_p} | **BE:** ${be_p} | **TP:** ${tp_p} (R:R 1:{tr.get('rr', 1.0)})\nบันทึกเข้า Live Journal เรียบร้อย",
                priority=5,
                tags=["rotating_light", "coin"]
            )
        elif etype == "BE_HIT":
            send_ntfy_alert(
                f"BREAKEVEN LOCKED [{mode}] {direction} @ ${entry_p}",
                f"**กลยุทธ์:** {factor} วิ่งแตะจุดบังทุน **${be_p}** แล้ว!\nแนะนำเลื่อน Stop Loss มาหน้าทุนทันทีเพื่อปิดความเสี่ยง 0%",
                priority=4,
                tags=["shield", "white_check_mark"]
            )
        elif etype == "TP_WIN":
            send_ntfy_alert(
                f"TAKE PROFIT HIT! [{mode}] +${pnl}/oz",
                f"**กลยุทธ์:** {factor} ({direction} @ ${entry_p} -> TP ${tp_p})\n**กำไรสุทธิ:** +${pnl}/oz | **Self-Learning:** บันทึกสถิติชนะเข้า KB เรียบร้อย ({updated_learning_state['overall_wins']}W/{updated_learning_state['overall_losses']}L)",
                priority=5,
                tags=["tada", "moneybag"]
            )
        elif etype == "SL_LOSS":
            muted_now = factor in (updated_learning_state.get("muted_trend_factors", []) + updated_learning_state.get("muted_scalp_shapes", []))
            learn_msg = f"Self-Learning Guard สั่ง AUTO-MUTE `{factor}` ชั่วคราวทันที!" if muted_now else f"Self-Learning บันทึกบทเรียนเข้า KB แล้ว ({updated_learning_state['overall_wins']}W/{updated_learning_state['overall_losses']}L)"
            send_ntfy_alert(
                f"STOP LOSS HIT [{mode}] ${pnl}/oz",
                f"**กลยุทธ์:** {factor} ({direction} @ ${entry_p} -> SL ${sl_p})\n**ขาดทุนสุทธิ:** ${pnl}/oz\n**AI Action:** {learn_msg}",
                priority=5,
                tags=["warning", "x"]
            )

def send_cycle_pulse_notification(quant_data, scalp_data, learning_state, is_startup=False):
    """ส่งสรุปรอบคำนวณทุกครั้ง (ทุก 15 นาที) เข้า ntfy เพื่อยืนยันว่าบอทคำนวณและอัปเดตเวลาตลอด"""
    now_th = datetime.now(THAI_TZ)
    spot = quant_data["spot_price"]
    setup = quant_data["setup"]
    gates = quant_data["gates"]
    sc_an = scalp_data["anatomy"]
    sc_set = scalp_data["setup"]

    prefix = "เปิดระบบ Gold Quant V12" if is_startup else "อัปเดตรอบคำนวณ 15 นาที"
    title = f"{prefix} ({now_th.strftime('%H:%M น.')}) | Spot ${spot}"

    scalp_stat = f"ON (ห่าง {sc_set['dist_points']} จุด)" if sc_set["shape_active"] else f"MUTED ({sc_an['shape']})"
    trend_stat = f"READY (ห่าง ${gates['dist_to_entry']})" if gates["all_passed"] else f"WAIT (ห่าง ${gates['dist_to_entry']})"

    msg = (
        f"**เวลาคำนวณ:** {now_th.strftime('%d/%m/%Y %H:%M:%S น.')}\n"
        f"**Spot XAU/USD:** ${spot} ({quant_data['feed_source']})\n"
        f"**VP 24H:** {sc_an['title_th']} | Scalp 500pt: **{scalp_stat}**\n"
        f"**Trend Setup:** {setup['direction']} @ ${setup['entry']} (**{trend_stat}**)\n"
        f"**Live Journal:** {learning_state['overall_wins']}W / {learning_state['overall_losses']}L (Net ${learning_state['overall_pnl']})"
    )
    send_ntfy_alert(title, msg, priority=4, tags=["clock3", "bar_chart"])

def local_git_push():
    if IS_GITHUB_ACTIONS or not os.path.isdir(".git"):
        return
    try:
        subprocess.run(["git", "add", "data/state.json", "data/alert_state.json", "data/trade_journal.json", "data/scalp_journal.json", "data/learning_weights.json", "knowledge_base/"], check=True)
        subprocess.run(["git", "commit", "-m", f"Auto-sync Gold Quant V12 {datetime.now().strftime('%H:%M')}"], check=True)
        subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], check=True)
        subprocess.run(["git", "push"], check=True)
    except Exception as e:
        print(f"Local git push skipped: {e}")

def run_single_cycle(is_startup=False, force_flag=False, silent_pulse=False):
    kb_text, kb_files, learning_state = ensure_and_load_knowledge_base()
    print(f"[KB & SELF-LEARNING] Loaded {len(kb_files)} docs | Muted Trend={learning_state['muted_trend_factors']} | Muted Scalp={learning_state['muted_scalp_shapes']}")

    df_intraday_vault, df_daily, df_dxy, df_us10y, spot_price, bid, ask, feed_source, open_interest = fetch_market_data()
    futures_last = float(df_intraday_vault['Close'].iloc[-1])
    spot_offset = (spot_price - futures_last) if spot_price else 0.0
    active_spot = round(spot_price, 2) if spot_price else round(futures_last, 2)
    print(f"[SPOT FEED] {feed_source}: ${active_spot} (Bid ${bid} / Ask ${ask}) | Vault: {len(df_intraday_vault)} M15 bars")

    ind = compute_mtf_indicators(df_intraday_vault, spot_offset=spot_offset)
    backtest_data = run_walkforward_backtest(df_intraday_vault, ind, spot_offset=spot_offset)
    best_params = backtest_data["best"]

    scalp_data = run_vp_anatomy_scalp_engine(
        df_intraday_vault, ind, active_spot,
        spot_offset=spot_offset,
        live_muted_shapes=learning_state["muted_scalp_shapes"]
    )
    sc_bt = scalp_data["backtest"]
    print(f"[SCALP 500PT] Anatomy={scalp_data['anatomy']['shape']} (Active={scalp_data['setup']['shape_active']}) | WR={sc_bt['win_rate']}%, R-PF={sc_bt['r_profit_factor']}, Net={sc_bt['net_r']}R ({sc_bt['trades']} trades)")

    vp_data = calculate_volume_profile(df_intraday_vault, spot_offset=spot_offset)
    quant_data = calculate_quant_and_setup(
        df_intraday_vault, df_daily, df_dxy, df_us10y, vp_data, ind,
        spot_price, bid, ask, feed_source, spot_offset, open_interest,
        tuned_params=best_params,
        live_muted_factors=learning_state["muted_trend_factors"]
    )
    news_list = fetch_usd_news()

    trend_triggered = bool(quant_data["gates"]["all_passed"] or force_flag)
    scalp_triggered = bool(scalp_data["gates"]["all_passed"] or (force_flag and scalp_data["setup"]["shape_active"]))

    trend_j, scalp_j, lifecycle_events = update_all_journals(
        quant_data, scalp_data, df_intraday_vault, spot_offset,
        trend_triggered=trend_triggered, scalp_triggered=scalp_triggered
    )

    # Re-evaluate Self-Learning immediately if any trade closed during this cycle!
    if any(ev["event"] in ("TP_WIN", "SL_LOSS") for ev in lifecycle_events):
        learning_state = evaluate_live_self_learning()

    dispatch_lifecycle_notifications(lifecycle_events, learning_state)

    if not silent_pulse:
        send_cycle_pulse_notification(quant_data, scalp_data, learning_state, is_startup=is_startup)

    ai_council = run_multi_agent_thaillm(quant_data, vp_data, backtest_data, scalp_data, news_list, learning_state, kb_files)

    os.makedirs("data", exist_ok=True)
    state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "quant": quant_data,
        "volume_profile": vp_data,
        "backtest": backtest_data,
        "scalp": scalp_data,
        "journal": trend_j,
        "scalp_journal": scalp_j,
        "self_learning": learning_state,
        "news": news_list,
        "ai_analysis": ai_council["final_report"],
        "ai_council": ai_council
    }
    
    with open("data/state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        
    local_git_push()
    print(f"Spot Gold Quant V12 Completed @ {datetime.now(THAI_TZ).strftime('%H:%M:%S')} | Spot=${quant_data['spot_price']}")

def main():
    if "--test-alert" in sys.argv:
        send_ntfy_alert(
            "ทดสอบระบบ XAU/USD Quant & Self-Learning V12",
            f"เชื่อมต่อ **ntfy ({NTFY_TOPIC})** สำเร็จ! พร้อมแจ้งเตือนทุก 15 นาที + ENTRY / BE / TP / SL",
            priority=4
        )
        return

    force_flag = "--force-alert" in sys.argv
    no_startup = "--no-startup-alert" in sys.argv
    silent_pulse = "--silent" in sys.argv

    if "--loop" in sys.argv:
        try:
            idx = sys.argv.index("--loop")
            interval_mins = int(sys.argv[idx + 1])
        except Exception:
            interval_mins = 15
        print(f"=== เริ่มทำงานโหมด 24/7 Auto-Loop V12 ทุก {interval_mins} นาที (กด Ctrl+C เพื่อหยุด) ===")
        first_run = True
        while True:
            try:
                run_single_cycle(is_startup=(first_run and not no_startup), force_flag=False, silent_pulse=silent_pulse)
                first_run = False
            except Exception as e:
                print(f"[LOOP ERROR] {e}")
            next_run = (datetime.now(THAI_TZ) + timedelta(minutes=interval_mins)).strftime("%H:%M:%S")
            print(f"รออัปเดตรอบถัดไปเวลา {next_run} น. ...\n")
            now_c = datetime.now(THAI_TZ); n_q = ((now_c.minute // interval_mins) + 1) * interval_mins; nxt_dt = now_c.replace(minute=0, second=5, microsecond=0) + timedelta(minutes=n_q); time.sleep(max(10, (nxt_dt - now_c).total_seconds()))
    else:
        run_single_cycle(is_startup=(not no_startup), force_flag=force_flag, silent_pulse=silent_pulse)

if __name__ == "__main__":
    main()


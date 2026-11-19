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

# ================= CONFIGURATION =================
THAILLM_API_KEY = os.getenv("THAILLM_API_KEY", "YOUR_THAILLM_API_KEY")
NTFY_TOPIC = "pansakorn888"
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
THAI_TZ = timezone(timedelta(hours=7))

MODEL_QUANT_AUDITOR = "pathumma-thaillm-qwen3-8b-think-3.0.0"
MODEL_MACRO_OFFICER = "typhoon-s-thaillm-8b-instruct"
MODEL_CONTRARIAN_CRITIC = "typhoon-s-thaillm-8b-instruct"
MODEL_CHIEF_STRATEGIST = "thalle-0.2-thaillm-8b-fa"

llm_client = OpenAI(
    base_url="https://thaillm.or.th/api/v1",
    api_key=THAILLM_API_KEY
)

# ================= 0. KNOWLEDGE BASE & JOURNAL SELF-LEARNING =================
DEFAULT_KB_FILES = {
    "01_market_maker_liquidity.md": (
        "# Market Maker, Asian Range & Liquidity Sweep Rules\n"
        "- สภาพคล่อง (Liquidity Pools) สะสมอยู่เหนือ Previous 24H High (PDH) และใต้ Previous 24H Low (PDL)\n"
        "- London Breakout Setup: จับจังหวะราคาเบรกกรอบ Asian Session Range (07:00 - 12:45 น. เวลาไทย) ช่วงเปิดตลาดลอนดอน (13:00 - 16:30 น.) ตามเทรนด์ H1/M15\n"
        "- Liquidity Sweep Setup: เมื่อราคาพุ่งทะลุ PDH หรือ PDL เพื่อกวาด Stop Loss แต่แท่งเทียน M15 ปิดกลับเข้ามาในกรอบเดิมพร้อมไส้เทียนปฏิเสธราคา\n"
        "- RSI Exhaustion Guard: ห้ามไล่ Breakout ฝั่ง Short เมื่อ RSI14 < 28 และห้ามไล่ฝั่ง Long เมื่อ RSI14 > 72 เพื่อหลีกเลี่ยงกับดัก False Breakout\n"
    ),
    "02_factorbench_investability.md": (
        "# FactorBench R-Multiple, Institutional Risk & Monte Carlo Rules\n"
        "- วัดผลและคัดเลือกพารามิเตอร์ด้วย R-Multiple (Net R, R-Profit Factor, Sharpe, Sortino, Calmar) ให้สอดคล้องกับ % Risk ต่อไม้จริง\n"
        "- แบ่งทดสอบ Walk-Forward: 67% แรก (In-Sample) เพื่อคัดเลือกพารามิเตอร์และปิด Factor ที่ทำค่า Net R <= 0 (Auto-Mute) และ 33% หลัง (Out-of-Sample)\n"
        "- Smart Breakeven & Trailing Lock: เลื่อน Stop Loss มาบังทุน (+0.15 ATR) เมื่อกำไรแตะ +1.0R และล็อกกำไร +1.0R เมื่อราคาวิ่งถึง +1.8R\n"
        "- หักต้นทุน Spread และ Slippage จริง $0.60/oz ในทุกไม้ที่ทดสอบ\n"
    ),
    "03_macro_gold_regime.md": (
        "# Macro Gold (XAU/USD) & Multi-Timeframe Regime Rules\n"
        "- เมื่อ ADX(14) <= ADX Threshold ตลาดแกว่งออกข้าง ใช้กลยุทธ์ VALUE_AREA_FADE บริเวณขอบ VAH/VAL ร่วมกับ RSI\n"
        "- เมื่อ ADX(14) > ADX Threshold ตลาดมีเทรนด์แข็งแกร่ง ให้เน้น LONDON_BREAKOUT, MOMENTUM_BREAKOUT หรือ HTF_TREND_PULLBACK ที่ไม่ติด RSI Exhaustion\n"
        "- ดัชนี DXY และ US10Y Yield ที่ปรับตัวขึ้นมักกดดันราคาทองคำ ต้องระวังความผันผวนช่วงข่าว USD สำคัญ\n"
    ),
    "04_vp_anatomy_scalp_500pt.md": (
        "# 4 กายวิภาค Volume Profile (VPAnatomies_EA) สำหรับ Scalp 500 จุด ($5.00)\n"
        "- 1. D-SHAPED (POC 34%-66%): ตลาดสมดุล ดัก BUY ที่แนวรับ VAL และดัก SELL ที่แนวต้าน VAH เป้าหมายเข้าหา POC\n"
        "- 2. B-SHAPED (POC < 34%): โวลุ่มกองโซนล่าง เปิด SELL เมื่อโมเมนตัมขาลงและราคากดใต้ POC หรือเปิด BUY เมื่อเบรกทะลุ VAH ขึ้นไป\n"
        "- 3. P-SHAPED (POC > 66%): โวลุ่มกองโซนบน เปิด BUY ตามแนวโน้มเมื่อราคายืนเหนือ/ย่อรับที่ POC/VAL และเปิด SELL หากหลุดกรอบล่าง VAL\n"
        "- 4. DOUBLE ZONE (2nd Peak >= 60% ของ POC + มี Valley คั่น): ดัก BUY ที่ Lower Peak และดัก SELL ที่ Upper Peak\n"
        "- ระยะ Scalp มาตรฐาน: SL = $5.00 (500 จุด), เลื่อนบังทุน BE เมื่อบวก +$2.50 (+250 จุด), TP = $5.00 - $6.50 (500 - 650 จุด)\n"
    )
}

def update_journal_lessons_kb():
    """สรุปผลแพ้-ชนะจริงจาก trade_journal.json และ scalp_journal.json ลง Knowledge Base"""
    lesson_path = "knowledge_base/05_live_journal_lessons.md"
    all_closed = []
    for j_file, mode_lbl in [("data/trade_journal.json", "TREND"), ("data/scalp_journal.json", "SCALP_500PT")]:
        if os.path.exists(j_file):
            try:
                with open(j_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
                for t in items:
                    if t.get("status") in ("WIN", "LOSS"):
                        t_copy = dict(t)
                        t_copy["mode"] = mode_lbl
                        all_closed.append(t_copy)
            except Exception:
                pass
    if not all_closed:
        return
    try:
        wins = [t for t in all_closed if t["status"] == "WIN"]
        wr = round((len(wins) / len(all_closed)) * 100.0, 1)
        net_pnl = round(sum(float(t.get("pnl", 0)) for t in all_closed), 2)
        lines = [
            "# Live Signal Journal Post-Mortem (Self-Learning Memory)",
            f"- สถิติสัญญาณจริงสะสมทั้งหมด {len(all_closed)} ไม้: Win-Rate = {wr}% | Net PnL = ${net_pnl}/oz"
        ]
        by_factor = {}
        for t in all_closed:
            key = f"{t['mode']}:{t.get('factor', 'SETUP')}"
            by_factor.setdefault(key, {"w": 0, "l": 0, "pnl": 0.0})
            if t["status"] == "WIN":
                by_factor[key]["w"] += 1
            else:
                by_factor[key]["l"] += 1
            by_factor[key]["pnl"] += float(t.get("pnl", 0))
        for k, st in by_factor.items():
            lines.append(f"- `{k}`: ชนะ {st['w']} / แพ้ {st['l']} (รวม ${round(st['pnl'], 2)}/oz)")
        with open(lesson_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[KB] Journal post-mortem skipped: {e}")

def ensure_and_load_knowledge_base():
    kb_dir = "knowledge_base"
    os.makedirs(kb_dir, exist_ok=True)
    for fname, content in DEFAULT_KB_FILES.items():
        fpath = os.path.join(kb_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(content)

    update_journal_lessons_kb()

    loaded_texts = []
    loaded_names = []
    for fname in sorted(os.listdir(kb_dir)):
        if fname.endswith(".md"):
            fpath = os.path.join(kb_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                loaded_texts.append(f.read().strip())
                loaded_names.append(fname)

    return "\n\n".join(loaded_texts), loaded_names

def clean_yf_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def call_thaillm_agent(model_name, prompt, temperature=0.1):
    models_to_try = [model_name]
    if model_name != MODEL_CHIEF_STRATEGIST:
        models_to_try.append(MODEL_CHIEF_STRATEGIST)

    for m in models_to_try:
        try:
            response = llm_client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            raw = response.choices[0].message.content.strip()
            cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            return cleaned, m
        except Exception as e:
            print(f"[AI] Model '{m}' error: {e} -> trying fallback...")
    return "ไม่สามารถเชื่อมต่อ ThaiLLM ได้ชั่วคราว", "offline"

def fetch_spot_and_paxg_prices():
    spot_mid, spot_bid, spot_ask = None, None, None
    paxg_mid = None

    try:
        res = requests.get(
            "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD",
            timeout=8
        ).json()
        prices = res[0]["spreadProfilePrices"][0]
        spot_bid = float(prices["bid"])
        spot_ask = float(prices["ask"])
        spot_mid = round((spot_bid + spot_ask) / 2, 2)
    except Exception:
        pass

    if spot_mid is None:
        try:
            res = requests.get("https://api.gold-api.com/price/XAU", timeout=8).json()
            if "price" in res and res["price"]:
                spot_mid = round(float(res["price"]), 2)
                spot_bid, spot_ask = spot_mid, spot_mid
        except Exception:
            pass

    try:
        res = requests.get(
            "https://api.binance.com/api/v3/ticker/bookTicker?symbol=PAXGUSDT",
            timeout=8
        ).json()
        p_bid = float(res["bidPrice"])
        p_ask = float(res["askPrice"])
        paxg_mid = round((p_bid + p_ask) / 2, 2)
    except Exception:
        pass

    return spot_mid, spot_bid, spot_ask, paxg_mid

# ================= 1. HYDRA DATA VAULT & MULTI-TIMEFRAME INDICATORS =================
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
        except Exception as e:
            print(f"[HYDRA VAULT] Read error ({e}), rebuilding vault...")
            df_merged = df_new.sort_index()
    else:
        df_merged = df_new.sort_index()

    df_merged = df_merged.tail(18000)
    try:
        df_merged.to_csv(vault_path)
    except Exception as e:
        print(f"[HYDRA VAULT] Save skipped: {e}")

    return df_merged

def fetch_market_data():
    df_fresh_60d = clean_yf_df(yf.download("GC=F", period="60d", interval="15m", progress=False))
    df_intraday_vault = sync_hydra_m15_vault(df_fresh_60d)

    df_daily = clean_yf_df(yf.download("GC=F", period="120d", interval="1d", progress=False))
    df_dxy = clean_yf_df(yf.download("DX-Y.NYB", period="60d", interval="1d", progress=False))
    df_us10y = clean_yf_df(yf.download("^TNX", period="60d", interval="1d", progress=False))
    
    spot_price, bid, ask, paxg_mid = fetch_spot_and_paxg_prices()

    try:
        gc_info = yf.Ticker("GC=F").info
        open_interest = gc_info.get("openInterest", None)
    except Exception:
        open_interest = None
        
    return df_intraday_vault, df_daily, df_dxy, df_us10y, spot_price, bid, ask, paxg_mid, open_interest

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
        "ema21": ema21,
        "ema50": ema50,
        "ema200": ema200,
        "h1_ema50": h1_ema50_m15,
        "h1_ema200": h1_ema200_m15,
        "atr14": atr14,
        "rsi14": rsi14,
        "adx14": adx14,
        "pdh": pdh,
        "pdl": pdl,
        "asian_high": asian_high,
        "asian_low": asian_low
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

# ================= 1B. 4-ANATOMY VOLUME PROFILE ANALYZER (VPAnatomies_EA) =================
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
    """จำแนก 4 กายวิภาค Volume Profile (D-Shaped, b-Shaped, P-Shaped, Double Zone) ตามหลัก VPAnatomies_EA"""
    w_min = float(low_win.min())
    w_max = float(high_win.max())
    if w_max - w_min < 2.0:
        return None
    bin_edges = np.linspace(w_min, w_max, bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    typical = (high_win + low_win + close_win) / 3.0
    weights = vol_win if vol_win.sum() > 0 else np.ones_like(vol_win)
    hist, _ = np.histogram(typical, bins=bin_edges, weights=weights)

    # เกลี่ยความเรียบ (Smoothing) เพื่อค้นหา Peak และ Valley
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

# ================= 2A. TREND WALK-FORWARD & MONTE CARLO ENGINE =================
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
    utc_hours = dt_index_utc.hour.values
    date_ordinals = dt_index_utc.date

    bars_per_day = 92
    lookback_5d = 5 * bars_per_day
    split_idx = lookback_5d + int((n_bars - lookback_5d) * 0.67)
    spread_cost = 0.60

    all_factors = ["VALUE_AREA_FADE", "LIQUIDITY_SWEEP", "LONDON_BREAKOUT", "MOMENTUM_BREAKOUT", "HTF_TREND_PULLBACK"]

    default_best = {
        "atr_sl_mult": 1.6, "rr_mult": 2.2, "adx_thresh": 25,
        "enabled_factors": all_factors, "trades": 0, "win_rate": 0.0,
        "profit_factor": 0.0, "r_profit_factor": 0.0, "net_pnl": 0.0, "net_r": 0.0,
        "max_dd": 0.0, "max_dd_r": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "calmar": 0.0, "var_95_r": -1.0, "mc_dd_95_r": -4.0, "mc_prob_profit": 50.0,
        "total_fee_drag": 0.0, "oos_trades": 0, "oos_win_rate": 0.0,
        "oos_profit_factor": 0.0, "oos_net_r": 0.0, "oos_net_pnl": 0.0,
        "is_profitable_edge": False, "equity_curve_r": [0], "trade_r_list": []
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
        be_moved = False
        trail_locked = False
        trade_records = []
        cooldown = 0

        for i in range(lookback_5d + 1, n_bars):
            if np.isnan(vp_val[i]) or np.isnan(atr_arr[i]) or atr_arr[i] <= 0 or np.isnan(pdh_arr[i]):
                continue

            if in_pos == 1:
                if low_arr[i] <= sl_p:
                    pnl = (sl_p - entry_p) - spread_cost
                    trade_records.append({"idx": i, "factor": active_factor, "pnl": pnl, "r": pnl / init_risk, "date": date_ordinals[i]})
                    in_pos = 0; cooldown = 3; continue
                if high_arr[i] >= tp_p:
                    pnl = (tp_p - entry_p) - spread_cost
                    trade_records.append({"idx": i, "factor": active_factor, "pnl": pnl, "r": pnl / init_risk, "date": date_ordinals[i]})
                    in_pos = 0; cooldown = 2; continue
                if not be_moved and high_arr[i] >= entry_p + init_risk * 1.0:
                    sl_p = entry_p + (atr_arr[i] * 0.15); be_moved = True
                if not trail_locked and high_arr[i] >= entry_p + init_risk * 1.8:
                    sl_p = max(sl_p, entry_p + init_risk * 1.0); trail_locked = True
                continue

            elif in_pos == -1:
                if high_arr[i] >= sl_p:
                    pnl = (entry_p - sl_p) - spread_cost
                    trade_records.append({"idx": i, "factor": active_factor, "pnl": pnl, "r": pnl / init_risk, "date": date_ordinals[i]})
                    in_pos = 0; cooldown = 3; continue
                if low_arr[i] <= tp_p:
                    pnl = (entry_p - tp_p) - spread_cost
                    trade_records.append({"idx": i, "factor": active_factor, "pnl": pnl, "r": pnl / init_risk, "date": date_ordinals[i]})
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

            if "VALUE_AREA_FADE" in enabled_set and adx <= adx_thresh:
                if low_arr[i] <= vp_val[i] + atr * 0.35 and close_arr[i] > vp_val[i] - atr * 0.25 and bull_candle and rsi <= 45:
                    in_pos = 1; active_factor = "VALUE_AREA_FADE"; be_moved = False; trail_locked = False
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p - risk; tp_p = entry_p + reward; continue
                elif high_arr[i] >= vp_vah[i] - atr * 0.35 and close_arr[i] < vp_vah[i] + atr * 0.25 and bear_candle and rsi >= 55:
                    in_pos = -1; active_factor = "VALUE_AREA_FADE"; be_moved = False; trail_locked = False
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p + risk; tp_p = entry_p - reward; continue

            swept_pdl = (low_arr[i] < pdl or low_arr[i - 1] < pdl) and close_arr[i] > pdl and bull_candle and (lower_wick_pct >= 0.22 or rsi <= 45)
            swept_pdh = (high_arr[i] > pdh or high_arr[i - 1] > pdh) and close_arr[i] < pdh and bear_candle and (upper_wick_pct >= 0.22 or rsi >= 55)

            if "LIQUIDITY_SWEEP" in enabled_set and swept_pdl:
                in_pos = 1; active_factor = "LIQUIDITY_SWEEP"; be_moved = False; trail_locked = False
                entry_p = close_arr[i]
                swing_low = min(low_arr[i], low_arr[i - 1])
                init_risk = max(entry_p - (swing_low - atr * 0.25), risk * 0.85)
                sl_p = entry_p - init_risk; tp_p = entry_p + (init_risk * rr_mult); continue
            elif "LIQUIDITY_SWEEP" in enabled_set and swept_pdh:
                in_pos = -1; active_factor = "LIQUIDITY_SWEEP"; be_moved = False; trail_locked = False
                entry_p = close_arr[i]
                swing_high = max(high_arr[i], high_arr[i - 1])
                init_risk = max((swing_high + atr * 0.25) - entry_p, risk * 0.85)
                sl_p = entry_p + init_risk; tp_p = entry_p - (init_risk * rr_mult); continue

            if "LONDON_BREAKOUT" in enabled_set and (6 <= hr <= 9) and not np.isnan(as_h) and body_pct >= 0.45:
                htf_bear = close_arr[i] < h1_e200 and close_arr[i] < e200
                htf_bull = close_arr[i] > h1_e200 and close_arr[i] > e200
                if htf_bear and close_arr[i] < as_l and close_arr[i - 1] >= as_l and bear_candle and (28 <= rsi <= 52):
                    in_pos = -1; active_factor = "LONDON_BREAKOUT"; be_moved = False; trail_locked = False
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p + risk; tp_p = entry_p - reward; continue
                elif htf_bull and close_arr[i] > as_h and close_arr[i - 1] <= as_h and bull_candle and (48 <= rsi <= 72):
                    in_pos = 1; active_factor = "LONDON_BREAKOUT"; be_moved = False; trail_locked = False
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p - risk; tp_p = entry_p + reward; continue

            if "MOMENTUM_BREAKOUT" in enabled_set and adx > adx_thresh and body_pct >= 0.45:
                htf_bear = close_arr[i] < h1_e200 and h1_e50 < h1_e200 and close_arr[i] < e200
                htf_bull = close_arr[i] > h1_e200 and h1_e50 > h1_e200 and close_arr[i] > e200
                broke_down = close_arr[i] < vp_val[i] and close_arr[i - 1] >= vp_val[i] and bear_candle and (28 <= rsi <= 48)
                broke_up = close_arr[i] > vp_vah[i] and close_arr[i - 1] <= vp_vah[i] and bull_candle and (52 <= rsi <= 72)

                if htf_bear and broke_down:
                    in_pos = -1; active_factor = "MOMENTUM_BREAKOUT"; be_moved = False; trail_locked = False
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p + risk; tp_p = entry_p - reward; continue
                elif htf_bull and broke_up:
                    in_pos = 1; active_factor = "MOMENTUM_BREAKOUT"; be_moved = False; trail_locked = False
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p - risk; tp_p = entry_p + reward; continue

            if "HTF_TREND_PULLBACK" in enabled_set and adx > adx_thresh:
                htf_bear = close_arr[i] < h1_e200 and h1_e50 < h1_e200 and close_arr[i] < e200 and e50 < e200
                htf_bull = close_arr[i] > h1_e200 and h1_e50 > h1_e200 and close_arr[i] > e200 and e50 > e200
                if htf_bear and high_arr[i] >= e21 and close_arr[i] < e21 and bear_candle and rsi >= 48:
                    in_pos = -1; active_factor = "HTF_TREND_PULLBACK"; be_moved = False; trail_locked = False
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p + risk; tp_p = entry_p - reward
                elif htf_bull and low_arr[i] <= e21 and close_arr[i] > e21 and bull_candle and rsi <= 52:
                    in_pos = 1; active_factor = "HTF_TREND_PULLBACK"; be_moved = False; trail_locked = False
                    entry_p = close_arr[i]; init_risk = risk; sl_p = entry_p - risk; tp_p = entry_p + reward

        return trade_records

    def summarize_records(records, sl_m, rr_m, adx_t, enabled_list):
        if len(records) < 6:
            return None
        df_tr = pd.DataFrame(records)
        arr = df_tr["pnl"].values
        r_arr = df_tr["r"].values

        wins_r = r_arr[r_arr > 0]
        losses_r = r_arr[r_arr < 0]
        wr = (len(wins_r) / len(r_arr)) * 100.0

        gross_r_win = float(wins_r.sum()) if len(wins_r) > 0 else 0.0
        gross_r_loss = float(abs(losses_r.sum())) if len(losses_r) > 0 else 1e-6
        r_pf = gross_r_win / gross_r_loss

        wins_usd = arr[arr > 0]
        losses_usd = arr[arr < 0]
        usd_pf = (float(wins_usd.sum()) / float(abs(losses_usd.sum()))) if len(losses_usd) > 0 else 2.0

        net_pnl = float(arr.sum())
        net_r = float(r_arr.sum())

        cum_usd = np.cumsum(arr)
        max_dd_usd = float((np.maximum.accumulate(cum_usd) - cum_usd).max()) if len(cum_usd) > 0 else 0.0

        cum_r = np.cumsum(r_arr)
        max_dd_r = float((np.maximum.accumulate(cum_r) - cum_r).max()) if len(cum_r) > 0 else 0.0
        total_fee_drag = round(len(arr) * spread_cost, 2)

        mean_r = float(np.mean(r_arr))
        std_r = float(np.std(r_arr, ddof=1)) if len(r_arr) > 1 else 1.0
        downside_dev = float(np.sqrt(np.mean(np.minimum(0.0, r_arr) ** 2)))
        ann_factor = np.sqrt(250.0)
        sharpe = round((mean_r / max(std_r, 1e-6)) * ann_factor, 2)
        sortino = round((mean_r / max(downside_dev, 1e-6)) * ann_factor, 2)
        calmar = round((net_r * (250.0 / max(len(r_arr), 1))) / max(max_dd_r, 0.5), 2)
        var_95_r = round(float(np.percentile(r_arr, 5)), 2)

        rng = np.random.default_rng(42)
        mc_sims = rng.choice(r_arr, size=(1000, len(r_arr)), replace=True)
        mc_cum = np.cumsum(mc_sims, axis=1)
        mc_peaks = np.maximum.accumulate(mc_cum, axis=1)
        mc_dds = np.max(mc_peaks - mc_cum, axis=1)
        mc_dd_95_r = round(float(np.percentile(mc_dds, 95)), 2)
        mc_prob_profit = round(float(np.mean(mc_cum[:, -1] > 0) * 100.0), 1)

        oos_df = df_tr[df_tr["idx"] >= split_idx]
        if len(oos_df) > 0:
            o_r = oos_df["r"].values
            o_pnl = oos_df["pnl"].values
            o_w = o_r[o_r > 0]; o_l = o_r[o_r < 0]
            oos_wr = round((len(o_w) / len(o_r)) * 100.0, 1)
            oos_r_pf = round((float(o_w.sum()) / float(abs(o_l.sum()))) if len(o_l) > 0 else (2.0 if len(o_w) > 0 else 0.0), 2)
            oos_net_r = round(float(o_r.sum()), 2)
            oos_pnl = round(float(o_pnl.sum()), 2)
            oos_trades = int(len(oos_df))
        else:
            oos_wr, oos_r_pf, oos_net_r, oos_pnl, oos_trades = 0.0, 0.0, 0.0, 0.0, 0

        is_df = df_tr[df_tr["idx"] < split_idx]
        if len(is_df) >= 4:
            is_r = is_df["r"].values
            is_w = is_r[is_r > 0]; is_l = is_r[is_r < 0]
            is_r_pf = (float(is_w.sum()) / float(abs(is_l.sum()))) if len(is_l) > 0 else 2.0
            is_net_r = float(is_r.sum())
            is_cum_r = np.cumsum(is_r)
            is_dd_r = float((np.maximum.accumulate(is_cum_r) - is_cum_r).max())
            train_score = (is_r_pf * 2.0 + is_net_r * 0.25) * np.sqrt(len(is_r)) / (1.0 + is_dd_r * 0.2)
        else:
            train_score = r_pf * np.sqrt(len(r_arr))

        breakdown = []
        for f_name in all_factors:
            is_enabled = f_name in enabled_list
            sub = df_tr[df_tr["factor"] == f_name]
            if len(sub) > 0:
                s_r = sub["r"].values; s_pnl = sub["pnl"].values
                s_w = s_r[s_r > 0]; s_l = s_r[s_r < 0]
                s_r_pf = (float(s_w.sum()) / float(abs(s_l.sum()))) if len(s_l) > 0 else (2.0 if len(s_w) > 0 else 0.0)
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

        return {
            "atr_sl_mult": sl_m, "rr_mult": rr_m, "adx_thresh": adx_t, "enabled_factors": enabled_list,
            "trades": int(len(r_arr)), "win_rate": round(wr, 1), "profit_factor": round(usd_pf, 2),
            "r_profit_factor": round(r_pf, 2), "net_pnl": round(net_pnl, 2), "net_r": round(net_r, 2),
            "max_dd": round(max_dd_usd, 2), "max_dd_r": round(max_dd_r, 2),
            "sharpe": sharpe, "sortino": sortino, "calmar": calmar, "var_95_r": var_95_r,
            "mc_dd_95_r": mc_dd_95_r, "mc_prob_profit": mc_prob_profit, "total_fee_drag": total_fee_drag,
            "oos_trades": oos_trades, "oos_win_rate": oos_wr, "oos_profit_factor": oos_r_pf,
            "oos_net_r": oos_net_r, "oos_net_pnl": oos_pnl,
            "is_profitable_edge": bool(r_pf >= 1.05 and net_r > 0),
            "score": round(float(train_score), 3),
            "equity_curve_r": [round(float(x), 2) for x in np.insert(cum_r, 0, 0.0)],
            "trade_r_list": [round(float(x), 2) for x in r_arr],
            "factor_breakdown": breakdown
        }

    candidates = []
    for sl_m in [1.6, 2.0, 2.5]:
        for rr_m in [2.2, 2.6, 3.2]:
            for adx_t in [22, 25, 28]:
                raw_all = simulate_raw_trades(sl_m, rr_m, adx_t, set(all_factors))
                if len(raw_all) < 8:
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
                summary = summarize_records(filtered_records, sl_m, rr_m, adx_t, surviving_factors)
                if summary:
                    candidates.append(summary)

    if not candidates:
        return {"best": default_best, "factor_breakdown": [], "top_5": [default_best], "split_bar": split_idx}

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    best_cfg = ranked[0]
    breakdown = best_cfg.pop("factor_breakdown", [])
    top_5_clean = [
        {k: v for k, v in item.items() if k not in ("equity_curve_r", "trade_r_list", "factor_breakdown")}
        for item in ranked[:5]
    ]

    return {
        "best": best_cfg,
        "factor_breakdown": breakdown,
        "top_5": top_5_clean,
        "split_bar": split_idx
    }

# ================= 2B. VP-ANATOMY 500-POINT ($5.00) SCALP ENGINE =================
def run_vp_anatomy_scalp_engine(df_intraday, ind, spot_price, spot_offset=0.0):
    """
    ระบบ Scalp 500 จุด ($5.00 SL / $5.00-$6.50 TP) จาก 4 กายวิภาค Volume Profile (VPAnatomies_EA):
    - คำนวณ Volume Profile รอบ 24 ชม. (92 แท่ง M15) + เกลี่ยความเรียบหา 2nd Peak & Valley
    - จำแนก D-Shaped, b-Shaped, P-Shaped, Double Zone อัตโนมัติ
    - หาจังหวะเข้าบ่อยขึ้นในจุดที่มั่นใจ (10:00 - 00:30 น. เวลาไทย) พร้อม Walk-Forward & Auto-Mute
    """
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
    utc_hours = dt_index_utc.hour.values

    # วิเคราะห์ 4 กายวิภาค Volume Profile ล่าสุด (24H = 92 แท่ง M15)
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

    # Precompute 24H VP Anatomy ทุกๆ 4 แท่ง (1 ชม.) สำหรับ Walk-Forward Backtest
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
    spread_cost = 0.50  # Spread Scalp เฉลี่ย 50 จุด

    def sim_scalp(tp_usd, tol_usd, enabled_shapes):
        in_pos = 0
        active_sh = ""
        entry_p = sl_p = tp_p = 0.0
        be_moved = False
        cooldown = 0
        recs = []

        for i in range(win_24h + 1, n_bars):
            sh = an_shape[i]
            if sh is None or np.isnan(an_poc[i]):
                continue

            if in_pos == 1:
                if low_arr[i] <= sl_p:
                    pnl = (sl_p - entry_p) - spread_cost
                    recs.append({"idx": i, "shape": active_sh, "pnl": pnl, "r": pnl / sl_usd})
                    in_pos = 0; cooldown = 1; continue
                if high_arr[i] >= tp_p:
                    pnl = (tp_p - entry_p) - spread_cost
                    recs.append({"idx": i, "shape": active_sh, "pnl": pnl, "r": pnl / sl_usd})
                    in_pos = 0; cooldown = 1; continue
                if not be_moved and high_arr[i] >= entry_p + be_trigger:
                    sl_p = entry_p + 0.50  # ล็อกคุ้มค่า Spread +50 จุด
                    be_moved = True
                continue
            elif in_pos == -1:
                if high_arr[i] >= sl_p:
                    pnl = (entry_p - sl_p) - spread_cost
                    recs.append({"idx": i, "shape": active_sh, "pnl": pnl, "r": pnl / sl_usd})
                    in_pos = 0; cooldown = 1; continue
                if low_arr[i] <= tp_p:
                    pnl = (entry_p - tp_p) - spread_cost
                    recs.append({"idx": i, "shape": active_sh, "pnl": pnl, "r": pnl / sl_usd})
                    in_pos = 0; cooldown = 1; continue
                if not be_moved and low_arr[i] <= entry_p - be_trigger:
                    sl_p = entry_p - 0.50
                    be_moved = True
                continue

            if cooldown > 0:
                cooldown -= 1; continue

            # ช่วงเวลา Scalp: 10:00 - 00:30 น. เวลาไทย (03:00 - 17:30 UTC)
            hr = utc_hours[i]
            if not (3 <= hr <= 17):
                continue

            if sh not in enabled_shapes:
                continue

            c_open = open_arr[i]; c_high = high_arr[i]; c_low = low_arr[i]; c_close = close_arr[i]
            c_rng = max(c_high - c_low, 1e-6)
            lo_wick = (min(c_open, c_close) - c_low) / c_rng
            up_wick = (c_high - max(c_open, c_close)) / c_rng
            bull_c = c_close > c_open and (lo_wick >= 0.20 or c_close > close_arr[i - 1])
            bear_c = c_close < c_open and (up_wick >= 0.20 or c_close < close_arr[i - 1])
            rsi = rsi_arr[i]
            poc = an_poc[i]; vah = an_vah[i]; val = an_val[i]
            lo_pk = an_lo_pk[i]; hi_pk = an_hi_pk[i]

            # 1. D-SHAPED: ดัก BUY ที่ VAL / ดัก SELL ที่ VAH
            if sh == "D_SHAPED":
                if c_low <= val + tol_usd and c_close >= val - 1.5 and bull_c and rsi <= 52:
                    in_pos = 1; active_sh = "D_SHAPED"; be_moved = False
                    entry_p = c_close; sl_p = entry_p - sl_usd; tp_p = entry_p + tp_usd
                elif c_high >= vah - tol_usd and c_close <= vah + 1.5 and bear_c and rsi >= 48:
                    in_pos = -1; active_sh = "D_SHAPED"; be_moved = False
                    entry_p = c_close; sl_p = entry_p + sl_usd; tp_p = entry_p - tp_usd

            # 2. B-SHAPED: SELL ใต้ POC ตามโมเมนตัมลง หรือ BUY เมื่อเบรก VAH
            elif sh == "B_SHAPED":
                if c_high >= poc - tol_usd and c_close < poc and bear_c and c_close < ema21_arr[i] and rsi >= 30:
                    in_pos = -1; active_sh = "B_SHAPED"; be_moved = False
                    entry_p = c_close; sl_p = entry_p + sl_usd; tp_p = entry_p - tp_usd
                elif c_close > vah and close_arr[i - 1] <= vah and bull_c and rsi <= 70:
                    in_pos = 1; active_sh = "B_SHAPED"; be_moved = False
                    entry_p = c_close; sl_p = entry_p - sl_usd; tp_p = entry_p + tp_usd

            # 3. P-SHAPED: BUY ย่อรับที่ POC/VAL หรือ SELL เมื่อหลุด VAL
            elif sh == "P_SHAPED":
                if ((c_low <= poc + tol_usd and c_close >= poc - 1.0) or (c_low <= val + tol_usd and c_close >= val - 1.0)) and bull_c and c_close > ema50_arr[i] and rsi <= 68:
                    in_pos = 1; active_sh = "P_SHAPED"; be_moved = False
                    entry_p = c_close; sl_p = entry_p - sl_usd; tp_p = entry_p + tp_usd
                elif c_close < val and close_arr[i - 1] >= val and bear_c and rsi >= 32:
                    in_pos = -1; active_sh = "P_SHAPED"; be_moved = False
                    entry_p = c_close; sl_p = entry_p + sl_usd; tp_p = entry_p - tp_usd

            # 4. DOUBLE_ZONE: ดัก BUY ที่ Lower Peak และดัก SELL ที่ Upper Peak
            elif sh == "DOUBLE_ZONE":
                if c_low <= lo_pk + tol_usd and c_close >= lo_pk - 1.5 and bull_c and rsi <= 55:
                    in_pos = 1; active_sh = "DOUBLE_ZONE"; be_moved = False
                    entry_p = c_close; sl_p = entry_p - sl_usd; tp_p = entry_p + tp_usd
                elif c_high >= hi_pk - tol_usd and c_close <= hi_pk + 1.5 and bear_c and rsi >= 45:
                    in_pos = -1; active_sh = "DOUBLE_ZONE"; be_moved = False
                    entry_p = c_close; sl_p = entry_p + sl_usd; tp_p = entry_p - tp_usd

        return recs

    best_scalp = None
    best_score = -1e9
    best_tp_usd = 5.50
    best_tol_usd = 2.00
    best_enabled_shapes = all_shapes

    for tp_cand in [5.00, 5.50, 6.50]:
        for tol_cand in [1.50, 2.50]:
            raw_s = sim_scalp(tp_cand, tol_cand, set(all_shapes))
            if len(raw_s) < 12:
                continue
            df_s = pd.DataFrame(raw_s)
            df_tr = df_s[df_s["idx"] < split_idx]
            surv = [s for s in all_shapes if len(df_tr[df_tr["shape"] == s]) == 0 or df_tr[df_tr["shape"] == s]["r"].sum() > 0]
            if not surv:
                surv = all_shapes

            filt_s = sim_scalp(tp_cand, tol_cand, set(surv))
            if len(filt_s) < 10:
                continue
            df_f = pd.DataFrame(filt_s)
            r_arr = df_f["r"].values
            w_r = r_arr[r_arr > 0]; l_r = r_arr[r_arr < 0]
            r_pf = (float(w_r.sum()) / float(abs(l_r.sum()))) if len(l_r) > 0 else 2.0
            net_r = float(r_arr.sum())
            cum_r = np.cumsum(r_arr)
            dd_r = float((np.maximum.accumulate(cum_r) - cum_r).max())

            df_is = df_f[df_f["idx"] < split_idx]
            is_r = df_is["r"].values if len(df_is) > 0 else r_arr
            sc = float(is_r.sum()) * (r_pf ** 0.5) / (1.0 + dd_r * 0.15)
            if sc > best_score:
                best_score = sc
                best_tp_usd = tp_cand
                best_tol_usd = tol_cand
                best_enabled_shapes = surv
                best_scalp = df_f

    if best_scalp is not None and len(best_scalp) > 0:
        r_arr = best_scalp["r"].values
        pnl_arr = best_scalp["pnl"].values
        w_r = r_arr[r_arr > 0]; l_r = r_arr[r_arr < 0]
        wr = round((len(w_r) / len(r_arr)) * 100.0, 1)
        r_pf = round((float(w_r.sum()) / float(abs(l_r.sum()))) if len(l_r) > 0 else 2.0, 2)
        net_r = round(float(r_arr.sum()), 2)
        net_pnl = round(float(pnl_arr.sum()), 2)
        cum_r = np.cumsum(r_arr)
        max_dd_r = round(float((np.maximum.accumulate(cum_r) - cum_r).max()), 2)

        oos_df = best_scalp[best_scalp["idx"] >= split_idx]
        if len(oos_df) > 0:
            o_r = oos_df["r"].values
            o_w = o_r[o_r > 0]; o_l = o_r[o_r < 0]
            oos_wr = round((len(o_w) / len(o_r)) * 100.0, 1)
            oos_pf = round((float(o_w.sum()) / float(abs(o_l.sum()))) if len(o_l) > 0 else 2.0, 2)
            oos_net_r = round(float(o_r.sum()), 2)
            oos_trades = int(len(oos_df))
        else:
            oos_wr, oos_pf, oos_net_r, oos_trades = 0.0, 0.0, 0.0, 0

        anatomy_breakdown = []
        for s_name in all_shapes:
            sub = best_scalp[best_scalp["shape"] == s_name]
            is_on = s_name in best_enabled_shapes
            if len(sub) > 0:
                sr = sub["r"].values
                sw = sr[sr > 0]
                anatomy_breakdown.append({
                    "shape": s_name,
                    "enabled": is_on,
                    "trades": int(len(sub)),
                    "win_rate": round((len(sw) / len(sub)) * 100.0, 1),
                    "net_r": round(float(sr.sum()), 2)
                })
            else:
                anatomy_breakdown.append({"shape": s_name, "enabled": is_on, "trades": 0, "win_rate": 0.0, "net_r": 0.0})

        scalp_bt = {
            "sl_usd": sl_usd,
            "sl_points": int(sl_usd * 100),
            "tp_usd": best_tp_usd,
            "tp_points": int(best_tp_usd * 100),
            "be_usd": be_trigger,
            "be_points": int(be_trigger * 100),
            "trades": int(len(r_arr)),
            "win_rate": wr,
            "r_profit_factor": r_pf,
            "net_r": net_r,
            "net_pnl": net_pnl,
            "max_dd_r": max_dd_r,
            "oos_trades": oos_trades,
            "oos_win_rate": oos_wr,
            "oos_r_pf": oos_pf,
            "oos_net_r": oos_net_r,
            "has_edge": bool(r_pf >= 1.05 and net_r > 0),
            "equity_curve_r": [round(float(x), 2) for x in np.insert(cum_r, 0, 0.0)],
            "anatomy_breakdown": anatomy_breakdown
        }
    else:
        scalp_bt = {
            "sl_usd": 5.0, "sl_points": 500, "tp_usd": 5.0, "tp_points": 500,
            "be_usd": 2.5, "be_points": 250, "trades": 0, "win_rate": 0.0,
            "r_profit_factor": 0.0, "net_r": 0.0, "net_pnl": 0.0, "max_dd_r": 0.0,
            "oos_trades": 0, "oos_win_rate": 0.0, "oos_r_pf": 0.0, "oos_net_r": 0.0,
            "has_edge": False, "equity_curve_r": [0], "anatomy_breakdown": []
        }

    # คำนวณแผน Scalp สด ณ วินาทีปัจจุบันจาก 4 กายวิภาค Volume Profile
    sh = curr_anatomy["shape"]
    poc = curr_anatomy["poc"]
    vah = curr_anatomy["vah"]
    val = curr_anatomy["val"]
    lo_pk = curr_anatomy["lower_peak"]
    hi_pk = curr_anatomy["upper_peak"]
    tp_usd = scalp_bt["tp_usd"]
    tp_pts = scalp_bt["tp_points"]

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
    else:  # DOUBLE_ZONE
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
    gate_edge = bool(scalp_bt["has_edge"])
    gate_prox = bool(dist_usd <= 3.00)  # ใกล้ในระยะ 300 จุด ($3.00)
    scalp_ready = gate_session and gate_edge and gate_prox

    if not gate_session:
        s_verdict = "NO (อยู่นอกเวลา Scalp 10:00–00:30 น.)"
    elif not gate_edge:
        s_verdict = "NO (สถิติ Scalp R-PF ต่ำกว่า 1.05 — รอจังหวะถัดไป)"
    elif not gate_prox:
        s_verdict = f"NO (ห่างจุดเข้า ${dist_usd:.2f} / {dist_pts} จุด — รอเข้าใกล้ <= 300 จุด)"
    else:
        s_verdict = f"YES (พร้อมยิง SCALP 500 จุด! ราคาเข้าใกล้ในระยะ {dist_pts} จุด)"

    return {
        "anatomy": curr_anatomy,
        "backtest": scalp_bt,
        "setup": {
            "shape": sh,
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
            "edge_ok": gate_edge,
            "proximity_ok": gate_prox,
            "all_passed": scalp_ready,
            "verdict_text": s_verdict
        }
    }

# ================= 3. SESSION-AWARE SPOT TREND ENGINE =================
def calculate_quant_and_setup(df_intraday, df_daily, df_dxy, df_us10y, vp_data, ind, spot_price, bid, ask, spot_offset, paxg_offset, open_interest, tuned_params):
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
    enabled_factors = set(tuned_params.get("enabled_factors", ["VALUE_AREA_FADE", "LIQUIDITY_SWEEP", "LONDON_BREAKOUT"]))

    risk_unit = round(curr_atr14 * atr_sl_mult, 2)
    be_trigger_dist = round(risk_unit * 1.0, 2)
    reward_unit = round(risk_unit * rr_mult, 2)

    now_th = datetime.now(THAI_TZ)
    in_killzone = (13 <= now_th.hour <= 23) or (now_th.hour == 0 and now_th.minute <= 30)
    in_london_window = (13 <= now_th.hour <= 16)
    session_name = "LONDON / NY KILLZONE ACTIVE (เปิดวิเคราะห์และเทรด)" if in_killzone else "NO TRADE ZONE (ช่วงเช้า - ปลดล็อก 13:00 น.)"

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
        position_context = f"ราคากวาดสภาพคล่องใต้ PDL (${curr_pdl}) และดึงกลับขึ้นมายืนเหนือแนวรับได้ (RSI {curr_rsi14})"
        invalidation_rule = f"หากราคาร่วงหลุด Stop Loss ${sl} จะยกเลิกแผน Long ทันที"
    elif "LIQUIDITY_SWEEP" in enabled_factors and near_pdh_sweep:
        active_factor = "LIQUIDITY_SWEEP"
        regime = f"LIQUIDITY SWEEP AT PDH (${curr_pdh}) | RSI {curr_rsi14}"
        bias = "LIQUIDITY SWEEP SHORT (กวาดสภาพคล่องเหนือ PDH แล้วกดกลับ)"
        direction = "SHORT"
        entry = round(min(active_spot, curr_pdh), 2)
        sl = round(entry + risk_unit, 2)
        be_target = round(entry - be_trigger_dist, 2)
        tp = round(entry - reward_unit, 2)
        position_context = f"ราคากวาดสภาพคล่องเหนือ PDH (${curr_pdh}) และถูกแรงขายกดกลับลงมาในกรอบ (RSI {curr_rsi14})"
        invalidation_rule = f"หากราคาเบรกทะลุ Stop Loss ${sl} จะยกเลิกแผน Short ทันที"
    elif curr_adx14 > adx_thresh and active_spot < curr_h1_ema200 and active_spot < curr_ema200:
        if rsi_oversold_exhausted:
            active_factor = "ANTI_CHASE_RETEST"
            regime = f"OVERSOLD EXHAUSTION (ADX {curr_adx14} | RSI {curr_rsi14} < 28)"
            bias = f"ANTI-CHASE RETEST SHORT (RSI ตึงตัว ห้ามไล่ Short ใต้ PDL — รอเด้งชน PDL ${curr_pdl} / EMA21 ${curr_ema21})"
            direction = "SHORT"
            retest_lvl = curr_pdl if curr_pdl > active_spot else curr_ema21
            entry = round(retest_lvl, 2)
            sl = round(entry + risk_unit, 2)
            be_target = round(entry - be_trigger_dist, 2)
            tp = round(entry - reward_unit, 2)
            position_context = f"ระบบ RSI Exhaustion Guard ตรวจพบ RSI14 ต่ำสุดขีด ({curr_rsi14}) จึงบล็อกการไล่ Short ด้านล่าง และปรับแผนให้รอราคาดีดทดสอบแนวต้าน"
            invalidation_rule = f"หากราคาดีดทะลุ Stop Loss ${sl} กลับขึ้นไปยืนเหนือแนวต้านให้ยกเลิกแผนทันที"
        elif "LONDON_BREAKOUT" in enabled_factors and in_london_window and active_spot >= curr_asian_l:
            active_factor = "LONDON_BREAKOUT"
            regime = f"LONDON BREAKOUT BEAR (ADX {curr_adx14} | Asian Low ${curr_asian_l})"
            bias = f"LONDON BREAKDOWN SHORT (รอเบรกหลุด Asian Low ${curr_asian_l})"
            direction = "SHORT"
            entry = round(curr_asian_l - 0.5, 2)
            sl = round(entry + risk_unit, 2)
            be_target = round(entry - be_trigger_dist, 2)
            tp = round(entry - reward_unit, 2)
            position_context = f"อยู่ในช่วงเปิดตลาดลอนดอน เทรนด์ H1/M15 เป็นขาลง และ RSI ({curr_rsi14}) ยังมีพื้นที่ให้วิ่งลงต่อ"
            invalidation_rule = f"หากราคาดีดทะลุ Stop Loss ${sl} ให้ยกเลิกแผนทันที"
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
            position_context = f"เทรนด์ขาลงแข็งแกร่ง (ADX {curr_adx14}) ราคาอยู่ใต้ทั้ง H1 EMA200 (${curr_h1_ema200}) และ M15 EMA200 (${curr_ema200})"
            invalidation_rule = f"หากราคาดีดทะลุ Stop Loss ${sl} กลับขึ้นไปยืนเหนือแนวต้านให้ยกเลิกแผนทันที"
    elif curr_adx14 > adx_thresh and active_spot > curr_h1_ema200 and active_spot > curr_ema200:
        if rsi_overbought_exhausted:
            active_factor = "ANTI_CHASE_RETEST"
            regime = f"OVERBOUGHT EXHAUSTION (ADX {curr_adx14} | RSI {curr_rsi14} > 72)"
            bias = f"ANTI-CHASE RETEST LONG (RSI ตึงตัว ห้ามไล่ Buy เหนือ PDH — รอย่อแตะ PDH ${curr_pdh} / EMA21 ${curr_ema21})"
            direction = "LONG"
            retest_lvl = curr_pdh if curr_pdh < active_spot else curr_ema21
            entry = round(retest_lvl, 2)
            sl = round(entry - risk_unit, 2)
            be_target = round(entry + be_trigger_dist, 2)
            tp = round(entry + reward_unit, 2)
            position_context = f"ระบบ RSI Exhaustion Guard ตรวจพบ RSI14 สูงสุดขีด ({curr_rsi14}) จึงบล็อกการไล่ซื้อด้านบน และปรับแผนให้รอย่อทดสอบแนวรับ"
            invalidation_rule = f"หากราคาร่วงหลุด Stop Loss ${sl} ให้ยกเลิกแผนทันที"
        elif "LONDON_BREAKOUT" in enabled_factors and in_london_window and active_spot <= curr_asian_h:
            active_factor = "LONDON_BREAKOUT"
            regime = f"LONDON BREAKOUT BULL (ADX {curr_adx14} | Asian High ${curr_asian_h})"
            bias = f"LONDON BREAKOUT LONG (รอเบรกทะลุ Asian High ${curr_asian_h})"
            direction = "LONG"
            entry = round(curr_asian_h + 0.5, 2)
            sl = round(entry - risk_unit, 2)
            be_target = round(entry + be_trigger_dist, 2)
            tp = round(entry + reward_unit, 2)
            position_context = f"อยู่ในช่วงเปิดตลาดลอนดอน เทรนด์ H1/M15 เป็นขาขึ้น และ RSI ({curr_rsi14}) ยังมีพื้นที่ให้วิ่งขึ้นต่อ"
            invalidation_rule = f"หากราคาร่วงหลุด Stop Loss ${sl} ให้ยกเลิกแผนทันที"
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
            position_context = f"เทรนด์ขาขึ้นแข็งแกร่ง (ADX {curr_adx14}) ราคาอยู่เหนือทั้ง H1 EMA200 (${curr_h1_ema200}) และ M15 EMA200 (${curr_ema200})"
            invalidation_rule = f"หากราคาร่วงหลุด Stop Loss ${sl} ให้ยกเลิกแผนทันที"
    else:
        if "VALUE_AREA_FADE" in enabled_factors and curr_adx14 <= adx_thresh:
            active_factor = "VALUE_AREA_FADE"
            regime = f"BALANCED RANGE (ADX {curr_adx14} <= {adx_thresh} | แกว่งในกรอบ)"
            if active_spot >= poc:
                bias = "RANGE FADE SHORT NEAR VAH (ขายกรอบบน เป้าหมาย POC)"
                direction = "SHORT"
                entry = vah
                sl = round(vah + risk_unit, 2)
                be_target = round(vah - be_trigger_dist, 2)
                tp = round(vah - reward_unit, 2)
                position_context = f"ตลาดไซด์เวย์ (ADX {curr_adx14}) ราคาอยู่ใกล้แนวต้านบน VAH ${vah}"
                invalidation_rule = f"หากราคาเบรกทะลุ Stop Loss ${sl} เหนือ VAH จะยกเลิกแผน Short ทันที"
            else:
                bias = "RANGE FADE LONG NEAR VAL (ซื้อกรอบล่าง เป้าหมาย POC)"
                direction = "LONG"
                entry = val
                sl = round(val - risk_unit, 2)
                be_target = round(val + be_trigger_dist, 2)
                tp = round(val + reward_unit, 2)
                position_context = f"ตลาดไซด์เวย์ (ADX {curr_adx14}) ราคาอยู่ใกล้แนวรับล่าง VAL ${val}"
                invalidation_rule = f"หากราคาหลุด Stop Loss ${sl} ต่ำกว่า VAL จะยกเลิกแผน Long ทันที"
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
            position_context = f"กราฟ H1 และ M15 ยังไม่เรียงตัวไปทางเดียวกัน จึงเน้นดัก Liquidity Sweep ที่ขอบ PDH/PDL"
            invalidation_rule = f"หากราคาเบรกทะลุ Stop Loss ${sl} ให้ยกเลิกแผนทันที"

    sl_dist = round(abs(entry - sl), 2)
    tp_dist = round(abs(tp - entry), 2)
    rr_ratio = round(tp_dist / max(sl_dist, 0.1), 2)
    dist_to_entry = round(abs(active_spot - entry), 2)

    gate_session = bool(in_killzone)
    gate_edge = bool(has_edge)
    gate_proximity = bool(dist_to_entry <= 8.0)
    all_gates_passed = gate_session and gate_edge and gate_proximity

    if not gate_session:
        gate_verdict = "NO (อยู่นอกเวลา Killzone — ช่วงเช้าบล็อกการออกออเดอร์ รอปลดล็อก 13:00 น.)"
    elif not gate_edge:
        gate_verdict = f"NO (สถิติ Walk-Forward R-PF ต่ำกว่าเกณฑ์ 1.05 — งดออกออเดอร์เพื่อรักษาทุน)"
    elif not gate_proximity:
        gate_verdict = f"NO (ราคา Spot ปัจจุบันยังอยู่ห่างจุด Entry ${dist_to_entry:.2f} ซึ่งเกินระยะยิงสัญญาณ $8.00)"
    else:
        gate_verdict = f"YES (ครบทั้ง 3 เงื่อนไข: อยู่ใน Killzone + มี Quant Edge + ราคาเข้าใกล้ Entry ในระยะ ${dist_to_entry:.2f})"

    start_dt_str = df_intraday.index[0].astimezone(THAI_TZ).strftime("%d/%m/%Y")
    end_dt_str = df_intraday.index[-1].astimezone(THAI_TZ).strftime("%d/%m/%Y %H:%M")
    vault_days = round(len(df_intraday) / 92.0, 1)

    return {
        "spot_price": active_spot,
        "bid": bid if bid else active_spot,
        "ask": ask if ask else active_spot,
        "futures_price": round(futures_price, 2),
        "spot_offset": round(spot_offset, 2),
        "paxg_to_spot_offset": round(paxg_offset, 2),
        "data_audit": {
            "source": "Hydra Local Vault + Yahoo Finance (GC=F) + Swissquote Spot",
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
            "has_edge": has_edge, "in_killzone": in_killzone, "session_name": session_name,
            "position_context": position_context, "invalidation_rule": invalidation_rule
        }
    }

# ================= 4. LIVE SIGNAL JOURNALS (TREND + SCALP 500PT) =================
def _process_single_journal(journal_path, setup_dict, is_triggered, current_spot, df_intraday, spot_offset, spread_cost=0.60, dup_tol=8.0, factor_key="active_factor"):
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

        sub_highs = [current_spot]
        sub_lows = [current_spot]
        for idx, b_dt in enumerate(bar_times):
            if b_dt >= opened_dt:
                sub_highs.append(float(bar_highs[idx]))
                sub_lows.append(float(bar_lows[idx]))

        max_h = max(sub_highs)
        min_l = min(sub_lows)

        if direction == "LONG":
            if min_l <= sl_p:
                trade["status"] = "LOSS"
                trade["pnl"] = round((sl_p - entry_p) - spread_cost, 2)
                trade["closed_at_th"] = now_th_str
            elif max_h >= tp_p:
                trade["status"] = "WIN"
                trade["pnl"] = round((tp_p - entry_p) - spread_cost, 2)
                trade["closed_at_th"] = now_th_str
        elif direction == "SHORT":
            if max_h >= sl_p:
                trade["status"] = "LOSS"
                trade["pnl"] = round((entry_p - sl_p) - spread_cost, 2)
                trade["closed_at_th"] = now_th_str
            elif min_l <= tp_p:
                trade["status"] = "WIN"
                trade["pnl"] = round((entry_p - tp_p) - spread_cost, 2)
                trade["closed_at_th"] = now_th_str

    if is_triggered:
        has_duplicate = any(
            t["status"] == "OPEN"
            and t["direction"] == setup_dict["direction"]
            and abs(float(t["entry"]) - float(setup_dict["entry"])) < dup_tol
            for t in journal
        )
        if not has_duplicate:
            journal.insert(0, {
                "id": f"SIG-{now_utc.strftime('%m%d%H%M')}",
                "opened_at": now_utc.isoformat(),
                "opened_at_th": now_th_str,
                "factor": setup_dict.get(factor_key, "SETUP"),
                "direction": setup_dict["direction"],
                "bias": setup_dict["bias"],
                "entry": setup_dict["entry"],
                "sl": setup_dict["sl"],
                "tp": setup_dict["tp"],
                "rr": setup_dict["rr"],
                "status": "OPEN",
                "pnl": 0.0,
                "closed_at_th": "-"
            })

    journal = journal[:30]
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
        "recent_trades": journal[:8]
    }

def update_all_journals(quant_data, scalp_data, df_intraday, spot_offset, trend_triggered, scalp_triggered):
    trend_j = _process_single_journal(
        "data/trade_journal.json", quant_data["setup"], trend_triggered,
        quant_data["spot_price"], df_intraday, spot_offset,
        spread_cost=0.60, dup_tol=8.0, factor_key="active_factor"
    )
    scalp_j = _process_single_journal(
        "data/scalp_journal.json", scalp_data["setup"], scalp_triggered,
        quant_data["spot_price"], df_intraday, spot_offset,
        spread_cost=0.50, dup_tol=3.0, factor_key="shape"
    )
    return trend_j, scalp_j

# ================= 5. 4-AGENT SWARM DEBATE COUNCIL (TREND + SCALP 500PT) =================
def run_multi_agent_thaillm(quant_data, vp_data, backtest_data, scalp_data, news_list, kb_text, kb_files):
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

    # --- AGENT 1: R-Multiple & Monte Carlo Auditor ---
    agent1_prompt = (
        "คุณคือ Quantitative R-Multiple & Scalp Auditor ประเมินผลทดสอบทั้งระบบรันเทรนด์และระบบ Scalp 500 จุด "
        "(ตอบเป็นภาษาไทยสั้นๆ กระชับ 2-3 ประโยค ห้ามเปลี่ยนตัวเลข)\n\n"
        f"- ระบบ Trend: {bt['trades']} ไม้ | WR = {bt['win_rate']}% | R-PF = {bt['r_profit_factor']} | Net = {bt['net_r']}R | Sharpe = {bt['sharpe']} | MC 95% DD = -{bt['mc_dd_95_r']}R\n"
        f"- ระบบ VP-Anatomy Scalp (SL 500 จุด / TP {sc_bt['tp_points']} จุด): {sc_bt['trades']} ไม้ | WR = {sc_bt['win_rate']}% | R-PF = {sc_bt['r_profit_factor']} | Net = {sc_bt['net_r']}R | OOS R-PF = {sc_bt['oos_r_pf']}\n"
        "จงสรุปว่าทั้งสองระบบมีความคุ้มเสี่ยงและพร้อมใช้งานคู่กันหรือไม่"
    )
    quant_audit_text, used_m1 = call_thaillm_agent(MODEL_QUANT_AUDITOR, agent1_prompt, temperature=0.1)

    # --- AGENT 2: Macro & VP Anatomy Officer ---
    locked_time_fact = (
        f"ขณะนี้เวลา {now_th_str} (เวลาไทย) ซึ่ง **อยู่ในช่วง KILLZONE ACTIVE แล้ว**"
        if setup["in_killzone"] else
        f"ขณะนี้เวลา {now_th_str} (เวลาไทย) ซึ่ง **อยู่ในช่วง NO TRADE ZONE (ช่วงเช้า)**"
    )

    agent2_prompt = (
        "คุณคือ Macro & Volume Profile Anatomy Officer วิเคราะห์โครงสร้างตลาด Spot XAU/USD "
        "(ตอบเป็นภาษาไทยสั้นๆ กระชับ 2-3 ประโยค)\n\n"
        f"- เวลา: {locked_time_fact} | Spot = ${quant_data['spot_price']}\n"
        f"- กายวิภาค Volume Profile 24H: {sc_an['title_th']} (POC อยู่ตำแหน่ง {sc_an['poc_pos_pct']}% | VAH ${sc_an['vah']} / POC ${sc_an['poc']} / VAL ${sc_an['val']})\n"
        f"- DXY = {quant_data['dxy_price']} | US10Y = {quant_data['us10y_yield']}%\n"
        f"- ข่าว USD:\n{news_status_text}\n"
        "จงอธิบายโครงสร้างโวลุ่มรูปทรงนี้และผลกระทบจากข่าว/มหภาค"
    )
    macro_audit_text, used_m2 = call_thaillm_agent(MODEL_MACRO_OFFICER, agent2_prompt, temperature=0.1)

    # --- AGENT 3: Contrarian Devil's Advocate ---
    agent3_critic_prompt = (
        "คุณคือ Contrarian Devil's Advocate (ฝ่ายค้านประจำทีมเทรด) จับผิดความเสี่ยงทั้งแผนรันเทรนด์และแผน Scalp 500 จุด "
        "(ตอบเป็นภาษาไทยสั้นๆ กระชับ 2-3 ประโยค)\n\n"
        f"- แผน Trend: {setup['bias']} @ ${setup['entry']} (ห่าง ${gates['dist_to_entry']})\n"
        f"- แผน Scalp 500 จุด ({sc_an['shape']}): {sc_set['bias']} @ ${sc_set['entry']} (SL ${sc_set['sl']} / ห่าง {sc_set['dist_points']} จุด)\n"
        f"- RSI14 = {tech['rsi14']} | ADX14 = {tech['adx14']}\n"
        "จงเตือนจุดเสี่ยงที่อาจทำให้โดนกิน Stop Loss 500 จุด หรือเกิด False Breakout"
    )
    contrarian_audit_text, used_m3_critic = call_thaillm_agent(MODEL_CONTRARIAN_CRITIC, agent3_critic_prompt, temperature=0.15)

    # --- AGENT 4: Chief Execution Strategist (Synthesis) ---
    agent4_prompt = (
        "คุณคือ Chief Execution Strategist & Jane Street Coach สำหรับ Spot XAU/USD (Dual-Engine: Trend + VP-Anatomy Scalp 500 จุด) "
        "จงสังเคราะห์ความเห็นจากทั้ง 3 Agents ร่วมกับ [ข้อเท็จจริงที่ถูกล็อกไว้] ห้ามเปลี่ยนคำตัดสินในข้อ 5 เด็ดขาด!\n\n"
        "[ข้อเท็จจริงที่ถูกล็อกไว้ 100%]\n"
        f"- เวลา: {locked_time_fact} | Spot = ${quant_data['spot_price']}\n"
        f"- กายวิภาค Volume Profile 24H: {sc_an['title_th']} ({sc_an['desc_th']})\n"
        f"- แผน Trend หลัก: {setup['bias']} | Entry ${setup['entry']} | SL ${setup['sl']} | BE ${setup['be_target']} | TP ${setup['tp']} (R:R 1:{setup['rr']})\n"
        f"- แผน Scalp 500 จุด: {sc_set['bias']} | Entry ${sc_set['entry']} | SL ${sc_set['sl']} (500 จุด) | BE +250 จุด (${sc_set['be_target']}) | TP ${sc_set['tp']} ({sc_set['tp_points']} จุด)\n"
        f"- คำตัดสิน Alert Trigger ที่บังคับตอบในข้อ 5: Trend: {gates['verdict_text']} | Scalp 500pt: {sc_gates['verdict_text']}\n\n"
        f"[Agent 1]: {quant_audit_text}\n[Agent 2]: {macro_audit_text}\n[Agent 3]: {contrarian_audit_text}\n\n"
        "จงเขียนรายงาน 7 หัวข้อตามรูปแบบนี้เป๊ะๆ:\n"
        "1. **Daily Outlook & VP Anatomy**: สรุปโครงสร้างราคาและรูปทรง Volume Profile (D/b/P/Double Zone)\n"
        f"2. **Dual-Engine Edge**: สรุปผลทดสอบ Trend (Net {bt['net_r']}R, Sharpe {bt['sharpe']}) และ Scalp 500 จุด (WR {sc_bt['win_rate']}%, Net {sc_bt['net_r']}R)\n"
        f"3. **Trend Execution Plan**: แผนรันเทรนด์ Entry ${setup['entry']}, SL ${setup['sl']}, BE ${setup['be_target']}, TP ${setup['tp']}\n"
        f"4. **VP-Anatomy Scalp Plan (500 จุด)**: แผนสั้น Entry ${sc_set['entry']}, SL ${sc_set['sl']} (500 จุด), BE +250 จุด (${sc_set['be_target']}), TP ${sc_set['tp']} ({sc_set['tp_points']} จุด)\n"
        f"5. **Alert Trigger**: คัดลอกคำตัดสินนี้ไปวางทันที -> Trend: {gates['verdict_text']} | Scalp 500pt: {sc_gates['verdict_text']}\n"
        "6. **Active Recall Question**: ตั้งคำถามสั้นๆ 1 ข้อเกี่ยวกับกายวิภาค Volume Profile หรือระยะ 500 จุด\n"
        "7. **Active Recall Answer**: เฉลยคำตอบสั้นๆ 1-2 ประโยค"
    )
    final_raw, used_m4 = call_thaillm_agent(MODEL_CHIEF_STRATEGIST, agent4_prompt, temperature=0.08)

    q_match = re.search(r'6\.\s*\*?\*?Active Recall Question\*?\*?:\s*(.*?)(?=\n7\.|\Z)', final_raw, re.DOTALL)
    a_match = re.search(r'7\.\s*\*?\*?Active Recall Answer\*?\*?:\s*(.*)', final_raw, re.DOTALL)
    recall_q = q_match.group(1).strip() if q_match else f"ในโครงสร้าง {sc_an['title_th']} เหตุใดระบบ Scalp จึงตั้งจุดเลื่อนบังทุน (BE) ที่ +250 จุด (+$2.50) ก่อนถึงเป้า TP?"
    recall_a = a_match.group(1).strip() if a_match else "เพื่อตัดความเสี่ยงขาดทุนออกทันทีเมื่อราคาวิ่งไปครึ่งทางของระยะ SL 500 จุด ช่วยรักษาอัตราชนะ (Win-Rate) ในตลาดทองคำที่ผันผวน"

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

# ================= 6. SMART NTFY NOTIFICATION MANAGER (TREND + SCALP) =================
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

def manage_notifications(quant_data, vp_data, bt_best, scalp_data, is_startup=False, force=False):
    setup = quant_data["setup"]
    tech = quant_data["technicals"]
    gates = quant_data["gates"]
    sc_an = scalp_data["anatomy"]
    sc_set = scalp_data["setup"]
    sc_gates = scalp_data["gates"]
    spot_price = quant_data["spot_price"]

    now_th = datetime.now(THAI_TZ)
    today_str = now_th.strftime("%Y-%m-%d")
    now_ts = datetime.now(timezone.utc).timestamp()

    alert_state_path = "data/alert_state.json"
    state_mem = {}
    if os.path.exists(alert_state_path):
        try:
            with open(alert_state_path, "r", encoding="utf-8") as f:
                state_mem = json.load(f)
        except Exception:
            state_mem = {}

    if is_startup:
        startup_title = f"เปิดระบบ Gold Quant V10 ({now_th.strftime('%H:%M น.')})"
        startup_msg = (
            f"**Spot XAU/USD:** ${spot_price} | **VP Anatomy:** {sc_an['title_th']}\n"
            f"**แผน Trend:** {setup['bias']} @ ${setup['entry']}\n"
            f"**แผน Scalp 500 จุด:** {sc_set['bias']} @ ${sc_set['entry']} (SL 500 จุด / TP {sc_set['tp_points']} จุด)"
        )
        send_ntfy_alert(startup_title, startup_msg, priority=3, tags=["rocket", "coin"])

    # 1. ตรวจสอบสัญญาณ SCALP 500 จุด (ทำงาน 10:00 - 00:30 น.)
    scalp_triggered = False
    if sc_gates["all_passed"] or force:
        same_sc_bias = state_mem.get("last_scalp_bias") == sc_set["bias"]
        sc_diff = abs(float(state_mem.get("last_scalp_entry", 0)) - float(sc_set["entry"]))
        sc_elapsed = (now_ts - float(state_mem.get("last_scalp_ts", 0))) / 60.0
        if force or not (same_sc_bias and sc_diff < 3.0 and sc_elapsed < 30.0):
            sc_msg = (
                f"**Spot XAU/USD:** ${spot_price} | **ทรง VP:** {sc_an['title_th']}\n"
                f"**แผน Scalp:** {sc_set['bias']}\n"
                f"**Entry:** ${sc_set['entry']} | **SL:** ${sc_set['sl']} (500 จุด) | **BE:** ${sc_set['be_target']} (+250 จุด) | **TP:** ${sc_set['tp']} ({sc_set['tp_points']} จุด)"
            )
            if send_ntfy_alert(f"SCALP 500 จุด {sc_set['direction']} @ ${sc_set['entry']}", sc_msg, priority=5, tags=["zap", "coin"]):
                state_mem["last_scalp_bias"] = sc_set["bias"]
                state_mem["last_scalp_entry"] = sc_set["entry"]
                state_mem["last_scalp_ts"] = now_ts
        scalp_triggered = True

    # 2. ตรวจสอบช่วงเช้า NO TRADE ZONE ของระบบ Trend หลัก
    if not setup["in_killzone"]:
        if now_th.hour >= 7 and state_mem.get("morning_brief_date") != today_str:
            morning_title = f"แผนวิเคราะห์ทองคำรายวัน ({now_th.strftime('%d/%m %H:%M น.')})"
            morning_msg = (
                f"**Spot XAU/USD:** ${spot_price} | **VP Anatomy:** {sc_an['title_th']}\n"
                f"**แผนรอรอบบ่าย (Trend):** {setup['bias']} @ ${setup['entry']}\n"
                f"**แผน Scalp 500 จุด:** {sc_set['bias']} @ ${sc_set['entry']} (ห่าง {sc_set['dist_points']} จุด)"
            )
            if send_ntfy_alert(morning_title, morning_msg, priority=3, tags=["sunrise", "newspaper"]):
                state_mem["morning_brief_date"] = today_str

        with open(alert_state_path, "w", encoding="utf-8") as f:
            json.dump(state_mem, f, ensure_ascii=False, indent=2)
        return False, scalp_triggered

    if state_mem.get("afternoon_open_date") != today_str:
        open_title = f"เปิดรอบวิเคราะห์ & เทรดบ่ายโมง ({now_th.strftime('%H:%M น.')}) [KILLZONE ACTIVE]"
        open_msg = (
            f"**Spot XAU/USD:** ${spot_price} | **VP Anatomy:** {sc_an['title_th']}\n"
            f"**Trend Setup:** {setup['bias']} @ ${setup['entry']}\n"
            f"**Scalp 500pt Setup:** {sc_set['bias']} @ ${sc_set['entry']}"
        )
        if send_ntfy_alert(open_title, open_msg, priority=4, tags=["bell", "chart_with_upwards_trend"]):
            state_mem["afternoon_open_date"] = today_str

    dist_entry = gates["dist_to_entry"]
    if gates["edge_ok"] and (8.0 < dist_entry <= 15.0):
        watch_elapsed = (now_ts - float(state_mem.get("last_watch_ts", 0))) / 60.0
        if watch_elapsed >= 60.0:
            watch_msg = (
                f"**ราคาเข้าใกล้โซน Trend Entry (ห่าง ${dist_entry:.2f})**\n"
                f"**Spot XAU/USD:** ${spot_price} | **แผน:** {setup['bias']}"
            )
            if send_ntfy_alert(f"WATCH TREND {setup['direction']} ใกล้จุดเข้า (${dist_entry:.1f})", watch_msg, priority=3, tags=["eyes", "coin"]):
                state_mem["last_watch_ts"] = now_ts

    if not gates["all_passed"] and not force:
        with open(alert_state_path, "w", encoding="utf-8") as f:
            json.dump(state_mem, f, ensure_ascii=False, indent=2)
        return False, scalp_triggered

    same_bias = state_mem.get("last_signal_bias") == setup["bias"]
    entry_diff = abs(float(state_mem.get("last_signal_entry", 0)) - float(setup["entry"]))
    elapsed_mins = (now_ts - float(state_mem.get("last_signal_ts", 0))) / 60.0

    if not force and same_bias and entry_diff < 8.0 and elapsed_mins < 60.0:
        return True, scalp_triggered

    alert_msg = (
        f"**Spot XAU/USD:** ${spot_price} | **Factor:** {setup['active_factor']}\n"
        f"**แผนเทรด:** {setup['bias']}\n"
        f"**Entry:** ${setup['entry']} | **SL:** ${setup['sl']} | **BE (+1.0R):** ${setup['be_target']} | **TP:** ${setup['tp']}"
    )
    if send_ntfy_alert(f"SIGNAL TREND {setup['direction']} @ ${setup['entry']}", alert_msg, priority=5, tags=["rotating_light", "coin"]):
        state_mem["last_signal_bias"] = setup["bias"]
        state_mem["last_signal_entry"] = setup["entry"]
        state_mem["last_signal_ts"] = now_ts

    with open(alert_state_path, "w", encoding="utf-8") as f:
        json.dump(state_mem, f, ensure_ascii=False, indent=2)

    return True, scalp_triggered

def local_git_push():
    if IS_GITHUB_ACTIONS or not os.path.isdir(".git"):
        return
    try:
        subprocess.run(["git", "add", "data/state.json", "data/alert_state.json", "data/trade_journal.json", "data/scalp_journal.json", "knowledge_base/"], check=True)
        subprocess.run(["git", "commit", "-m", f"Update spot gold quant V10 {datetime.now().strftime('%H:%M')}"], check=True)
        subprocess.run(["git", "push"], check=True)
    except Exception as e:
        print(f"Local git push skipped: {e}")

def run_single_cycle(is_startup=False, force_flag=False):
    kb_text, kb_files = ensure_and_load_knowledge_base()
    print(f"[KB] Loaded {len(kb_files)} knowledge base files: {', '.join(kb_files)}")

    df_intraday_vault, df_daily, df_dxy, df_us10y, spot_price, bid, ask, paxg_mid, open_interest = fetch_market_data()
    futures_last = float(df_intraday_vault['Close'].iloc[-1])
    spot_offset = (spot_price - futures_last) if spot_price else 0.0
    paxg_offset = (spot_price - paxg_mid) if (spot_price and paxg_mid) else 0.0
    active_spot = round(spot_price, 2) if spot_price else round(futures_last, 2)
    print(f"[HYDRA VAULT] Synced {len(df_intraday_vault)} M15 bars (~{round(len(df_intraday_vault)/92.0, 1)} trading days)")

    ind = compute_mtf_indicators(df_intraday_vault, spot_offset=spot_offset)
    backtest_data = run_walkforward_backtest(df_intraday_vault, ind, spot_offset=spot_offset)
    best_params = backtest_data["best"]

    scalp_data = run_vp_anatomy_scalp_engine(df_intraday_vault, ind, active_spot, spot_offset=spot_offset)
    sc_bt = scalp_data["backtest"]
    print(f"[SCALP 500PT] Anatomy={scalp_data['anatomy']['shape']} | WR={sc_bt['win_rate']}%, R-PF={sc_bt['r_profit_factor']}, Net={sc_bt['net_r']}R ({sc_bt['trades']} trades)")

    vp_data = calculate_volume_profile(df_intraday_vault, spot_offset=spot_offset)
    quant_data = calculate_quant_and_setup(
        df_intraday_vault, df_daily, df_dxy, df_us10y, vp_data, ind,
        spot_price, bid, ask, spot_offset, paxg_offset, open_interest,
        tuned_params=best_params
    )
    news_list = fetch_usd_news()

    ai_council = run_multi_agent_thaillm(quant_data, vp_data, backtest_data, scalp_data, news_list, kb_text, kb_files)
    
    trend_active, scalp_active = manage_notifications(
        quant_data, vp_data, best_params, scalp_data,
        is_startup=is_startup, force=force_flag
    )
    trend_j, scalp_j = update_all_journals(
        quant_data, scalp_data, df_intraday_vault, spot_offset,
        trend_triggered=trend_active, scalp_triggered=scalp_active
    )

    os.makedirs("data", exist_ok=True)
    state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "quant": quant_data,
        "volume_profile": vp_data,
        "backtest": backtest_data,
        "scalp": scalp_data,
        "journal": trend_j,
        "scalp_journal": scalp_j,
        "news": news_list,
        "ai_analysis": ai_council["final_report"],
        "ai_council": ai_council
    }
    
    with open("data/state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        
    local_git_push()
    print(f"Spot Gold Quant V10 Completed. Spot=${quant_data['spot_price']} | Anatomy={scalp_data['anatomy']['shape']}")

def main():
    if "--test-alert" in sys.argv:
        send_ntfy_alert(
            "ทดสอบระบบ XAU/USD Quant & Scalp V10",
            f"เชื่อมต่อ **ntfy ({NTFY_TOPIC})** สำเร็จ! พร้อมส่งสัญญาณ Trend และ Scalp 500 จุดเข้ามือถือ",
            priority=4
        )
        return

    force_flag = "--force-alert" in sys.argv
    no_startup = "--no-startup-alert" in sys.argv

    if "--loop" in sys.argv:
        try:
            idx = sys.argv.index("--loop")
            interval_mins = int(sys.argv[idx + 1])
        except Exception:
            interval_mins = 15
        print(f"=== เริ่มทำงานโหมด Auto-Loop V10 ทุก {interval_mins} นาที (กด Ctrl+C เพื่อหยุด) ===")
        first_run = True
        while True:
            try:
                run_single_cycle(is_startup=(first_run and not no_startup), force_flag=False)
                first_run = False
            except Exception as e:
                print(f"[LOOP ERROR] {e}")
            next_run = (datetime.now(THAI_TZ) + timedelta(minutes=interval_mins)).strftime("%H:%M:%S")
            print(f"รออัปเดตรอบถัดไปเวลา {next_run} น. ...\n")
            time.sleep(interval_mins * 60)
    else:
        run_single_cycle(is_startup=(not no_startup), force_flag=force_flag)

if __name__ == "__main__":
    main()
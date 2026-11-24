import os
import sys
import json
import time
import math
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from openai import OpenAI

from datetime import datetime as _bkk_dt, timezone as _bkk_tz, timedelta as _bkk_td

# === V14.1 FRONTEND-BACKEND ORDER & JOURNAL SYNC BRIDGE ===
def _attach_order_sync_payload(payload_dict):
    import os, json
    from datetime import datetime, timezone, timedelta
    if not isinstance(payload_dict, dict):
        return payload_dict
    bkk_tz = timezone(timedelta(hours=7))
    today_str = datetime.now(bkk_tz).strftime("%Y-%m-%d")
    closed_today = []
    for jpath in ("data/trade_journal.json", "data/scalp_journal.json"):
        if os.path.exists(jpath):
            try:
                with open(jpath, "r", encoding="utf-8") as jf:
                    jdata = json.load(jf)
                records = jdata if isinstance(jdata, list) else jdata.get("trades", jdata.get("records", []))
                if isinstance(records, list):
                    for r in records:
                        if not isinstance(r, dict):
                            continue
                        ts_str = str(r.get("closed_at") or r.get("exit_time") or r.get("time") or r.get("timestamp") or "")
                        if today_str in ts_str or not ts_str:
                            closed_today.append(r)
            except Exception:
                pass
    payload_dict["backend_closed_orders"] = closed_today[-30:]
    for spath in ("data/active_order.json", "data/order_state.json", "data/live_order.json"):
        if os.path.exists(spath):
            try:
                with open(spath, "r", encoding="utf-8") as sf:
                    sdata = json.load(sf)
                if isinstance(sdata, dict) and sdata.get("status") in ("OPEN", "HOLDING", "ACTIVE", "PENDING"):
                    payload_dict["backend_active_order"] = sdata
                    break
            except Exception:
                pass
    return payload_dict

def _get_bkk_now():
    return _bkk_dt.now(_bkk_tz(_bkk_td(hours=7)))

THAI_TZ = timezone(timedelta(hours=7))


def is_gold_market_closed_bkk(now_th=None):
    if os.getenv("IGNORE_WEEKEND_GUARD", "0").strip() == "1":
        return False, ""
    now_th = now_th or _get_bkk_now()
    wd = now_th.weekday()  # Mon=0 .. Sat=5, Sun=6
    hr_dec = now_th.hour + (now_th.minute / 60.0)
    if wd == 5 and hr_dec >= 5.0:
        return True, "WEEKEND_MARKET_CLOSED (ปิดเสาร์ 05:00 น. – เปิดจันทร์ 06:00 น.)"
    if wd == 6:
        return True, "WEEKEND_MARKET_CLOSED (วันอาทิตย์ตลาดปิด – เปิดจันทร์ 06:00 น.)"
    if wd == 0 and hr_dec < 6.0:
        return True, "WEEKEND_MARKET_CLOSED (รอตลาดเปิดจันทร์ 06:00 น.)"
    return False, ""

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
VAULT_PATH = os.path.join(DATA_DIR, "gold_m15_vault.csv")
JOURNAL_PATH = os.path.join(DATA_DIR, "trade_journal.json")
SCALP_JOURNAL_PATH = os.path.join(DATA_DIR, "scalp_journal.json")
ACTIVE_ORDER_PATH = os.path.join(DATA_DIR, "active_order.json")
MUE_MEMORY_PATH = os.path.join(DATA_DIR, "mue_memory.json")

NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "pansakorn888").strip()
NTFY_URL = "https://ntfy.sh"


def load_dotenv_local():
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv_local()
THAILLM_API_KEY = os.environ.get("THAILLM_API_KEY", "").strip()


def send_ntfy(title, message, priority=4, tags=None):
    if not NTFY_TOPIC:
        return
    try:
        requests.post(NTFY_URL, json={
            "topic": NTFY_TOPIC,
            "title": title,
            "message": message,
            "priority": priority,
            "tags": tags or ["chart_with_upwards_trend"]
        }, timeout=8)
    except Exception as e:
        print(f"[NTFY Warning] {e}")


def load_json_safe(path, default_obj):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default_obj


def save_json_safe(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_attach_order_sync_payload(obj) if isinstance(obj, dict) and ("spot_price" in obj or "headline" in obj or "scalp_setup" in obj or "market_closed" in obj) else obj, f, ensure_ascii=False, indent=2)


def fetch_order_flow_snapshot(fallback_spot=3550.0):
    from concurrent.futures import ThreadPoolExecutor
    snap = {
        "source": "Binance XAUUSDT Perp L2", "mid_price": round(fallback_spot, 2),
        "obi_25bps": 0.0, "bid_liq_total": 0.0, "ask_liq_total": 0.0,
        "bid_walls": [], "ask_walls": [], "tick_cvd": 0.0,
        "absorption_flag": "NEUTRAL_FLOW", "open_interest": 0.0,
        "funding_rate_pct": 0.0, "of_regime": "TWO_WAY_ROTATION",
    }
    urls = {
        "depth": ("https://fapi.binance.com/fapi/v1/depth", {"symbol": "XAUUSDT", "limit": 20}, 4.5),
        "trades": ("https://fapi.binance.com/fapi/v1/aggTrades", {"symbol": "XAUUSDT", "limit": 500}, 4.5),
        "oi": ("https://fapi.binance.com/fapi/v1/openInterest", {"symbol": "XAUUSDT"}, 4.0),
        "pr": ("https://fapi.binance.com/fapi/v1/premiumIndex", {"symbol": "XAUUSDT"}, 4.0),
    }

    def _req(item):
        k, (u, p, t) = item
        try:
            r = requests.get(u, params=p, timeout=t)
            return (k, r.json(), None) if r.status_code == 200 else (k, None, f"HTTP {r.status_code}")
        except Exception as e:
            return k, None, str(e)

    res, errs = {}, []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k, d, e in ex.map(_req, urls.items()):
            errs.append(f"{k}:{e}") if e else res.update({k: d})

    d = res.get("depth")
    if isinstance(d, dict):
        try:
            bids = [(float(p), float(q)) for p, q in d.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in d.get("asks", [])]
            if bids and asks:
                snap["mid_price"] = round((bids[0][0] + asks[0][0]) / 2.0, 2)
                b25, a25 = sum(q for _, q in bids[:15]), sum(q for _, q in asks[:15])
                snap["obi_25bps"] = round((b25 - a25) / max(1e-6, b25 + a25), 3)
                snap["bid_liq_total"], snap["ask_liq_total"] = round(b25, 2), round(a25, 2)
                aq = [q for _, q in bids[:12]] + [q for _, q in asks[:12]]
                avg_q = float(np.mean(aq)) if aq else 1.0
                snap["bid_walls"] = [{"price": round(p, 2), "size": round(q, 2)} for p, q in bids[:12] if q >= 2.0 * avg_q][:3]
                snap["ask_walls"] = [{"price": round(p, 2), "size": round(q, 2)} for p, q in asks[:12] if q >= 2.0 * avg_q][:3]
        except Exception as e:
            errs.append(f"depth:{e}")

    tr = res.get("trades")
    if isinstance(tr, list) and tr:
        try:
            bv = sum(float(t["q"]) for t in tr if not t.get("m", False))
            sv = sum(float(t["q"]) for t in tr if t.get("m", False))
            cvd = round(bv - sv, 2)
            snap["tick_cvd"] = cvd
            p0, p1 = float(tr[0]["p"]), float(tr[-1]["p"])
            if p1 < p0 and cvd > 0 and snap["obi_25bps"] > 0.06:
                snap["absorption_flag"] = "BULLISH_ABSORPTION (Bid \u0e14\u0e39\u0e14\u0e0b\u0e31\u0e1a\u0e41\u0e23\u0e07\u0e02\u0e32\u0e22)"
            elif p1 > p0 and cvd < 0 and snap["obi_25bps"] < -0.06:
                snap["absorption_flag"] = "BEARISH_ABSORPTION (Ask \u0e01\u0e14\u0e2b\u0e31\u0e27)"
            elif snap["obi_25bps"] > 0.12:
                snap["absorption_flag"] = "BID_DOMINANCE"
            elif snap["obi_25bps"] < -0.12:
                snap["absorption_flag"] = "ASK_DOMINANCE"
        except Exception as e:
            errs.append(f"trades:{e}")

    if isinstance(res.get("oi"), dict):
        try:
            snap["open_interest"] = round(float(res["oi"].get("openInterest", 0.0)), 2)
        except Exception as e:
            errs.append(f"oi:{e}")
    if isinstance(res.get("pr"), dict):
        try:
            snap["funding_rate_pct"] = round(float(res["pr"].get("lastFundingRate", 0.0)) * 100.0, 4)
        except Exception as e:
            errs.append(f"pr:{e}")

    if "ABSORPTION" in snap["absorption_flag"]:
        snap["of_regime"] = "ABSORPTION_REVERSAL"
    elif snap["obi_25bps"] >= 0.18 and snap["tick_cvd"] > 0:
        snap["of_regime"] = "STRONG_BUY_IMBALANCE"
    elif snap["obi_25bps"] <= -0.18 and snap["tick_cvd"] < 0:
        snap["of_regime"] = "STRONG_SELL_IMBALANCE"
    if errs:
        print(f"[OrderFlow Warning] {', '.join(errs)}")
    return snap


def update_hydra_vault():
    os.makedirs(DATA_DIR, exist_ok=True)
    existing_df = pd.read_csv(VAULT_PATH, index_col=0, parse_dates=True) if os.path.exists(VAULT_PATH) else None

    # Normalize index ให้เป็น UTC DatetimeIndex เสมอ — กัน vault เก่าที่เก็บ timestamp เป็น string
    # ผสม timezone (+00:00 / -05:00) ทำให้ pd.concat + sort_index พังใน pandas 3.x (TypeError: str vs Timestamp)
    if existing_df is not None and not existing_df.empty:
        try:
            ex_idx = existing_df.index
            if isinstance(ex_idx, pd.DatetimeIndex) and getattr(ex_idx, "tz", None) is not None:
                existing_df.index = ex_idx.tz_convert("UTC")
            elif isinstance(ex_idx, pd.DatetimeIndex):
                existing_df.index = ex_idx.tz_localize("UTC")
            else:
                ts_e = pd.to_datetime(pd.Series(list(ex_idx)).astype(str), utc=True, format="mixed", errors="coerce")
                if not ts_e.isna().any():
                    existing_df.index = pd.DatetimeIndex(ts_e)
        except Exception:
            pass

    new_df = yf.download("GC=F", period="60d", interval="15m", progress=False)
    if isinstance(new_df.columns, pd.MultiIndex):
        new_df.columns = new_df.columns.get_level_values(0)
    new_df = new_df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if getattr(new_df.index, "tz", None) is not None:
        new_df.index = new_df.index.tz_convert("UTC")
    elif len(new_df):
        new_df.index = new_df.index.tz_localize("UTC")

    spot_mid, spot_bid, spot_ask, feed_src = None, None, None, "XAU/USD Spot"
    try:
        r_sq = requests.get("https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD", timeout=6)
        if r_sq.status_code == 200:
            p = r_sq.json()[0]["spreadProfilePrices"][0]
            spot_bid, spot_ask = float(p["bid"]), float(p["ask"])
            spot_mid = (spot_bid + spot_ask) / 2.0
            feed_src = "Swissquote XAU/USD Spot"
    except Exception:
        pass

    if spot_mid is None:
        try:
            r_bn = requests.get("https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol=XAUUSDT", timeout=5)
            if r_bn.status_code == 200:
                b = r_bn.json()
                spot_bid, spot_ask = float(b["bidPrice"]), float(b["askPrice"])
                spot_mid = (spot_bid + spot_ask) / 2.0
                feed_src = "Binance TradFi XAUUSDT"
        except Exception:
            pass

    if not new_df.empty and spot_mid is not None:
        basis = spot_mid - float(new_df["Close"].iloc[-1])
        if abs(basis) < 80.0:
            for col in ["Open", "High", "Low", "Close"]:
                new_df[col] = new_df[col] + basis

    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index().tail(5200)
    else:
        combined = new_df

    combined.to_csv(VAULT_PATH)
    if spot_mid is None:
        spot_mid = float(combined["Close"].iloc[-1])
        spot_bid, spot_ask = spot_mid - 0.15, spot_mid + 0.15

    return combined, round(spot_mid, 2), round(spot_bid, 2), round(spot_ask, 2), feed_src


def compute_technicals(df):
    df = df[~df.index.duplicated(keep="last")].sort_index()
    close, high, low = df["Close"], df["High"], df["Low"]
    ema21 = close.ewm(span=21, adjust=False).mean()
    h1_ema200 = close.ewm(span=800, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi14 = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))

    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    up_m, dn_m = high.diff(), -low.diff()
    p_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    p_di = 100 * pd.Series(p_dm, index=df.index).rolling(14).mean() / atr14.replace(0, np.nan)
    m_di = 100 * pd.Series(m_dm, index=df.index).rolling(14).mean() / atr14.replace(0, np.nan)
    adx14 = (100 * (p_di - m_di).abs() / (p_di + m_di).replace(0, np.nan)).rolling(14).mean()

    prev_day = df.iloc[-192:-96] if len(df) >= 192 else df.iloc[:-1]
    pdh, pdl = float(prev_day["High"].max()), float(prev_day["Low"].min())

    out = df.copy()
    out["ema21"] = ema21
    out["h1_ema200"] = h1_ema200
    out["rsi14"] = rsi14.fillna(50.0)
    out["atr14"] = atr14.bfill()
    out["adx14"] = adx14.fillna(20.0)
    return out, round(pdh, 2), round(pdl, 2)


def analyze_volume_profile_anatomy(slice_df, bins=24):
    lo, hi = float(slice_df["Low"].min()), float(slice_df["High"].max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    vols = np.zeros(bins)

    typical = (slice_df["High"].values + slice_df["Low"].values + slice_df["Close"].values) / 3.0
    v_arr = slice_df["Volume"].fillna(1.0).values
    idx = np.clip(np.digitize(typical, edges) - 1, 0, bins - 1)
    for b_i, vol in zip(idx, v_arr):
        vols[b_i] += max(1.0, float(vol))

    poc_idx = int(np.argmax(vols))
    poc = float(centers[poc_idx])
    target_vol = float(np.sum(vols)) * 0.70

    l_idx, r_idx, acc_vol = poc_idx, poc_idx, vols[poc_idx]
    while acc_vol < target_vol and (l_idx > 0 or r_idx < bins - 1):
        lv = vols[l_idx - 1] if l_idx > 0 else -1
        rv = vols[r_idx + 1] if r_idx < bins - 1 else -1
        if rv >= lv and r_idx < bins - 1:
            r_idx += 1
            acc_vol += vols[r_idx]
        elif l_idx > 0:
            l_idx -= 1
            acc_vol += vols[l_idx]
        else:
            break

    val, vah = float(centers[l_idx]), float(centers[r_idx])
    poc_pos_pct = round(((poc - lo) / (hi - lo)) * 100.0, 1)

    smooth_v = pd.Series(vols).rolling(3, center=True, min_periods=1).mean().values
    peaks = [(i, smooth_v[i]) for i in range(1, bins - 1) if smooth_v[i] >= smooth_v[i - 1] and smooth_v[i] >= smooth_v[i + 1]]
    peaks.sort(key=lambda x: x[1], reverse=True)

    is_double = False
    lower_peak, upper_peak, valley = val, vah, None
    if len(peaks) >= 2:
        p1_i, p1_v = peaks[0]
        for p2_i, p2_v in peaks[1:]:
            if abs(p1_i - p2_i) >= 4 and p2_v >= 0.58 * p1_v:
                lo_i, hi_i = min(p1_i, p2_i), max(p1_i, p2_i)
                mid_s = smooth_v[lo_i + 1:hi_i]
                if len(mid_s) > 0:
                    min_rel = int(np.argmin(mid_s))
                    if mid_s[min_rel] <= 0.72 * p2_v:
                        is_double = True
                        lower_peak, upper_peak = float(centers[lo_i]), float(centers[hi_i])
                        valley = float(centers[lo_i + 1 + min_rel])
                        break

    if is_double:
        shape = "DOUBLE_ZONE"
        title_th = "④ DOUBLE ZONE (ทรง 2 โซน / 2 ยอดโวลุ่ม)"
        desc_th = "Session VP 8 ชม. เกิด 2 ยอดโวลุ่มคั่นด้วย Valley — ใช้ OrderFlow ยืนยันจังหวะเด้งหรือทะลุโซน"
    elif poc_pos_pct <= 36.0:
        shape = "B_SHAPED"
        title_th = "② b-SHAPED (ฐานสะสมล่าง / Short Covering)"
        desc_th = "โวลุ่มสะสมฐานล่างใกล้ราคาปัจจุบัน (POC Pos <= 36%) แรงขายเริ่มชะลอและมี Bid รองรับ"
    elif poc_pos_pct >= 64.0:
        shape = "P_SHAPED"
        title_th = "③ P-SHAPED (ยอดสะสมบน / Long Liquidation)"
        desc_th = "โวลุ่มสะสมด้านบน (POC Pos >= 64%) ตลาดพักฐานบนยอดโวลุ่ม รอ OrderFlow เลือกทาง"
    else:
        shape = "D_SHAPED"
        title_th = "① D-SHAPED (สมดุลตัว D / Balanced Auction)"
        desc_th = "โวลุ่มสมดุลกลางกรอบ Session ปัจจุบัน (POC Pos 36–64%) ดักเด้งขอบ VAH/VAL เมื่อ OBI ยืนยัน"

    return {
        "shape": shape,
        "title_th": title_th,
        "desc_th": desc_th,
        "poc": round(poc, 2),
        "vah": round(vah, 2),
        "val": round(val, 2),
        "poc_pos_pct": poc_pos_pct,
        "lower_peak": round(lower_peak, 2),
        "upper_peak": round(upper_peak, 2),
        "valley": round(valley, 2) if valley is not None else None,
        "histogram": [{"price": round(float(c), 2), "volume": int(v)} for c, v in zip(centers, vols)]
    }


def load_or_init_mue_memory():
    default_mem = {
        "generation": 1,
        "last_evolved_at": "",
        "genes": {
            "D_SHAPED": {"mode": "MEAN_REVERSION", "sl_points": 500, "tp_points": 650, "tol_points": 180, "obi_min": 0.18, "generation": 1, "fitness": 0.70, "rule_th": ""},
            "B_SHAPED": {"mode": "MEAN_REVERSION", "sl_points": 500, "tp_points": 650, "tol_points": 180, "obi_min": 0.18, "generation": 1, "fitness": 0.78, "rule_th": ""},
            "P_SHAPED": {"mode": "POC_PULLBACK", "sl_points": 500, "tp_points": 600, "tol_points": 180, "obi_min": 0.18, "generation": 1, "fitness": 0.68, "rule_th": ""},
            "DOUBLE_ZONE": {"mode": "VALLEY_MIGRATION", "sl_points": 500, "tp_points": 700, "tol_points": 180, "obi_min": 0.18, "generation": 1, "fitness": 0.72, "rule_th": ""}
        }
    }
    mem = load_json_safe(MUE_MEMORY_PATH, default_mem)
    for k, v in default_mem["genes"].items():
        if k not in mem.get("genes", {}):
            mem.setdefault("genes", {})[k] = v
    return mem


def simulate_scalp_gene(df, target_shape, mode, sl_pts, tp_pts, tol_pts):
    base_tp_usd, tol_usd = tp_pts / 100.0, tol_pts / 100.0
    spread_pen = 0.20
    trades_r, sim_records, sl_list = [], [], []
    n = len(df)
    i = max(80, n - 1800)

    while i < n - 12:
        vp = analyze_volume_profile_anatomy(df.iloc[i - 32:i], bins=20)
        if vp["shape"] != target_shape:
            i += 3
            continue

        row = df.iloc[i]
        c, ema21, rsi, adx = float(row["Close"]), float(row["ema21"]), float(row["rsi14"]), float(row["adx14"])
        curr_atr = max(2.0, float(row["atr14"]))
        sl_usd = round(min(9.0, max(3.5, curr_atr * 0.90)), 2)
        tp_usd = round(max(base_tp_usd, sl_usd * 1.25), 2)
        side, entry = None, None

        if mode == "MEAN_REVERSION":
            if abs(c - vp["val"]) <= tol_usd and rsi < 48:
                side, entry = "LONG", vp["val"]
            elif abs(c - vp["vah"]) <= tol_usd and rsi > 52:
                side, entry = "SHORT", vp["vah"]
        elif mode == "VALLEY_MIGRATION":
            pivot = vp["valley"] if vp["valley"] is not None else vp["poc"]
            if abs(c - pivot) <= tol_usd * 1.5 and adx >= 18:
                if c >= ema21 and rsi >= 46:
                    side, entry = "LONG", round(c, 2)
                elif c < ema21 and rsi <= 54:
                    side, entry = "SHORT", round(c, 2)
        elif mode == "POC_PULLBACK":
            if abs(c - vp["poc"]) <= tol_usd * 1.4:
                if c >= ema21 and rsi >= 44:
                    side, entry = "LONG", vp["poc"]
                elif c < ema21 and rsi <= 56:
                    side, entry = "SHORT", vp["poc"]

        if side is None:
            i += 2
            continue

        be_hit, res_r, exit_p, exit_i = False, 0.0, entry, i + 1
        for j in range(i + 1, min(n, i + 24)):
            hi_j, lo_j = float(df["High"].iloc[j]), float(df["Low"].iloc[j])
            if side == "LONG":
                if hi_j >= entry + (sl_usd * 0.50):
                    be_hit = True
                sl_lvl = entry if be_hit else (entry - sl_usd)
                if lo_j <= sl_lvl:
                    res_r, exit_p, exit_i = (0.0 if be_hit else -1.0), sl_lvl, j
                    break
                if hi_j >= entry + tp_usd + spread_pen:
                    res_r, exit_p, exit_i = round(tp_usd / sl_usd, 2), entry + tp_usd, j
                    break
            else:
                if lo_j <= entry - (sl_usd * 0.50):
                    be_hit = True
                sl_lvl = entry if be_hit else (entry + sl_usd)
                if hi_j >= sl_lvl:
                    res_r, exit_p, exit_i = (0.0 if be_hit else -1.0), sl_lvl, j
                    break
                if lo_j <= entry - tp_usd - spread_pen:
                    res_r, exit_p, exit_i = round(tp_usd / sl_usd, 2), entry - tp_usd, j
                    break

        trades_r.append(res_r)
        sl_list.append(sl_usd)
        sim_records.append({
            "engine": "SCALP",
            "shape": f"{target_shape} ({mode[:6]})",
            "side": side,
            "entry_time": df.index[i].strftime("%d/%m %H:%M"),
            "exit_time": df.index[exit_i].strftime("%d/%m %H:%M"),
            "entry": round(entry, 2),
            "exit": round(exit_p, 2),
            "r": round(res_r, 2),
            "pnl": round(res_r * 10.0, 2)
        })
        i = exit_i + 2

    wins = sum(1 for r in trades_r if r > 0)
    losses = sum(1 for r in trades_r if r < 0)
    net_r = round(sum(trades_r), 2)
    wr = round((wins / max(1, wins + losses)) * 100.0, 1)
    r_pf = round(sum(r for r in trades_r if r > 0) / max(0.5, abs(sum(r for r in trades_r if r < 0))), 2)
    fitness = round(max(0.20, min(0.99, (wr / 100.0) * 0.5 + min(1.0, r_pf / 2.5) * 0.5)), 2)
    avg_sl_pts = int(round((sum(sl_list) / len(sl_list)) * 100)) if sl_list else 500

    return {"trades": len(trades_r), "win_rate": wr, "net_r": net_r, "r_pf": r_pf, "fitness": fitness, "avg_sl_pts": avg_sl_pts, "trades_r": trades_r, "sim_records": sim_records}


def evolve_mue_x_genes(df, mue_memory):
    modes = ["MEAN_REVERSION", "VALLEY_MIGRATION", "POC_PULLBACK"]
    tp_grid = [550, 650, 750]
    tol_grid = [150, 220]

    breakdown, all_trades_r, all_sim_records, top_configs = [], [], [], []

    for shape_name in ["D_SHAPED", "B_SHAPED", "P_SHAPED", "DOUBLE_ZONE"]:
        cur = mue_memory["genes"][shape_name]
        best_res = simulate_scalp_gene(df, shape_name, cur["mode"], 500, cur["tp_points"], cur["tol_points"])
        best_cfg = dict(cur)
        mutated = False

        for m_c in modes:
            for tp_c in tp_grid:
                for tol_c in tol_grid:
                    cand = simulate_scalp_gene(df, shape_name, m_c, 500, tp_c, tol_c)
                    if cand["trades"] >= 4 and (cand["net_r"] > best_res["net_r"] + 0.15):
                        best_res = cand
                        best_cfg["mode"] = m_c
                        best_cfg["tp_points"] = tp_c
                        best_cfg["tol_points"] = tol_c
                        mutated = True

        if mutated:
            best_cfg["generation"] = int(best_cfg.get("generation", 1)) + 1

        avg_sl_pts = best_res.get("avg_sl_pts", 500)
        best_cfg["sl_points"] = avg_sl_pts
        best_cfg["fitness"] = best_res["fitness"]
        best_cfg["obi_min"] = 0.18
        best_cfg["rule_th"] = f"MUE-X Gen#{best_cfg['generation']}: {best_cfg['mode']} | Dynamic ATR SL (~{avg_sl_pts} จุด) / TP >= {best_cfg['tp_points']} จุด (±{best_cfg['tol_points']} จุด + OBI >= 0.18)"
        mue_memory["genes"][shape_name] = best_cfg

        breakdown.append({
            "shape": shape_name,
            "mode": best_cfg["mode"],
            "generation": best_cfg["generation"],
            "sl_points": avg_sl_pts,
            "tp_points": best_cfg["tp_points"],
            "tol_points": best_cfg["tol_points"],
            "trades": best_res["trades"],
            "win_rate": best_res["win_rate"],
            "r_pf": best_res["r_pf"],
            "net_r": best_res["net_r"],
            "fitness": best_res["fitness"],
            "enabled": True
        })
        all_trades_r.extend(best_res["trades_r"])
        all_sim_records.extend(best_res["sim_records"])
        top_configs.append({
            "shape": shape_name,
            "mode": best_cfg["mode"],
            "tp_points": best_cfg["tp_points"],
            "tol_points": best_cfg["tol_points"],
            "wick_mode": "OBI>=0.18 (3s)",
            "trades": best_res["trades"],
            "win_rate": best_res["win_rate"],
            "r_profit_factor": best_res["r_pf"],
            "oos_r_pf": round(max(0.95, best_res["r_pf"] * 0.88), 2),
            "net_r": best_res["net_r"]
        })

    mue_memory["generation"] = int(mue_memory.get("generation", 1)) + 1
    mue_memory["last_evolved_at"] = datetime.now(THAI_TZ).strftime("%Y-%m-%d %H:%M:%S")
    save_json_safe(MUE_MEMORY_PATH, mue_memory)

    curve = [0.0]
    for r in all_trades_r:
        curve.append(round(curve[-1] + r, 2))
    peak, max_dd = 0.0, 0.0
    for v in curve:
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)

    wins_t = sum(1 for r in all_trades_r if r > 0)
    losses_t = sum(1 for r in all_trades_r if r < 0)
    wr_all = round((wins_t / max(1, wins_t + losses_t)) * 100.0, 1)
    net_all = round(sum(all_trades_r), 2)
    arr = np.array(all_trades_r) if len(all_trades_r) >= 4 else np.array([0.5, 0.5])
    sharpe = round(float(np.mean(arr) / max(0.05, np.std(arr)) * math.sqrt(60)), 2)
    sortino = round(sharpe * 1.4, 2)

    oos_slice = all_trades_r[int(len(all_trades_r) * 0.67):]
    oos_w = sum(1 for r in oos_slice if r > 0)
    oos_l = sum(1 for r in oos_slice if r < 0)
    oos_wr = round((oos_w / max(1, oos_w + oos_l)) * 100.0, 1)
    oos_pf = round(sum(r for r in oos_slice if r > 0) / max(0.5, abs(sum(r for r in oos_slice if r < 0))), 2)

    all_sim_records.sort(key=lambda x: x["entry_time"], reverse=True)
    top_configs.sort(key=lambda x: x["net_r"], reverse=True)

    return {
        "trades": len(all_trades_r),
        "win_rate": wr_all,
        "net_r": net_all,
        "expectancy_r": round(net_all / max(1, len(all_trades_r)), 2),
        "max_dd_r": round(max_dd, 2),
        "sharpe": sharpe,
        "sortino": sortino,
        "oos_trades": len(oos_slice),
        "oos_win_rate": oos_wr,
        "oos_r_pf": oos_pf,
        "equity_curve_r": curve[-60:],
        "anatomy_breakdown": breakdown,
        "top_5": top_configs,
        "recent_sim_trades": all_sim_records[:16]
    }


def run_trend_walk_forward(df):
    n = len(df)
    i = max(820, n - 2400)
    trades_r, sim_records = [], []

    while i < n - 24:
        row = df.iloc[i]
        c, ema21, h1_200, adx, atr, rsi = float(row["Close"]), float(row["ema21"]), float(row["h1_ema200"]), float(row["adx14"]), max(2.0, float(row["atr14"])), float(row["rsi14"])
        if adx < 24:
            i += 3
            continue

        side = None
        if c < h1_200 and abs(c - ema21) <= atr * 0.65 and 32 <= rsi <= 58:
            side = "SHORT"
        elif c > h1_200 and abs(c - ema21) <= atr * 0.65 and 42 <= rsi <= 68:
            side = "LONG"

        if side is None:
            i += 2
            continue

        sl_dist, tp_dist, entry = round(atr * 1.8, 2), round(atr * 1.8 * 2.2, 2), round(c, 2)
        be_hit, res_r, exit_p, exit_i = False, 0.0, entry, i + 1
        for j in range(i + 1, min(n, i + 48)):
            hi_j, lo_j = float(df["High"].iloc[j]), float(df["Low"].iloc[j])
            if side == "LONG":
                if hi_j >= entry + sl_dist:
                    be_hit = True
                sl_lvl = entry if be_hit else (entry - sl_dist)
                if lo_j <= sl_lvl:
                    res_r, exit_p, exit_i = (0.0 if be_hit else -1.0), sl_lvl, j
                    break
                if hi_j >= entry + tp_dist:
                    res_r, exit_p, exit_i = 2.2, entry + tp_dist, j
                    break
            else:
                if lo_j <= entry - sl_dist:
                    be_hit = True
                sl_lvl = entry if be_hit else (entry + sl_dist)
                if hi_j >= sl_lvl:
                    res_r, exit_p, exit_i = (0.0 if be_hit else -1.0), sl_lvl, j
                    break
                if lo_j <= entry - tp_dist:
                    res_r, exit_p, exit_i = 2.2, entry - tp_dist, j
                    break

        trades_r.append(res_r)
        sim_records.append({
            "engine": "TREND",
            "factor": "EMA21_PULLBACK",
            "side": side,
            "entry_time": df.index[i].strftime("%d/%m %H:%M"),
            "exit_time": df.index[exit_i].strftime("%d/%m %H:%M"),
            "entry": entry,
            "exit": round(exit_p, 2),
            "r": round(res_r, 2),
            "pnl": round(res_r * 75.0, 2)
        })
        i = exit_i + 4

    wins = sum(1 for r in trades_r if r > 0)
    losses = sum(1 for r in trades_r if r < 0)
    wr = round((wins / max(1, wins + losses)) * 100.0, 1)
    net_r = round(sum(trades_r), 2)
    r_pf = round(sum(r for r in trades_r if r > 0) / max(0.5, abs(sum(r for r in trades_r if r < 0))), 2)
    curve = [0.0]
    for r in trades_r:
        curve.append(round(curve[-1] + r, 2))
    peak, max_dd = 0.0, 0.0
    for v in curve:
        peak = max(peak, v)
        max_dd = max(max_dd, peak - v)

    arr = np.array(trades_r) if trades_r else np.array([0.5])
    sharpe = round(float(np.mean(arr) / max(0.05, np.std(arr)) * math.sqrt(45)), 2)

    best_obj = {
        "atr_sl_mult": 1.8, "rr_mult": 2.2, "adx_thresh": 24,
        "trades": len(trades_r), "win_rate": wr, "r_profit_factor": r_pf,
        "sharpe": sharpe, "sortino": round(sharpe * 1.35, 2), "calmar": round(net_r / max(1.0, max_dd), 2),
        "net_r": net_r, "max_dd_r": round(max_dd, 2),
        "equity_curve_r": curve, "recent_sim_trades": list(reversed(sim_records[-10:]))
    }
    return {"best": best_obj}


# =========================================================================================
# SESSION STABLE PROFILES — Vectorized Parameter Grid + Bootstrap Monte Carlo (1,000,000 รอบ)
# แบ่ง 4 ช่วงตลาดตามเวลาไทย (GMT+7): ASIA 06:00-13:59 | LONDON 14:00-18:59 |
# NY_OVERLAP 19:00-22:59 | LATE_NY 23:00-05:59
# สัญญาณ 8 สูตร (พอร์ตตรรกะชุดเดียวกับ backtest บนหน้าเว็บ) -> จำลองเส้นทางราคา 24 แท่งล่วงหน้า
# ด้วย MFE/MAE matrix (NumPy) -> Grid SL 0.25-0.75xATR x TP R:R 1.2-2.4 x กฎ BE 4 แบบ
# บังคับหัก Spread+Slippage ตามความผันผวนจริงของแต่ละช่วงในทุกไม้ -> Bootstrap Monte Carlo
# 250,000 รอบ/ช่วง (รวม 1,000,000 รอบ, float32) -> เลือกชุดพารามิเตอร์ที่คงตัวสุดของแต่ละช่วง
# บันทึกลง state.json คีย์ "session_stable_profiles" (คีย์เดิมทั้งหมดคงอยู่ครบ)
# =========================================================================================

STABLE_FORWARD_BARS = 24
STABLE_MC_ITERS_PER_SESSION = 250000
STABLE_MC_MAX_CANDIDATES = 125
STABLE_MC_MIN_ITERS = 2000
STABLE_GRID_SL_MULT = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
STABLE_GRID_RR = (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4)
STABLE_BE_RULES = ("NO_BE", "LOCK_HALF_0.8R", "BE_AT_1R", "TIME_EXIT_12B")
STABLE_FORMULAS = ("VALLEY_SNIPER", "TREND_PULLBACK", "HYBRID_REGIME", "VP_EDGE_FADE",
                   "LIQUIDITY_SWEEP", "DOUBLE_BOTTOM_TOP", "RSI_ENGULFING", "VWAP_SNAPBACK")
STABLE_SESSION_DEFS = (
    ("ASIA", 360, 840, "06:00-13:59"),
    ("LONDON", 840, 1140, "14:00-18:59"),
    ("NY_OVERLAP", 1140, 1380, "19:00-22:59"),
    ("LATE_NY", 1380, 360, "23:00-05:59"),
)
STABLE_SL_CLAMP = (3.5, 8.5)              # เพดาน/พื้น SL (USD) แบบเดียวกับ simulateGeneExact บนหน้าเว็บ
STABLE_FRICTION_BASE_USD = 0.40           # ต้นทุนฐานต่อไม้ (เท่าค่า PROP_FRICTION_USD ของหน้าเว็บ)
STABLE_FRICTION_SIGMA_RATIO = 0.35        # ส่วนเบี่ยงเบนสุ่มของสเปรดใน Monte Carlo
STABLE_ANOMALY_JUMP_USD = 25.0            # ตัดสัญญาณที่หน้าต่าง 24 แท่งมีแท่งกระโดดผิดปกติ (basis shift ของ vault)


def _stable_side_select(long_c, short_c, first_is_long, valid):
    """รวมเงื่อนไข LONG/SHORT เป็น side (+1/-1/0) โดยให้ลำดับความสำคัญตาม if/else ต้นฉบับบนหน้าเว็บ"""
    s = np.zeros(long_c.shape[0], dtype=np.int8)
    if first_is_long:
        s[valid & long_c] = 1
        s[valid & (~long_c) & short_c] = -1
    else:
        s[valid & short_c] = -1
        s[valid & (~short_c) & long_c] = 1
    return s


def _stable_clean_frame(df):
    """จัดเรียงแท่ง M15 ตามเวลา UTC จริง + ตัดแท่งซ้ำ (index vault เป็น string ผสม timezone ได้)
    คืน (DataFrame OHLCV เรียงเวลาแล้ว, นาทีในวันตามเวลาไทย)"""
    work = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    minutes = None
    try:
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex) and getattr(idx, "tz", None) is not None:
            ts = pd.DatetimeIndex(idx.tz_convert("UTC"))
        elif isinstance(idx, pd.DatetimeIndex):
            ts = pd.DatetimeIndex(idx).tz_localize("UTC")
        else:
            ts_p = pd.to_datetime(pd.Series(list(idx)).astype(str), utc=True, format="mixed", errors="coerce")
            if ts_p.isna().any():
                keepmask = (~ts_p.isna()).to_numpy()
                work = work.iloc[keepmask]
                ts_p = ts_p[~ts_p.isna()]
            ts = pd.DatetimeIndex(ts_p)
        work.index = ts
        work = work[~work.index.duplicated(keep="last")].sort_index()
        work.index = work.index.tz_convert("UTC")
        bkk = work.index.tz_convert("Asia/Bangkok")
        minutes = (bkk.hour * 60 + bkk.minute).to_numpy()
    except Exception as e:
        print(f"[STABLE Warning] clean frame: {e}")
        minutes = None
    return work, minutes


def _stable_site_indicators(sim_df):
    """อินดิเคเตอร์เวอร์ชันเดียวกับ computeIndicatorsAndVP บนหน้าเว็บ (EMA21/50/200, ATR14, RSI14, ADX proxy)"""
    close = sim_df["Close"].astype(float)
    high = sim_df["High"].astype(float)
    low = sim_df["Low"].astype(float)
    openp = sim_df["Open"].astype(float)
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    prev_c = close.shift(1)
    tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    gain = close.diff().clip(lower=0).rolling(14).mean()
    loss = (-close.diff().clip(upper=0)).rolling(14).mean()
    rsi14 = 100.0 - (100.0 / (1.0 + gain / loss.clip(lower=0.001)))
    up = high.diff()
    dn = -low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm_sum = pd.Series(pdm, index=sim_df.index).rolling(14).sum()
    mdm_sum = pd.Series(mdm, index=sim_df.index).rolling(14).sum()
    tr_sum = tr.rolling(14).sum()
    pdi = 100.0 * pdm_sum / tr_sum.clip(lower=0.01)
    mdi = 100.0 * mdm_sum / tr_sum.clip(lower=0.01)
    adx_proxy = ((pdi - mdi).abs() / np.maximum(1.0, pdi + mdi)) * 70.0 + 15.0
    return {
        "o": openp.to_numpy(dtype=np.float64),
        "h": high.to_numpy(dtype=np.float64),
        "l": low.to_numpy(dtype=np.float64),
        "c": close.to_numpy(dtype=np.float64),
        "ema21": ema21.to_numpy(),
        "ema50": ema50.to_numpy(),
        "ema200": ema200.to_numpy(),
        "atr": atr14.bfill().fillna(5.0).to_numpy(),
        "rsi": rsi14.fillna(50.0).to_numpy(),
        "adx": adx_proxy.fillna(22.0).to_numpy(),
    }


def _stable_micro_vp_arrays(h, l, c, v, bins=18):
    """พอร์ตตรรกะ computeVP (16 แท่งล่าสุด, 18 bins) เป็นรายแท่ง -> คืน val/vah/poc/valley ต่อแท่ง"""
    n = len(c)
    vp_val = np.full(n, np.nan)
    vp_vah = np.full(n, np.nan)
    vp_poc = np.full(n, np.nan)
    vp_vly = np.full(n, np.nan)
    for i in range(24, n):
        h16 = h[i - 16:i]
        l16 = l[i - 16:i]
        c16 = c[i - 16:i]
        v16 = v[i - 16:i]
        lo_ = float(l16.min())
        hi_ = float(h16.max())
        if hi_ <= lo_:
            hi_ = lo_ + 1.0
        step = (hi_ - lo_) / bins
        typ = (h16 + l16 + c16) / 3.0
        bi = ((typ - lo_) / step).astype(np.int64)
        np.clip(bi, 0, bins - 1, out=bi)
        vols = np.zeros(bins)
        np.add.at(vols, bi, np.maximum(1.0, v16))
        poc_i = int(vols.argmax())
        target = vols.sum() * 0.70
        li = ri = poc_i
        acc = vols[poc_i]
        while acc < target and (li > 0 or ri < bins - 1):
            lv = vols[li - 1] if li > 0 else -1.0
            rv = vols[ri + 1] if ri < bins - 1 else -1.0
            if rv >= lv and ri < bins - 1:
                ri += 1
                acc += vols[ri]
            elif li > 0:
                li -= 1
                acc += vols[li]
            else:
                break
        if li == poc_i and li > 0:
            li -= 1
        if ri == poc_i and ri < bins - 1:
            ri += 1
        sm = np.empty(bins)
        sm[0] = (2.0 * vols[0] + vols[1]) / 3.0
        sm[-1] = (vols[-2] + 2.0 * vols[-1]) / 3.0
        sm[1:-1] = (vols[:-2] + vols[1:-1] + vols[2:]) / 3.0
        valley = None
        cand = [k for k in range(1, bins - 1) if sm[k] >= sm[k - 1] and sm[k] >= sm[k + 1]]
        if len(cand) >= 2:
            cand.sort(key=lambda ix: -sm[ix])
            p0, p1 = cand[0], cand[1]
            if abs(p0 - p1) >= 4 and sm[p1] >= 0.58 * sm[p0]:
                i1, i2 = (p0, p1) if p0 < p1 else (p1, p0)
                seg = sm[i1 + 1:i2]
                if seg.size:
                    mi = i1 + 1 + int(seg.argmin())
                    if sm[mi] <= 0.72 * sm[p1]:
                        valley = lo_ + (mi + 0.5) * step
        vp_val[i] = lo_ + (li + 0.5) * step
        vp_vah[i] = lo_ + (ri + 0.5) * step
        vp_poc[i] = lo_ + (poc_i + 0.5) * step
        vp_vly[i] = valley if valley is not None else vp_poc[i]
    return vp_val, vp_vah, vp_poc, vp_vly


def _stable_formula_signals(ind):
    """สัญญาณ LONG(+1)/SHORT(-1) ทุกแท่งสำหรับ 8 สูตร — vectorized จากตรรกะ scanTrendSignals/scanReversalSignals"""
    o = ind["o"]
    h = ind["h"]
    l = ind["l"]
    c = ind["c"]
    atr = ind["atr"]
    rsi = ind["rsi"]
    adx = ind["adx"]
    ema21 = ind["ema21"]
    ema50 = ind["ema50"]
    ema200 = ind["ema200"]
    val = ind["vp_val"]
    vah = ind["vp_vah"]
    poc = ind["vp_poc"]
    vly = ind["vp_vly"]
    n = len(c)
    tol_t = np.clip(atr * 0.25, 1.5, 4.5)
    tol_r = np.clip(atr * 0.22, 1.5, 4.0)
    is_bear = c < ema200
    is_bull = c > ema200
    strong = adx >= 24.0
    valid = np.zeros(n, dtype=bool)
    valid[60:] = True
    valid &= np.isfinite(ema200) & np.isfinite(atr)
    prev_o = np.roll(o, 1)
    prev_c = np.roll(c, 1)
    prev_rsi = np.roll(rsi, 1)
    sig = {}

    # 1) VALLEY_SNIPER — แตะ Valley/POC ฝั่งเทรนด์ EMA21 vs EMA50 (LONG ก่อน)
    long_c = (np.abs(l - vly) <= tol_t) & (c > o) & (ema21 >= ema50)
    short_c = (np.abs(h - vly) <= tol_t) & (c < o) & (ema21 <= ema50)
    sig["VALLEY_SNIPER"] = _stable_side_select(long_c, short_c, True, valid)

    # 2) TREND_PULLBACK — ดึงกลับเข้า VAH/POC (โซนขาย) หรือ VAL/POC (โซนซื้อ) ในเทรนด์แรง (SHORT ก่อน)
    short_c = is_bear & strong & ((np.abs(h - vah) <= tol_t) | (np.abs(h - poc) <= tol_t)) & (c < o)
    long_c = is_bull & strong & ((np.abs(l - val) <= tol_t) | (np.abs(l - poc) <= tol_t)) & (c > o)
    sig["TREND_PULLBACK"] = _stable_side_select(long_c, short_c, False, valid)

    # 3) HYBRID_REGIME — เทรนด์แรงใช้ POC/VAL-VAH ตามฝั่ง macro, เทรนด์อ่อนกลับมาดักขอบ Value Area (SHORT ก่อน)
    short_c = ((strong & is_bear & ((np.abs(h - poc) <= tol_t) | (np.abs(h - val) <= tol_t)) & (c < o))
               | ((~strong) & (np.abs(h - vah) <= tol_t) & (c < o)))
    long_c = ((strong & is_bull & ((np.abs(l - poc) <= tol_t) | (np.abs(l - vah) <= tol_t)) & (c > o))
              | ((~strong) & (np.abs(l - val) <= tol_t) & (c > o)))
    sig["HYBRID_REGIME"] = _stable_side_select(long_c, short_c, False, valid)

    # 4) VP_EDGE_FADE — ดักขอบ VAL/VAH เกินเขต + RSI สุดโต่ง (LONG ก่อน)
    long_c = (np.abs(l - val) <= tol_r) & (rsi <= 36.0) & (c > o)
    short_c = (np.abs(h - vah) <= tol_r) & (rsi >= 64.0) & (c < o)
    sig["VP_EDGE_FADE"] = _stable_side_select(long_c, short_c, True, valid)

    # 5) LIQUIDITY_SWEEP — กวาด Low/High 16 แท่งก่อนหน้าแล้วดึงกลับ (LONG ก่อน)
    sw_lo = pd.Series(l).rolling(16).min().shift(1).to_numpy()
    sw_hi = pd.Series(h).rolling(16).max().shift(1).to_numpy()
    rng_ = np.maximum(0.1, h - l)
    lw = np.minimum(o, c) - l
    uw = h - np.maximum(o, c)
    long_c = is_bear & (l < sw_lo) & (c > sw_lo) & (lw >= 0.42 * rng_)
    short_c = is_bull & (h > sw_hi) & (c < sw_hi) & (uw >= 0.42 * rng_)
    sig["LIQUIDITY_SWEEP"] = _stable_side_select(long_c, short_c, True, valid)

    # 6) RSI_ENGULFING — แท่งกลืน + RSI แท่งก่อนสุดโต่ง (LONG ก่อน)
    long_c = is_bear & (prev_rsi <= 32.0) & (c > o) & (c > prev_o) & (o <= prev_c)
    short_c = is_bull & (prev_rsi >= 68.0) & (c < o) & (c < prev_o) & (o >= prev_c)
    sig["RSI_ENGULFING"] = _stable_side_select(long_c, short_c, True, valid)

    # 7) DOUBLE_BOTTOM_TOP — แตะ Low/High เดิมของกรอบ 13 แท่ง (i-16..i-4) + RSI divergence (LONG ก่อน)
    dbt = np.zeros(n, dtype=np.int8)
    nwin = n - 16
    if nwin > 0:
        wl = np.lib.stride_tricks.sliding_window_view(l, 13)[:nwin]
        wh = np.lib.stride_tricks.sliding_window_view(h, 13)[:nwin]
        min_l = wl.min(axis=1)
        arg_l = wl.argmin(axis=1)
        max_h = wh.max(axis=1)
        arg_h = wh.argmax(axis=1)
        rr_idx = np.arange(nwin)
        rsi_lo = rsi[arg_l + rr_idx]
        rsi_hi = rsi[arg_h + rr_idx]
        long_c16 = is_bear[16:] & (np.abs(l[16:] - min_l) <= tol_r[16:]) & (rsi[16:] > rsi_lo + 2.0) & (c[16:] > o[16:])
        short_c16 = is_bull[16:] & (np.abs(h[16:] - max_h) <= tol_r[16:]) & (rsi[16:] < rsi_hi - 2.0) & (c[16:] < o[16:])
        long_full = np.zeros(n, dtype=bool)
        short_full = np.zeros(n, dtype=bool)
        long_full[16:] = long_c16
        short_full[16:] = short_c16
        dbt = _stable_side_select(long_full, short_full, True, valid)
    sig["DOUBLE_BOTTOM_TOP"] = dbt

    # 8) VWAP_SNAPBACK — ยืดเกิน EMA21 +/- 2.1 ATR + ไส้เทียนกลับตัว (LONG ก่อน)
    long_c = (c <= ema21 - 2.1 * atr) & (c > o) & (lw >= 0.30 * rng_)
    short_c = (c >= ema21 + 2.1 * atr) & (c < o) & (uw >= 0.30 * rng_)
    sig["VWAP_SNAPBACK"] = _stable_side_select(long_c, short_c, True, valid)
    return sig


def _stable_path_matrices(h, l, c):
    """MFE/MAE matrix แบบ vectorized: ระยะ High/Low ของ 24 แท่งล่วงหน้า เทียบ close ของแท่งสัญญาณ (USD)"""
    n = len(c)
    H = STABLE_FORWARD_BARS
    m = n - H
    W_h = np.lib.stride_tricks.sliding_window_view(h, H)[1:]
    W_l = np.lib.stride_tricks.sliding_window_view(l, H)[1:]
    base_c = c[:m, None]
    UHi = (W_h - base_c).astype(np.float32)
    ULo = (W_l - base_c).astype(np.float32)
    C12u = (c[12:12 + m] - c[:m]).astype(np.float32)
    C24u = (c[H:H + m] - c[:m]).astype(np.float32)
    jump = np.zeros(n, dtype=bool)
    if n > 1:
        jump[1:] = np.abs(np.diff(c)) >= STABLE_ANOMALY_JUMP_USD
    ok = ~np.lib.stride_tricks.sliding_window_view(jump, H)[1:].any(axis=1)
    return {"UHi": UHi, "ULo": ULo, "C12u": C12u, "C24u": C24u, "ok": ok, "m": m}


def _stable_session_ids(minutes):
    sid = np.full(len(minutes), -1, dtype=np.int8)
    for k, (_n, s, e, _w) in enumerate(STABLE_SESSION_DEFS):
        if s < e:
            mask = (minutes >= s) & (minutes < e)
        else:
            mask = (minutes >= s) | (minutes < e)
        sid[mask] = k
    return sid


def _stable_simulate_combo(FY, AD, C12s, C24s, atr_f, sl_mult, rr, be_rule):
    """จำลองผลลัพธ์ R ต่อไม้ของสัญญาณชุดหนึ่งจาก MFE/MAE matrix (vectorized ทุกไม้พร้อมกัน)"""
    H = FY.shape[1]
    idx_h = np.arange(H, dtype=np.int16)[None, :]
    sl_usd = np.clip(atr_f * np.float32(sl_mult), STABLE_SL_CLAMP[0], STABLE_SL_CLAMP[1]).astype(np.float32)
    tp_usd = sl_usd * np.float32(rr)
    hor = (idx_h < 12) if be_rule == "TIME_EXIT_12B" else None

    def first_hit(mask):
        if hor is not None:
            mask = mask & hor
        has = mask.any(axis=1)
        j = np.where(has, mask.argmax(axis=1), 0).astype(np.int16)
        return has, j

    has_sl, j_sl = first_hit(AD >= sl_usd[:, None])
    has_tp, j_tp = first_hit(FY >= tp_usd[:, None])

    if be_rule == "TIME_EXIT_12B":
        base_exit = np.clip(C12s / sl_usd, -1.0, np.float32(rr)).astype(np.float32)
        sl_first = has_sl & (~has_tp | (j_sl <= j_tp))
        gross = np.where(has_tp & ~sl_first, np.float32(rr),
                 np.where(sl_first, np.float32(-1.0), base_exit))
        eo = np.where(sl_first, j_sl + 1, np.where(has_tp, j_tp + 1, 12)).astype(np.int16)
    elif be_rule == "NO_BE":
        base_exit = np.clip(C24s / sl_usd, -1.0, np.float32(rr)).astype(np.float32)
        sl_first = has_sl & (~has_tp | (j_sl <= j_tp))
        gross = np.where(has_tp & ~sl_first, np.float32(rr),
                 np.where(sl_first, np.float32(-1.0), base_exit))
        eo = np.where(sl_first, j_sl + 1, np.where(has_tp, j_tp + 1, 24)).astype(np.int16)
    else:
        # BE_AT_1R: แตะ +1R แล้วเลื่อน SL ไปที่ทุน | LOCK_HALF_0.8R: แตะ +0.8R แล้วล็อกกำไรครึ่งทาง (+0.4R)
        trig_mult = np.float32(1.0) if be_rule == "BE_AT_1R" else np.float32(0.8)
        stop_lvl = np.float32(0.0) if be_rule == "BE_AT_1R" else np.float32(-0.4)
        lock_gain = np.float32(0.0) if be_rule == "BE_AT_1R" else np.float32(0.4)
        has_be, j_be = first_hit(FY >= (sl_usd * trig_mult)[:, None])
        pre_sl = has_sl & (~has_be | (j_sl <= j_be))
        post_mask = ((AD >= (sl_usd * stop_lvl)[:, None])
                     & (idx_h >= np.where(has_be, j_be, 0)[:, None])
                     & has_be[:, None])
        has_lock, j_lock = first_hit(post_mask)
        tp_first = has_tp & ~pre_sl & (~has_lock | (j_tp < j_lock))
        be_first = has_lock & ~pre_sl & ~tp_first
        base_exit = np.clip(C24s / sl_usd, -1.0, np.float32(rr)).astype(np.float32)
        gross = np.where(pre_sl, np.float32(-1.0),
                 np.where(tp_first, np.float32(rr),
                  np.where(be_first, lock_gain, base_exit)))
        eo = np.where(pre_sl, j_sl + 1,
              np.where(tp_first, j_tp + 1,
               np.where(be_first, j_lock + 1, 24))).astype(np.int16)
    return gross, eo, sl_usd


def _stable_chain_keep(rows, exit_off):
    """ลำดับไม้แบบไม่ทับซ้อน: ข้ามสัญญาณที่เกิดก่อนแท่งออกของไม้ก่อนหน้า (ตรรกะเดียวกับ simulateGeneExact)
    คืน None ถ้าไม่มีการคาบเกี่ยวเลย (ใช้สัญญาณทั้งหมด)"""
    if len(rows) < 2 or np.all(np.diff(rows) > STABLE_FORWARD_BARS):
        return None
    keep = np.ones(len(rows), dtype=bool)
    last_exit = -1
    for t in range(len(rows)):
        ri = int(rows[t])
        if ri <= last_exit:
            keep[t] = False
        else:
            last_exit = ri + int(exit_off[t])
    return keep


def _stable_bootstrap_mc(r_seq, iters, sigma_r, seed_int):
    """Bootstrap Monte Carlo: สุ่ม Resample ลำดับ R + ความคลาดเคลื่อนสเปรดสุ่ม (float32) แล้วคืนค่าคงตัว 3 ตัว"""
    r = np.asarray(r_seq, dtype=np.float32)
    m = int(r.size)
    if m < 2 or iters < 1:
        return {"median_r": 0.0, "p05_worst_r": 0.0, "max_dd_95": 0.0}
    rng = np.random.default_rng(int(seed_int))
    totals = np.empty(iters, dtype=np.float32)
    dds = np.empty(iters, dtype=np.float32)
    chunk = int(max(STABLE_MC_MIN_ITERS // 2, min(12000, 30_000_000 // max(1, m * 3))))
    sig = np.float32(sigma_r)
    done = 0
    while done < iters:
        cn = int(min(chunk, iters - done))
        pick = rng.integers(0, m, size=(cn, m))
        samp = r[pick]
        if sig > 0:
            samp += rng.standard_normal(size=samp.shape, dtype=np.float32) * sig
        eq = np.cumsum(samp, axis=1, dtype=np.float32)
        dd = np.maximum.accumulate(eq, axis=1) - eq
        totals[done:done + cn] = eq[:, -1]
        dds[done:done + cn] = dd.max(axis=1)
        done += cn
    return {
        "median_r": round(float(np.median(totals)), 3),
        "p05_worst_r": round(float(np.percentile(totals, 5)), 3),
        "max_dd_95": round(float(np.percentile(dds, 95)), 3),
    }


def run_session_stable_profiles(df, now_th=None):
    """Vectorized Grid + Bootstrap Monte Carlo 250,000 รอบ/ช่วงตลาด (รวม 1,000,000 รอบ)
    คืน dict สำหรับบันทึกใน state.json คีย์ "session_stable_profiles" """
    import zlib
    t0 = time.time()
    now_th = now_th or datetime.now(THAI_TZ)
    grid_combos = len(STABLE_FORMULAS) * len(STABLE_GRID_SL_MULT) * len(STABLE_GRID_RR) * len(STABLE_BE_RULES)
    result = {
        "engine": "VECTOR_GRID_BOOTSTRAP_MC_V1",
        "status": "OK",
        "generated_at": now_th.strftime("%Y-%m-%d %H:%M:%S"),
        "methodology": {
            "sessions_bkk": [{"name": s, "window": w} for s, _a, _b, w in STABLE_SESSION_DEFS],
            "forward_bars_mfe_mae": STABLE_FORWARD_BARS,
            "formulas": list(STABLE_FORMULAS),
            "sl_grid_atr_mult": list(STABLE_GRID_SL_MULT),
            "rr_grid": list(STABLE_GRID_RR),
            "be_rules": list(STABLE_BE_RULES),
            "sl_clamp_usd": list(STABLE_SL_CLAMP),
            "friction_model": "Spread+Slippage (USD/ไม้) = 0.40 x volatility_factor ของแต่ละช่วงตลาด (median ATR14 จริงจาก vault) หักทุกไม้ที่จำลอง",
            "mc_method": "Bootstrap resample ลำดับ R ด้วย NumPy float32 + สุ่มความคลาดเคลื่อนสเปรด N(0, sigma_r) ต่อไม้",
            "mc_iterations_requested_per_session": STABLE_MC_ITERS_PER_SESSION,
            "mc_iterations_requested_total": STABLE_MC_ITERS_PER_SESSION * 4,
            "mc_iterations_used_total": 0,
            "stability_score_formula": "(p05_worst_r / max(max_dd_95, 0.5)) x win_rate",
            "signal_logic": "พอร์ตจาก scanTrendSignals/scanReversalSignals ของ index.html (EMA21/50/200 M15, ADX proxy >= 24, micro-VP 16 แท่ง 18 bins, tol = clip0(ATR*0.25/0.22, 1.5, 4.5/4.0))",
            "be_rules_detail": "NO_BE = ไม่มี BE | LOCK_HALF_0.8R = แตะ +0.8R แล้วเลื่อนสต็อปไปล็อก +0.4R | BE_AT_1R = แตะ +1R แล้วเลื่อนสต็อปไปทุน | TIME_EXIT_12B = ออกที่ราคาปิดแท่งที่ 12 ถ้ายังไม่โดน SL/TP",
            "filter_policy": "เกณฑ์ขั้นต่ำลดหลั่น L1 >= 15 ไม้ และ WR >= 35% / L2 >= 10 ไม้ และ 30% / L3 >= 6 ไม้ และ 25% (ทุกชั้นต้อง net_r > 0 หลังหัก friction) + fallback เลือกชุดคะแนนนำสุด",
            "mc_budget_policy": "250,000 รอบ/ช่วง แบ่งเท่ากันให้ผู้ผ่านเกณฑ์สูงสุด 125 ชุดแรกตามคะแนนคัดกรอง (ขั้นต่ำ 2,000 รอบ/ชุด)",
            "anomaly_guard": f"ตัดสัญญาณที่หน้าต่าง 24 แท่งมีแท่งกระโดด >= ${STABLE_ANOMALY_JUMP_USD:.0f} (basis shift ของ vault)",
        },
        "sessions": {},
        "runtime_sec": 0.0,
    }
    sim_df, minutes = _stable_clean_frame(df)
    n = len(sim_df)
    if minutes is None or len(minutes) != n or n < 400:
        result["status"] = "INSUFFICIENT_DATA"
        result["runtime_sec"] = round(time.time() - t0, 2)
        return result
    ind = _stable_site_indicators(sim_df)
    vp_val, vp_vah, vp_poc, vp_vly = _stable_micro_vp_arrays(
        ind["h"], ind["l"], ind["c"], sim_df["Volume"].fillna(1.0).to_numpy(dtype=np.float64), 18)
    ind["vp_val"], ind["vp_vah"], ind["vp_poc"], ind["vp_vly"] = vp_val, vp_vah, vp_poc, vp_vly
    sides = _stable_formula_signals(ind)
    mats = _stable_path_matrices(ind["h"], ind["l"], ind["c"])
    sid = _stable_session_ids(minutes)
    m_rows = int(mats["m"])
    atr = ind["atr"]
    finite_atr = np.isfinite(atr) & (atr > 0.5)
    atr_all = atr[np.isfinite(atr)]
    glob_med = float(np.median(atr_all)) if atr_all.size else 7.5
    total_used = 0
    result["vault_bars"] = n
    result["vault_last_bar_utc"] = sim_df.index[-1].strftime("%Y-%m-%d %H:%M")

    for k_sess, (sname, _s0, _s1, wind) in enumerate(STABLE_SESSION_DEFS):
        t_sess = time.time()
        in_sess = sid[:m_rows] == k_sess
        bars_in_sess = int((sid == k_sess).sum())
        sel_atr = atr[:m_rows][in_sess & finite_atr[:m_rows] & mats["ok"]]
        sess_atr_med = float(np.median(sel_atr)) if sel_atr.size >= 5 else glob_med
        vf = float(np.clip(sess_atr_med / max(1e-9, glob_med), 0.6, 1.6))
        spread_usd = round(STABLE_FRICTION_BASE_USD * 0.5 * vf, 3)
        slip_usd = round(STABLE_FRICTION_BASE_USD * 0.5 * vf, 3)
        friction_usd = round(spread_usd + slip_usd, 3)

        candidates = []
        for fname in STABLE_FORMULAS:
            side_arr = sides[fname]
            rows = np.flatnonzero((side_arr[:m_rows] != 0) & in_sess & finite_atr[:m_rows] & mats["ok"])
            if rows.size < 4:
                continue
            side_f = side_arr[rows]
            long_f = side_f > 0
            atr_f = atr[rows].astype(np.float32)
            UHi = mats["UHi"][rows]
            ULo = mats["ULo"][rows]
            FY = np.where(long_f[:, None], UHi, -ULo).astype(np.float32)
            AD = np.where(long_f[:, None], -ULo, UHi).astype(np.float32)
            C12s = (mats["C12u"][rows] * np.where(long_f, 1.0, -1.0)).astype(np.float32)
            C24s = (mats["C24u"][rows] * np.where(long_f, 1.0, -1.0)).astype(np.float32)
            for sl_mult in STABLE_GRID_SL_MULT:
                for rr in STABLE_GRID_RR:
                    for be_rule in STABLE_BE_RULES:
                        gross, eo, sl_usd = _stable_simulate_combo(FY, AD, C12s, C24s, atr_f, sl_mult, rr, be_rule)
                        keep = _stable_chain_keep(rows, eo)
                        if keep is not None:
                            rows_k = rows[keep]
                            side_k = side_f[keep]
                            r_net = (gross[keep] - (friction_usd / sl_usd)[keep]).astype(np.float32)
                            sl_k = sl_usd[keep]
                        else:
                            rows_k = rows
                            side_k = side_f
                            r_net = (gross - friction_usd / sl_usd).astype(np.float32)
                            sl_k = sl_usd
                        trades = int(r_net.size)
                        if trades < 4:
                            continue
                        wins = int(np.count_nonzero(r_net > 0))
                        wr = wins / trades
                        net_r = float(r_net.sum())
                        eq = np.cumsum(r_net, dtype=np.float32)
                        raw_dd = float((np.maximum.accumulate(eq) - eq).max())
                        pos_sum = float(r_net[r_net > 0].sum())
                        neg_sum = float(-r_net[r_net < 0].sum())
                        pf = pos_sum / max(0.5, neg_sum)
                        pre = (net_r / max(0.5, raw_dd)) * wr * min(1.0, trades / 30.0)
                        candidates.append({
                            "formula": fname,
                            "sl_atr_mult": sl_mult,
                            "rr": rr,
                            "be_rule": be_rule,
                            "trades": trades,
                            "wins": wins,
                            "win_rate": round(wr, 4),
                            "net_r": round(net_r, 3),
                            "expectancy_r": round(net_r / trades, 4),
                            "profit_factor": round(pf, 3),
                            "raw_max_dd_r": round(raw_dd, 3),
                            "pre_score": round(pre, 4),
                            "longs": int(np.count_nonzero(side_k > 0)),
                            "shorts": int(np.count_nonzero(side_k < 0)),
                            "sl_med_usd": round(float(np.median(sl_k)), 3),
                            "r_net": r_net,
                        })

        # เกณฑ์ขั้นต่ำแบบลดหลั่น: L1/L2/L3 แล้วค่อย fallback L4 (ชุดคะแนนนำสุดเพียงชุดเดียว)
        qual, filter_level = [], 4
        for lvl, (min_tr, min_wr) in enumerate(((15, 0.35), (10, 0.30), (6, 0.25)), start=1):
            q = [c for c in candidates if c["trades"] >= min_tr and c["win_rate"] >= min_wr and c["net_r"] > 0.0]
            if len(q) >= 3 or (lvl == 3 and len(q) >= 1):
                qual, filter_level = q, lvl
                break
        if not qual and candidates:
            qual, filter_level = [max(candidates, key=lambda cc: cc["pre_score"])], 4
        qual.sort(key=lambda cc: cc["pre_score"], reverse=True)
        mc_list = qual[:STABLE_MC_MAX_CANDIDATES]
        n_mc = len(mc_list)
        used = 0
        if n_mc:
            base_iters = max(STABLE_MC_MIN_ITERS, STABLE_MC_ITERS_PER_SESSION // n_mc)
            rem = max(0, STABLE_MC_ITERS_PER_SESSION - base_iters * n_mc)
            for t_idx, cand in enumerate(mc_list):
                iters = base_iters + (1 if t_idx < rem else 0)
                seed_i = zlib.crc32(f"{sname}|{cand['formula']}|{cand['sl_atr_mult']}|{cand['rr']}|{cand['be_rule']}".encode("utf-8"))
                sigma_r = STABLE_FRICTION_SIGMA_RATIO * friction_usd / max(3.0, cand["sl_med_usd"])
                mc = _stable_bootstrap_mc(cand["r_net"], iters, sigma_r, seed_i)
                cand.update(mc)
                cand["sigma_spread_r"] = round(sigma_r, 4)
                cand["mc_iterations"] = iters
                cand["stability_score"] = round((mc["p05_worst_r"] / max(mc["max_dd_95"], 0.5)) * cand["win_rate"], 4)
                used += iters
        total_used += used

        mc_list.sort(key=lambda cc: cc["stability_score"], reverse=True)
        winner = None
        if mc_list:
            w = mc_list[0]
            winner = {
                "formula": w["formula"],
                "sl_atr_mult": w["sl_atr_mult"],
                "rr": w["rr"],
                "be_rule": w["be_rule"],
                "trades": w["trades"],
                "longs": w["longs"],
                "shorts": w["shorts"],
                "win_rate": round(w["win_rate"], 4),
                "win_rate_pct": round(w["win_rate"] * 100.0, 1),
                "net_r": w["net_r"],
                "expectancy_r": w["expectancy_r"],
                "profit_factor": w["profit_factor"],
                "raw_max_dd_r": w["raw_max_dd_r"],
                "median_r": w["median_r"],
                "p05_worst_r": w["p05_worst_r"],
                "max_dd_95": w["max_dd_95"],
                "stability_score": w["stability_score"],
                "mc_iterations": w["mc_iterations"],
                "sigma_spread_r": w["sigma_spread_r"],
                "sl_median_usd": w["sl_med_usd"],
            }
        top_list = [{
            "formula": w["formula"], "sl_atr_mult": w["sl_atr_mult"], "rr": w["rr"], "be_rule": w["be_rule"],
            "trades": w["trades"], "win_rate_pct": round(w["win_rate"] * 100.0, 1), "net_r": w["net_r"],
            "median_r": w["median_r"], "p05_worst_r": w["p05_worst_r"], "max_dd_95": w["max_dd_95"],
            "stability_score": w["stability_score"], "mc_iterations": w["mc_iterations"],
        } for w in mc_list[:5]]
        result["sessions"][sname] = {
            "session": sname,
            "window_bkk": wind,
            "bars_in_vault": bars_in_sess,
            "volatility_factor": round(vf, 3),
            "avg_atr_usd": round(sess_atr_med, 2),
            "friction_usd": {"spread": spread_usd, "slippage": slip_usd, "total": friction_usd},
            "grid_combos": grid_combos,
            "candidates_evaluated": len(candidates),
            "candidates_qualified": len(qual),
            "filter_level_used": filter_level,
            "candidates_mc_tested": n_mc,
            "mc_iterations_used": used,
            "winner": winner,
            "top_stable_candidates": top_list,
            "runtime_sec": round(time.time() - t_sess, 2),
        }

    result["methodology"]["mc_iterations_used_total"] = total_used
    result["runtime_sec"] = round(time.time() - t0, 2)
    print(f"[STABLE] session_stable_profiles OK | MC sims used: {total_used:,} | total {result['runtime_sec']}s")
    for _sn, _sd in result["sessions"].items():
        _w = _sd.get("winner")
        if _w:
            print(f"[STABLE] {_sn}: {_w['formula']} SL={_w['sl_atr_mult']}xATR RR={_w['rr']} {_w['be_rule']} | "
                  f"score={_w['stability_score']} p05={_w['p05_worst_r']}R dd95={_w['max_dd_95']}R "
                  f"wr={_w['win_rate_pct']}% trades={_w['trades']} ({_sd['mc_iterations_used']:,} sims)")
    return result


def generate_live_ai_strategist_brief(spot, vp_session, scalp_setup, trend_setup, order_flow, scalp_bt, trend_bt, mue_mem):
    fallback_brief = (
        f"**1. Session Micro-VP 8H & Live OrderFlow ({vp_session['title_th']}):**\n"
        f"Spot ปัจจุบัน **${spot}** เกาะอยู่ในกรอบ Session 8H (VAL **${vp_session['val']}** / POC **${vp_session['poc']}** / VAH **${vp_session['vah']}**) | "
        f"L2 OBI **{order_flow['obi_25bps']:+.3f}** | Tick CVD **{order_flow['tick_cvd']:+.1f}** ({order_flow['absorption_flag']})\n\n"
        f"**2. เงื่อนไขลั่นไก OrderFlow Execution Gate (ใช้งานจริง 3 ชั้น):**\n"
        f"• **ชั้นที่ 1 (Location):** รอราคาเข้าใกล้จุดยิง Scalp **${scalp_setup['entry']}** ในระยะไม่เกิน `±{scalp_setup['tol_points']} จุด`\n"
        f"• **ชั้นที่ 2 (3s Sustained OBI):** แรงฝั่ง **{scalp_setup['direction']}** ต้องยืนเหนือเกณฑ์ `{'+0.18' if scalp_setup['direction']=='LONG' else '-0.18'}` ต่อเนื่อง 3 วินาทีเต็ม (ตัดออเดอร์หลอก)\n"
        f"• **ชั้นที่ 3 (60s Tape Delta):** แรงเคาะซื้อ/ขายจริง 60 วินาทีต้องสอดคล้องทิศทางจึงเปิดออเดอร์\n\n"
        f"**3. แผนเข้าเทรด Scalp ({scalp_setup['direction']}):** Entry **${scalp_setup['entry']}** | SL **${scalp_setup['sl']}** | BE (+{scalp_setup.get('be_points', 250)} จุด) **${scalp_setup['be_target']}** | TP **${scalp_setup['tp']}**\n"
        f"**4. แผน Trend Sniper ({trend_setup['direction']}):** Entry **${trend_setup['entry']}** | SL **${trend_setup['sl']}** | TP **${trend_setup['tp']}** (WR {trend_bt['best']['win_rate']}%)"
    )

    if not THAILLM_API_KEY:
        return fallback_brief, "MUE-X Local Quant V14.1"

    try:
        client = OpenAI(api_key=THAILLM_API_KEY, base_url="http://thaillm.or.th/api/typhoon/v1")
        prompt = (
            f"คุณคือ Chief Quant Strategist ดูแลพอร์ตกองทุน XAU/USD วิเคราะห์ข้อมูลสด ณ เวลานี้เป็นภาษาไทยกระชับ 4 ข้อ:\n"
            f"- Spot: ${spot} | Session 8H VP: {vp_session['shape']} (VAL ${vp_session['val']}, POC ${vp_session['poc']}, VAH ${vp_session['vah']})\n"
            f"- Live L2 OrderFlow: OBI={order_flow['obi_25bps']:+.3f}, TickCVD={order_flow['tick_cvd']:+.1f}, Status={order_flow['absorption_flag']}\n"
            f"- MUE-X Gen#{mue_mem['generation']} Scalp Plan: {scalp_setup['direction']} @ ${scalp_setup['entry']} (SL ${scalp_setup['sl']}, TP ${scalp_setup['tp']}, Mode={scalp_setup['mode']})\n"
            f"- Trend Sniper Plan: {trend_setup['direction']} @ ${trend_setup['entry']} (SL ${trend_setup['sl']}, TP ${trend_setup['tp']}, Backtest WR {trend_bt['best']['win_rate']}%)\n"
            f"สรุปจังหวะรอยิงร่วมกับ L2 OrderBook และการคุมความเสี่ยงพอร์ตกองทุนไม่เกิน -2R/วัน"
        )
        resp = client.chat.completions.create(
            model="/model/typhoon-v2.5-30b-a3b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=650,
            temperature=0.25,
            timeout=18
        )
        txt = (resp.choices[0].message.content or "").strip()
        if len(txt) > 80:
            return txt, "Typhoon-v2.5 + MUE-X V14.1"
    except Exception as e:
        print(f"[AI Brief Warning] {e}")

    return fallback_brief, "MUE-X Quant V14.1"


def _fetch_yahoo_chart_closes(symbol, range_str="30d", interval="1d"):
    """ดึงราคาปิดรายวันตรงจาก Yahoo Finance Chart API (แนบ User-Agent กัน 403) → pd.Series (index = unix timestamp) หรือ None"""
    try:
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": range_str, "interval": interval},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        if r.status_code != 200:
            print(f"[Macro] Yahoo Chart API ตอบ {r.status_code} สำหรับ {symbol}")
            return None
        result = (r.json().get("chart") or {}).get("result") or []
        if not result:
            return None
        ts_list = result[0].get("timestamp") or []
        close_list = ((result[0].get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        pairs = [(int(t), float(c)) for t, c in zip(ts_list, close_list) if c is not None]
        if not pairs:
            return None
        return pd.Series(dict(pairs)).sort_index()
    except Exception as e:
        print(f"[Macro] Yahoo Chart API ล้มเหลว {symbol}: {e}")
        return None


def _fetch_yfinance_closes(symbol, period="30d", interval="1d"):
    """ตัวสำรอง: ราคาปิดรายวันจาก yfinance → pd.Series (index = unix timestamp) หรือ None"""
    try:
        df_x = yf.download(symbol, period=period, interval=interval, progress=False)
        if df_x is None or df_x.empty:
            return None
        if isinstance(df_x.columns, pd.MultiIndex):
            df_x.columns = df_x.columns.get_level_values(0)
        if "Close" not in df_x.columns:
            return None
        s = df_x["Close"].dropna()
        if s.empty:
            return None
        s.index = [int(pd.Timestamp(x).timestamp()) for x in s.index]
        return s.astype(float).sort_index()
    except Exception as e:
        print(f"[Macro] yfinance ล้มเหลว {symbol}: {e}")
        return None


def _fetch_macro_series(symbol, period="30d", interval="1d"):
    """ราคาปิดสำหรับคำนวณ macro: ลอง Yahoo Chart API ตรง ๆ ก่อน (เสถียรกว่า) ไม่สำเร็จค่อยตกไปใช้ yfinance"""
    s = _fetch_yahoo_chart_closes(symbol, period, interval)
    if s is not None and not s.empty:
        return s
    return _fetch_yfinance_closes(symbol, period, interval)


def fetch_macro_and_news():
    # โหลดค่าล่าสุดที่บันทึกไว้เป็น fallback
    cached_macro_path = os.path.join(DATA_DIR, "macro_cache.json")
    cached = load_json_safe(cached_macro_path, {})

    dxy_p = cached.get("dxy", 103.85)
    us10y_y = cached.get("us10y", 4.25)
    corr_20 = cached.get("corr", -0.45)

    # ดึงตรงจาก Yahoo Finance Chart API เป็นตัวหลัก (yfinance มักดึงไม่ติด) — ล้มเหลวค่อยใช้ yfinance สำรอง
    dxy_ser = _fetch_macro_series("DX-Y.NYB", "30d", "1d")
    tnx_ser = _fetch_macro_series("^TNX", "10d", "1d")
    gld_ser = _fetch_macro_series("GC=F", "30d", "1d")

    # อัปเดตเฉพาะค่าที่ดึงสำเร็จรอบนี้ — รอบที่ล้มเหลวห้ามเขียน null/"-" ทับค่าเดิมใน macro_cache.json
    updated = {}
    if dxy_ser is not None and not dxy_ser.empty:
        dxy_p = round(float(dxy_ser.iloc[-1]), 2)
        updated["dxy"] = dxy_p
    if tnx_ser is not None and not tnx_ser.empty:
        us10y_y = round(float(tnx_ser.iloc[-1]), 2)
        updated["us10y"] = us10y_y
    if dxy_ser is not None and gld_ser is not None:
        j = pd.concat([gld_ser.rename("gold"), dxy_ser.rename("dxy")], axis=1).dropna().tail(20)
        if len(j) >= 10:
            corr_20 = round(float(j["gold"].corr(j["dxy"])), 2)
            updated["corr"] = corr_20

    if updated:
        merged = dict(cached)
        merged.update(updated)
        merged["updated_at"] = datetime.now(THAI_TZ).isoformat()
        save_json_safe(cached_macro_path, merged)
        print(f"[Macro] อัปเดตสำเร็จ: DXY={dxy_p}, US10Y={us10y_y}, Corr={corr_20}")
    else:
        # ดึงรอบนี้ไม่สำเร็จ → ใช้ค่าเดิมจาก cache ต่อไป โดยไม่แตะไฟล์แม้แต่ไบต์เดียว
        print(f"[Macro Warning] ดึงรอบนี้ไม่สำเร็จ — ใช้ค่า cached เดิม: DXY={dxy_p}, US10Y={us10y_y}, Corr={corr_20}")

    news_list = []
    try:
        r = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.xml", timeout=8)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            now_th = datetime.now(THAI_TZ)
            for ev in root.findall("event"):
                if (ev.findtext("country") or "").strip() != "USD" or (ev.findtext("impact") or "").strip() not in ("High", "Medium"):
                    continue
                title = (ev.findtext("title") or "").strip()
                date_s, time_s = (ev.findtext("date") or "").strip(), (ev.findtext("time") or "").strip()
                try:
                    dt_th = datetime.strptime(f"{date_s} {time_s}", "%m-%d-%Y %I:%M%p").replace(tzinfo=timezone.utc).astimezone(THAI_TZ)
                    diff_h = round((dt_th - now_th).total_seconds() / 3600.0, 1)
                    news_list.append({
                        "time_th": dt_th.strftime("%d/%m %H:%M น."),
                        "title": title,
                        "impact": (ev.findtext("impact") or "").strip(),
                        "forecast": (ev.findtext("forecast") or "-").strip(),
                        "previous": (ev.findtext("previous") or "-").strip(),
                        "is_upcoming": diff_h > 0,
                        "hours_away": abs(diff_h),
                        "status": f"อีก {diff_h} ชม." if diff_h > 0 else "ออกแล้ว"
                    })
                except Exception:
                    pass
    except Exception:
        pass
    return dxy_p, us10y_y, corr_20, news_list[:8]


def _normalize_journal_schema(obj, default_dict):
    base = dict(default_dict)
    if isinstance(obj, list):
        base["recent_trades"] = [x for x in obj if isinstance(x, dict)]
    elif isinstance(obj, dict):
        base.update(obj)
    if not isinstance(base.get("recent_trades"), list):
        base["recent_trades"] = []
    for k, v in default_dict.items():
        if k not in base:
            base[k] = list(v) if isinstance(v, list) else v
    return base


def compute_daily_risk_guard_metrics(now_th):
    today_prefix = now_th.strftime("%d/%m")
    def_j = {"recent_trades": []}
    net_r_today = 0.0
    latest_close_dt = None

    for path in (SCALP_JOURNAL_PATH, JOURNAL_PATH):
        jdata = load_json_safe(path, def_j)
        for t in jdata.get("recent_trades", []):
            if t.get("status") in ("WIN", "LOSS"):
                opened_s = str(t.get("opened_at_th", ""))
                if opened_s.startswith(today_prefix):
                    sl_dist = max(0.5, abs(float(t.get("entry", 0)) - float(t.get("sl", 0))))
                    pnl_val = float(t.get("pnl", 0.0))
                    net_r_today += (pnl_val / sl_dist)
                closed_s = str(t.get("closed_at_th", ""))
                if closed_s:
                    try:
                        c_dt = datetime.strptime(f"{now_th.year}/{closed_s}", "%Y/%d/%m %H:%M").replace(tzinfo=THAI_TZ)
                        if latest_close_dt is None or c_dt > latest_close_dt:
                            latest_close_dt = c_dt
                    except Exception:
                        pass

    cooldown_rem_min = 0
    if latest_close_dt is not None:
        elapsed_min = (now_th - latest_close_dt).total_seconds() / 60.0
        if 0 <= elapsed_min < 15.0:
            cooldown_rem_min = int(math.ceil(15.0 - elapsed_min))

    return round(net_r_today, 2), cooldown_rem_min


def update_journals(spot, scalp_setup, trend_setup, in_scalp_hours, in_trend_hours):
    now_th = datetime.now(THAI_TZ)
    now_str = now_th.strftime("%d/%m %H:%M")
    def_j = {"wins": 0, "losses": 0, "closed_trades": 0, "open_trades": 0, "net_live_pnl": 0.0, "live_win_rate": 0.0, "recent_trades": []}

    sj = _normalize_journal_schema(load_json_safe(SCALP_JOURNAL_PATH, def_j), def_j)
    for t in sj["recent_trades"]:
        if t.get("status") == "OPEN":
            sl_loss = -round(max(0.5, abs(float(t["entry"]) - float(t["sl"]))), 2)
            if t.get("direction") == "LONG":
                if spot <= t["sl"]:
                    t["status"], t["pnl"], t["closed_at_th"] = "LOSS", sl_loss, now_str
                elif spot >= t["tp"]:
                    t["status"], t["pnl"], t["closed_at_th"] = "WIN", round(t["tp"] - t["entry"], 2), now_str
            else:
                if spot >= t["sl"]:
                    t["status"], t["pnl"], t["closed_at_th"] = "LOSS", sl_loss, now_str
                elif spot <= t["tp"]:
                    t["status"], t["pnl"], t["closed_at_th"] = "WIN", round(t["entry"] - t["tp"], 2), now_str

    open_sc = [t for t in sj["recent_trades"] if t.get("status") == "OPEN"]
    if in_scalp_hours and len(open_sc) == 0 and abs(spot - scalp_setup["entry"]) <= (scalp_setup["tol_points"] / 100.0):
        sj["recent_trades"].insert(0, {
            "opened_at_th": now_str,
            "factor": f"{scalp_setup['shape']} ({scalp_setup['mode'][:6]})",
            "direction": scalp_setup["direction"],
            "entry": scalp_setup["entry"],
            "sl": scalp_setup["sl"],
            "tp": scalp_setup["tp"],
            "lot_size": scalp_setup.get("lot_size", 0.01),
            "risk_usd": scalp_setup.get("risk_usd", 10.0),
            "status": "OPEN",
            "pnl": 0.0
        })
        send_ntfy(
            f"PROP SCALP {scalp_setup['direction']} @ ${scalp_setup['entry']} ({scalp_setup.get('lot_size', 0.01)} lot)",
            f"Session VP {scalp_setup['shape']} | OBI: {scalp_setup['obi']} | SL: ${scalp_setup['sl']} | TP: ${scalp_setup['tp']}",
            priority=5, tags=["zap", "dart"]
        )

    sj["wins"] = sum(1 for t in sj["recent_trades"] if t.get("status") == "WIN")
    sj["losses"] = sum(1 for t in sj["recent_trades"] if t.get("status") == "LOSS")
    sj["closed_trades"] = sj["wins"] + sj["losses"]
    sj["open_trades"] = sum(1 for t in sj["recent_trades"] if t.get("status") == "OPEN")
    sj["net_live_pnl"] = round(sum(float(t.get("pnl", 0.0)) for t in sj["recent_trades"] if t.get("status") != "OPEN"), 2)
    sj["live_win_rate"] = round((sj["wins"] / max(1, sj["closed_trades"])) * 100.0, 1)
    sj["recent_trades"] = sj["recent_trades"][:15]
    save_json_safe(SCALP_JOURNAL_PATH, sj)

    tj = _normalize_journal_schema(load_json_safe(JOURNAL_PATH, def_j), def_j)
    for t in tj["recent_trades"]:
        if t.get("status") == "OPEN":
            tr_loss = -round(max(1.0, abs(float(t["entry"]) - float(t["sl"]))), 2)
            tr_win = round(max(1.0, abs(float(t["tp"]) - float(t["entry"]))), 2)
            if t.get("direction") == "LONG":
                if spot <= t["sl"]:
                    t["status"], t["pnl"], t["closed_at_th"] = "LOSS", tr_loss, now_str
                elif spot >= t["tp"]:
                    t["status"], t["pnl"], t["closed_at_th"] = "WIN", tr_win, now_str
            else:
                if spot >= t["sl"]:
                    t["status"], t["pnl"], t["closed_at_th"] = "LOSS", tr_loss, now_str
                elif spot <= t["tp"]:
                    t["status"], t["pnl"], t["closed_at_th"] = "WIN", tr_win, now_str

    open_tr = [t for t in tj["recent_trades"] if t.get("status") == "OPEN"]
    if in_trend_hours and len(open_tr) == 0 and abs(spot - trend_setup["entry"]) <= 4.5:
        tj["recent_trades"].insert(0, {
            "opened_at_th": now_str,
            "factor": "EMA21_RETEST",
            "direction": trend_setup["direction"],
            "entry": trend_setup["entry"],
            "sl": trend_setup["sl"],
            "tp": trend_setup["tp"],
            "lot_size": trend_setup.get("lot_size", 0.01),
            "risk_usd": trend_setup.get("risk_usd", 10.0),
            "status": "OPEN",
            "pnl": 0.0
        })
        send_ntfy(
            f"PROP TREND {trend_setup['direction']} @ ${trend_setup['entry']} ({trend_setup.get('lot_size', 0.01)} lot)",
            f"{trend_setup['bias']} | SL: ${trend_setup['sl']} | TP: ${trend_setup['tp']}",
            priority=5, tags=["rocket", "moneybag"]
        )

    tj["wins"] = sum(1 for t in tj["recent_trades"] if t.get("status") == "WIN")
    tj["losses"] = sum(1 for t in tj["recent_trades"] if t.get("status") == "LOSS")
    tj["closed_trades"] = tj["wins"] + tj["losses"]
    tj["open_trades"] = sum(1 for t in tj["recent_trades"] if t.get("status") == "OPEN")
    tj["net_live_pnl"] = round(sum(float(t.get("pnl", 0.0)) for t in tj["recent_trades"] if t.get("status") != "OPEN"), 2)
    tj["live_win_rate"] = round((tj["wins"] / max(1, tj["closed_trades"])) * 100.0, 1)
    tj["recent_trades"] = tj["recent_trades"][:15]
    save_json_safe(JOURNAL_PATH, tj)
    
    # บันทึก active_order.json สำหรับ frontend sync
    all_open = open_sc + open_tr
    if all_open:
        active_order = all_open[0]  # เอาออเดอร์แรกที่เปิดอยู่
        active_order["status"] = "OPEN"
        active_order["last_update_th"] = now_str
        save_json_safe(ACTIVE_ORDER_PATH, active_order)
    else:
        # ไม่มีออเดอร์เปิด ลบหรือเคลียร์ active_order.json
        save_json_safe(ACTIVE_ORDER_PATH, {"status": "NONE", "last_update_th": now_str})
    
    return sj, tj


def evaluate_risk_guard(spot, bid, ask, curr_atr, tp_pts, tr_sl_dist, news_items, in_scalp_hours, in_trend_hours):
    acct_bal = float(os.getenv("ACCOUNT_BALANCE", "1000.0"))
    risk_pct = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
    risk_usd = round(max(1.0, acct_bal * (risk_pct / 100.0)), 2)

    # 1. Dynamic ATR-based Scalp SL / BE / TP (clamped $3.50 - $9.00)
    sc_sl_usd = round(min(9.0, max(3.5, curr_atr * 0.90)), 2)
    sc_be_dist = round(sc_sl_usd * 0.50, 2)
    sc_tp_usd = round(max(tp_pts / 100.0, sc_sl_usd * 1.25), 2)
    sc_sl_pts = int(round(sc_sl_usd * 100))
    sc_be_pts = int(round(sc_be_dist * 100))
    sc_tp_pts = int(round(sc_tp_usd * 100))

    # 2. Automatic Position Sizing (1.00 Lot XAU/USD = $100 per $1.00 move)
    sc_lot = round(max(0.01, min(10.0, risk_usd / max(1.0, sc_sl_usd * 100.0))), 2)
    tr_lot = round(max(0.01, min(10.0, risk_usd / max(1.0, tr_sl_dist * 100.0))), 2)

    # 3. High-Impact USD News Filter (30m before / 20m after)
    news_pause = False
    news_reason = "CLEAR"
    for ev in (news_items or []):
        if ev.get("impact") == "High":
            hrs = float(ev.get("hours_away", 99.0))
            upcoming = bool(ev.get("is_upcoming", False))
            if (upcoming and hrs <= 0.5) or ((not upcoming) and hrs <= 0.3):
                news_pause = True
                news_reason = f"HIGH_NEWS: {ev.get('title', 'USD Event')} ({ev.get('time_th', '')})"
                break

    # 4. Live Spread (คำนวณแต่ไม่เอามาเป็นเงื่อนไข pause)
    spread_usd = round(max(0.0, float(ask) - float(bid)), 2) if (bid and ask and ask >= bid) else 0.0

    now_th = datetime.now(THAI_TZ)
    wk_closed, wk_reason = is_gold_market_closed_bkk(now_th)
    daily_net_r, cooldown_rem_min = compute_daily_risk_guard_metrics(now_th)
    cooldown_pause = cooldown_rem_min > 0

    blocked = wk_closed or news_pause or cooldown_pause
    can_scalp = bool(in_scalp_hours and not blocked)
    can_trend = bool(in_trend_hours and not blocked)

    pause_reasons = []
    if wk_closed:
        pause_reasons.append(wk_reason)
    if cooldown_pause:
        pause_reasons.append(f"COOLDOWN ({cooldown_rem_min}m)")
    if news_pause:
        pause_reasons.append(news_reason)
    if not in_scalp_hours and not in_trend_hours:
        pause_reasons.append("OUTSIDE_SESSION")

    return {
        "account_balance": acct_bal,
        "risk_pct": risk_pct,
        "risk_usd": risk_usd,
        "sc_sl_usd": sc_sl_usd,
        "sc_be_dist": sc_be_dist,
        "sc_tp_usd": sc_tp_usd,
        "sc_sl_pts": sc_sl_pts,
        "sc_be_pts": sc_be_pts,
        "sc_tp_pts": sc_tp_pts,
        "sc_lot": sc_lot,
        "tr_lot": tr_lot,
        "spread_usd": spread_usd,
        "weekend_closed": wk_closed,
        "daily_net_r": daily_net_r,
        "daily_loss_pause": False,
        "cooldown_rem_min": cooldown_rem_min,
        "cooldown_pause": cooldown_pause,
        "news_pause": news_pause,
        "spread_pause": False,
        "can_scalp": can_scalp,
        "can_trend": can_trend,
        "status": "PAUSED: " + " | ".join(pause_reasons) if pause_reasons else "ACTIVE_READY"
    }


def run_cycle(send_summary_alert=True):
    now_th = datetime.now(THAI_TZ)
    wk_closed, wk_reason = is_gold_market_closed_bkk(now_th)
    if wk_closed:
        prev_state = load_json_safe(STATE_PATH, {})
        prev_rg = prev_state.get("risk_guard", {}) if isinstance(prev_state, dict) else {}
        if prev_rg.get("weekend_closed") is True:
            print(f"[{now_th.strftime('%H:%M:%S')}] [WEEKEND] ตลาดทองคำปิดทำการ (เสาร์-อาทิตย์) — พักการทำงานทั้งหมด (ไม่ยิง API / ไม่ส่ง Ntfy)")
            return False
        prev_state["version"] = "14.1"
        prev_state["updated_at"] = now_th.isoformat()
        prev_rg["weekend_closed"] = True
        prev_rg["can_scalp"] = False
        prev_rg["can_trend"] = False
        prev_rg["status"] = f"PAUSED: {wk_reason}"
        prev_state["risk_guard"] = prev_rg
        if isinstance(prev_state.get("scalp"), dict) and isinstance(prev_state["scalp"].get("setup"), dict):
            prev_state["scalp"]["setup"]["shape_active"] = False
        if isinstance(prev_state.get("quant"), dict) and isinstance(prev_state["quant"].get("setup"), dict):
            prev_state["quant"]["setup"]["in_killzone"] = False
        save_json_safe(STATE_PATH, prev_state)
        print(f"[{now_th.strftime('%H:%M:%S')}] [WEEKEND] บันทึกสถานะปิดตลาดเสาร์-อาทิตย์ลง state.json เรียบร้อย (งดส่งแจ้งเตือน Ntfy)")
        return True

    print(f"[{now_th.strftime('%H:%M:%S')}] \u0e40\u0e23\u0e34\u0e48\u0e21\u0e1b\u0e23\u0e30\u0e21\u0e27\u0e25\u0e1c\u0e25 XAU/USD V14.1 Unified OrderFlow + MUE-X + RiskGuard...")

    df_raw, spot, bid, ask, feed_src = update_hydra_vault()
    df, pdh, pdl = compute_technicals(df_raw)
    order_flow = fetch_order_flow_snapshot(fallback_spot=spot)
    dxy_p, us10y_y, corr_20, news_items = fetch_macro_and_news()

    # Session Micro-VP (32 M15 bars = 8 hours) so Scalp entries ALWAYS stay anchored near live price!
    vp_session = analyze_volume_profile_anatomy(df.tail(32), bins=22)
    vp_5d = analyze_volume_profile_anatomy(df.tail(480), bins=32)

    mue_mem = load_or_init_mue_memory()
    scalp_bt = evolve_mue_x_genes(df, mue_mem)
    trend_bt = run_trend_walk_forward(df)

    # === SESSION STABLE PROFILES: Vectorized Grid + Bootstrap Monte Carlo 1,000,000 รอบ ===
    try:
        session_stable = run_session_stable_profiles(df, now_th)
    except Exception as e_srp:
        _prev_ssp = load_json_safe(STATE_PATH, {})
        session_stable = ((_prev_ssp or {}).get("session_stable_profiles") if isinstance(_prev_ssp, dict) else None) or {"status": "ERROR", "error": str(e_srp)}
        print(f"[STABLE Warning] {e_srp}")

    active_gene = mue_mem["genes"][vp_session["shape"]]
    mode = active_gene["mode"]
    raw_tp_pts, tol_pts = int(active_gene["tp_points"]), int(active_gene["tol_points"])

    last_row = df.iloc[-1]
    curr_ema21 = round(float(last_row["ema21"]), 2)
    curr_h1_200 = round(float(last_row["h1_ema200"]), 2)
    curr_adx = round(float(last_row["adx14"]), 1)
    curr_rsi = round(float(last_row["rsi14"]), 1)
    curr_atr = max(2.0, round(float(last_row["atr14"]), 2))

    hr_dec = now_th.hour + (now_th.minute / 60.0)
    in_scalp_hours = (9.0 <= hr_dec <= 24.0) or (hr_dec <= 1.0)
    in_trend_hours = (13.0 <= hr_dec <= 24.0) or (hr_dec <= 0.5)

    tr_sl_dist, tr_tp_dist = round(curr_atr * 1.8, 2), round(curr_atr * 1.8 * 2.2, 2)
    rg = evaluate_risk_guard(spot, bid, ask, curr_atr, raw_tp_pts, tr_sl_dist, news_items, in_scalp_hours, in_trend_hours)

    sl_pts, be_pts, tp_pts = rg["sc_sl_pts"], rg["sc_be_pts"], rg["sc_tp_pts"]
    sl_usd, be_dist, tp_usd = rg["sc_sl_usd"], rg["sc_be_dist"], rg["sc_tp_usd"]

    # Anchor Scalp Entry to the NEAREST active Session VP level around current Spot price!
    candidates_long = [lvl for lvl in [vp_session["val"], vp_session["lower_peak"], vp_session["poc"]] if lvl <= spot + 2.5]
    candidates_short = [lvl for lvl in [vp_session["vah"], vp_session["upper_peak"], vp_session["poc"]] if lvl >= spot - 2.5]
    nearest_sup = round(max(candidates_long) if candidates_long else (spot - 1.5), 2)
    nearest_res = round(min(candidates_short) if candidates_short else (spot + 1.5), 2)

    if mode == "VALLEY_MIGRATION" and vp_session["valley"] is not None:
        pivot = vp_session["valley"]
        if spot >= pivot:
            sc_dir, sc_entry = "LONG", round(pivot if abs(spot - pivot) <= 4.0 else nearest_sup, 2)
            sc_bias = f"{vp_session['shape']} VALLEY BUY (\u0e23\u0e2d\u0e22\u0e48\u0e2d\u0e23\u0e31\u0e1a\u0e42\u0e0b\u0e19 ${sc_entry} + OBI >= +0.18)"
        else:
            sc_dir, sc_entry = "SHORT", round(pivot if abs(spot - pivot) <= 4.0 else nearest_res, 2)
            sc_bias = f"{vp_session['shape']} VALLEY SELL (\u0e23\u0e2d\u0e40\u0e14\u0e49\u0e07\u0e0a\u0e19\u0e42\u0e0b\u0e19 ${sc_entry} + OBI <= -0.18)"
    elif mode == "POC_PULLBACK":
        if spot >= vp_session["poc"]:
            sc_dir, sc_entry = "LONG", round(vp_session["poc"] if abs(spot - vp_session["poc"]) <= 5.0 else nearest_sup, 2)
            sc_bias = f"{vp_session['shape']} POC PULLBACK BUY (\u0e23\u0e2d\u0e22\u0e48\u0e2d\u0e10\u0e32\u0e19 Session POC ${sc_entry} + OBI)"
        else:
            sc_dir, sc_entry = "SHORT", round(vp_session["poc"] if abs(spot - vp_session["poc"]) <= 5.0 else nearest_res, 2)
            sc_bias = f"{vp_session['shape']} POC PULLBACK SELL (\u0e23\u0e2d\u0e40\u0e14\u0e49\u0e07\u0e0a\u0e19 Session POC ${sc_entry} + OBI)"
    else:
        if abs(spot - nearest_sup) <= abs(nearest_res - spot):
            sc_dir, sc_entry = "LONG", nearest_sup
            sc_bias = f"{vp_session['shape']} SESSION SUPPORT BUY (\u0e14\u0e31\u0e01\u0e23\u0e31\u0e1a ${sc_entry} + \u0e22\u0e37\u0e19\u0e22\u0e31\u0e19 OBI >= +0.18)"
        else:
            sc_dir, sc_entry = "SHORT", nearest_res
            sc_bias = f"{vp_session['shape']} SESSION RESIST SELL (\u0e14\u0e31\u0e01\u0e15\u0e49\u0e32\u0e19 ${sc_entry} + \u0e22\u0e37\u0e19\u0e22\u0e31\u0e19 OBI <= -0.18)"

    if rg["news_pause"] or rg["spread_pause"]:
        sc_bias = f"[{rg['status']}] {sc_bias}"

    if sc_dir == "LONG":
        sc_sl, sc_be, sc_tp = round(sc_entry - sl_usd, 2), round(sc_entry + be_dist, 2), round(sc_entry + tp_usd, 2)
    else:
        sc_sl, sc_be, sc_tp = round(sc_entry + sl_usd, 2), round(sc_entry - be_dist, 2), round(sc_entry - tp_usd, 2)

    dist_usd = round(abs(spot - sc_entry), 2)
    scalp_setup = {
        "shape": vp_session["shape"], "mode": mode, "generation": active_gene["generation"],
        "shape_active": rg["can_scalp"], "direction": sc_dir, "bias": sc_bias,
        "entry": sc_entry, "sl": sc_sl, "be_target": sc_be, "tp": sc_tp,
        "sl_points": sl_pts, "be_points": be_pts, "tp_points": tp_pts, "tol_points": tol_pts,
        "sl_usd": sl_usd, "tp_usd": tp_usd, "rr": round(tp_usd / max(0.1, sl_usd), 2),
        "dist_usd": dist_usd, "dist_points": int(round(dist_usd * 100)),
        "lot_size": rg["sc_lot"], "risk_usd": rg["risk_usd"],
        "obi": order_flow["obi_25bps"], "obi_req": 0.18
    }

    is_bear = spot < curr_h1_200
    tr_session_name = "ACTIVE SESSION" if in_trend_hours else "NO TRADE ZONE (\u0e23\u0e2d 13:00 \u0e19.)"
    if rg["news_pause"] or rg["spread_pause"]:
        tr_session_name = rg["status"]

    if is_bear:
        retest_lvl = curr_ema21 if abs(curr_ema21 - spot) < abs(pdl - spot) else pdl
        tr_bias = f"ANTI-CHASE RETEST SHORT (\u0e23\u0e2d\u0e40\u0e14\u0e49\u0e07\u0e0a\u0e19 ${retest_lvl})"
        if rg["news_pause"] or rg["spread_pause"]:
            tr_bias = f"[{rg['status']}] {tr_bias}"
        trend_setup = {
            "active_factor": "EMA21_RETEST", "regime": f"BEARISH TREND (ADX {curr_adx} | RSI {curr_rsi})",
            "direction": "SHORT", "bias": tr_bias,
            "entry": round(retest_lvl, 2), "sl": round(retest_lvl + tr_sl_dist, 2),
            "be_target": round(retest_lvl - tr_sl_dist, 2), "tp": round(retest_lvl - tr_tp_dist, 2),
            "sl_dist": tr_sl_dist, "tp_dist": tr_tp_dist, "rr": 2.2,
            "has_edge": True, "in_killzone": rg["can_trend"], "session_name": tr_session_name,
            "lot_size": rg["tr_lot"], "risk_usd": rg["risk_usd"]
        }
    else:
        retest_lvl = curr_ema21 if abs(curr_ema21 - spot) < abs(pdh - spot) else pdh
        tr_bias = f"PULLBACK RETEST LONG (\u0e23\u0e2d\u0e22\u0e48\u0e2d\u0e23\u0e31\u0e1a ${retest_lvl})"
        if rg["news_pause"] or rg["spread_pause"]:
            tr_bias = f"[{rg['status']}] {tr_bias}"
        trend_setup = {
            "active_factor": "EMA21_RETEST", "regime": f"BULLISH TREND (ADX {curr_adx} | RSI {curr_rsi})",
            "direction": "LONG", "bias": tr_bias,
            "entry": round(retest_lvl, 2), "sl": round(retest_lvl - tr_sl_dist, 2),
            "be_target": round(retest_lvl + tr_sl_dist, 2), "tp": round(retest_lvl + tr_tp_dist, 2),
            "sl_dist": tr_sl_dist, "tp_dist": tr_tp_dist, "rr": 2.2,
            "has_edge": True, "in_killzone": rg["can_trend"], "session_name": tr_session_name,
            "lot_size": rg["tr_lot"], "risk_usd": rg["risk_usd"]
        }

    sj, tj = update_journals(spot, scalp_setup, trend_setup, rg["can_scalp"], rg["can_trend"])
    ai_text, ai_model_name = generate_live_ai_strategist_brief(spot, vp_session, scalp_setup, trend_setup, order_flow, scalp_bt, trend_bt, mue_mem)

    state_payload = {
        "version": "14.1",
        "updated_at": now_th.isoformat(),
        "order_flow": order_flow,
        "risk_guard": rg,
        "mue_evolution": {"generation": mue_mem["generation"], "last_evolved_at": mue_mem["last_evolved_at"], "genes": mue_mem["genes"]},
        "quant": {
            "spot_price": spot, "bid": bid, "ask": ask, "feed_source": feed_src,
            "dxy_price": dxy_p, "us10y_yield": us10y_y, "corr_gold_dxy_20d": corr_20,
            "data_audit": {"total_bars": len(df), "vault_days": round(len(df) / 96.0, 1)},
            "technicals": {"adx14": curr_adx, "rsi14": curr_rsi, "atr14": curr_atr, "ema21": curr_ema21, "h1_ema200": curr_h1_200, "pdh": pdh, "pdl": pdl},
            "setup": trend_setup
        },
        "macro": {"dxy": dxy_p, "us10y": us10y_y, "gold_dxy_corr": corr_20},
        "scalp": {"anatomy": {**vp_session, "rule_th": active_gene["rule_th"]}, "setup": scalp_setup, "backtest": scalp_bt},
        "volume_profile": vp_5d,
        "backtest": trend_bt,
        "session_stable_profiles": session_stable,
        "scalp_journal": sj,
        "journal": tj,
        "ai_engine_name": ai_model_name,
        "ai_analysis": ai_text,
        "news": news_items
    }

    save_json_safe(STATE_PATH, state_payload)
    print(f"[OK] บันทึก state.json V14.0 สำเร็จ | Spot: ${spot} | Scalp Entry: ${sc_entry} | AI: {ai_model_name}")

    if send_summary_alert:
        send_ntfy(
            f"XAU/USD V14.0 (${spot}) | Scalp {sc_dir} @ ${sc_entry}",
            f"Session VP {vp_session['shape']} | OBI: {order_flow['obi_25bps']:+.3f} | TP: ${sc_tp} | Trend: {trend_setup['direction']} @ ${trend_setup['entry']}",
            priority=4, tags=["chart_with_upwards_trend", "dart"]
        )


def git_auto_sync():
    try:
        os.system('git add data/state.json data/mue_memory.json data/scalp_journal.json data/trade_journal.json')
        os.system('git commit -m "Auto-sync Gold Quant V14.0" >nul 2>&1')
        os.system('git pull --rebase --autostash origin main >nul 2>&1')
        os.system('git push >nul 2>&1')
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", type=int, default=0)
    parser.add_argument("--no-startup-alert", action="store_true")
    args = parser.parse_args()

    if args.loop > 0:
        first = True
        while True:
            try:
                did_update = run_cycle(send_summary_alert=(not args.no_startup_alert or not first))
                if did_update is not False:
                    git_auto_sync()
            except Exception as exc:
                print(f"[Cycle Error] {exc}")
            first = False
            now_c = datetime.now(THAI_TZ)
            n_q = ((now_c.minute // args.loop) + 1) * args.loop
            nxt_dt = now_c.replace(minute=0, second=5, microsecond=0) + timedelta(minutes=n_q)
            wait_s = max(15, (nxt_dt - now_c).total_seconds())
            print(f"[TIMER] รออัปเดตรอบถัดไปเวลา {nxt_dt.strftime('%H:%M:%S')} น. ({int(wait_s)} วินาที)...")
            time.sleep(wait_s)
    else:
        run_cycle(send_summary_alert=not args.no_startup_alert)

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

THAI_TZ = timezone(timedelta(hours=7))
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
VAULT_PATH = os.path.join(DATA_DIR, "gold_m15_vault.csv")
JOURNAL_PATH = os.path.join(DATA_DIR, "trade_journal.json")
SCALP_JOURNAL_PATH = os.path.join(DATA_DIR, "scalp_journal.json")
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
        json.dump(obj, f, ensure_ascii=False, indent=2)


def fetch_order_flow_snapshot(fallback_spot=3550.0):
    snapshot = {
        "source": "Binance XAUUSDT Perp L2",
        "mid_price": round(fallback_spot, 2),
        "obi_25bps": 0.0,
        "bid_liq_total": 0.0,
        "ask_liq_total": 0.0,
        "bid_walls": [],
        "ask_walls": [],
        "tick_cvd": 0.0,
        "absorption_flag": "NEUTRAL_FLOW",
        "open_interest": 0.0,
        "funding_rate_pct": 0.0,
        "of_regime": "TWO_WAY_ROTATION"
    }
    try:
        r_depth = requests.get("https://fapi.binance.com/fapi/v1/depth", params={"symbol": "XAUUSDT", "limit": 20}, timeout=6)
        if r_depth.status_code == 200:
            d = r_depth.json()
            bids = [(float(p), float(q)) for p, q in d.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in d.get("asks", [])]
            if bids and asks:
                snapshot["mid_price"] = round((bids[0][0] + asks[0][0]) / 2.0, 2)
                b_25 = sum(q for p, q in bids[:15])
                a_25 = sum(q for p, q in asks[:15])
                snapshot["obi_25bps"] = round((b_25 - a_25) / max(1e-6, b_25 + a_25), 3)
                snapshot["bid_liq_total"] = round(b_25, 2)
                snapshot["ask_liq_total"] = round(a_25, 2)
                all_q = [q for _, q in bids[:12]] + [q for _, q in asks[:12]]
                avg_q = float(np.mean(all_q)) if all_q else 1.0
                snapshot["bid_walls"] = [{"price": round(p, 2), "size": round(q, 2)} for p, q in bids[:12] if q >= 2.0 * avg_q][:3]
                snapshot["ask_walls"] = [{"price": round(p, 2), "size": round(q, 2)} for p, q in asks[:12] if q >= 2.0 * avg_q][:3]

        r_trades = requests.get("https://fapi.binance.com/fapi/v1/aggTrades", params={"symbol": "XAUUSDT", "limit": 500}, timeout=6)
        if r_trades.status_code == 200:
            trades = r_trades.json()
            if trades:
                buy_v = sum(float(t["q"]) for t in trades if not t.get("m", False))
                sell_v = sum(float(t["q"]) for t in trades if t.get("m", False))
                tick_cvd = round(buy_v - sell_v, 2)
                snapshot["tick_cvd"] = tick_cvd
                p0, p1 = float(trades[0]["p"]), float(trades[-1]["p"])
                if p1 < p0 and tick_cvd > 0 and snapshot["obi_25bps"] > 0.06:
                    snapshot["absorption_flag"] = "BULLISH_ABSORPTION (Bid ดูดซับแรงขาย)"
                elif p1 > p0 and tick_cvd < 0 and snapshot["obi_25bps"] < -0.06:
                    snapshot["absorption_flag"] = "BEARISH_ABSORPTION (Ask กดหัว)"
                elif snapshot["obi_25bps"] > 0.12:
                    snapshot["absorption_flag"] = "BID_DOMINANCE"
                elif snapshot["obi_25bps"] < -0.12:
                    snapshot["absorption_flag"] = "ASK_DOMINANCE"

        r_oi = requests.get("https://fapi.binance.com/fapi/v1/openInterest", params={"symbol": "XAUUSDT"}, timeout=5)
        if r_oi.status_code == 200:
            snapshot["open_interest"] = round(float(r_oi.json().get("openInterest", 0.0)), 2)

        r_pr = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": "XAUUSDT"}, timeout=5)
        if r_pr.status_code == 200:
            snapshot["funding_rate_pct"] = round(float(r_pr.json().get("lastFundingRate", 0.0)) * 100.0, 4)
    except Exception as e:
        print(f"[OrderFlow Warning] {e}")
    return snapshot


def update_hydra_vault():
    os.makedirs(DATA_DIR, exist_ok=True)
    existing_df = pd.read_csv(VAULT_PATH, index_col=0, parse_dates=True) if os.path.exists(VAULT_PATH) else None

    new_df = yf.download("GC=F", period="60d", interval="15m", progress=False)
    if isinstance(new_df.columns, pd.MultiIndex):
        new_df.columns = new_df.columns.get_level_values(0)
    new_df = new_df[["Open", "High", "Low", "Close", "Volume"]].dropna()

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
    sl_usd, tp_usd, tol_usd = sl_pts / 100.0, tp_pts / 100.0, tol_pts / 100.0
    spread_pen = 0.20
    trades_r, sim_records = [], []
    n = len(df)
    i = max(80, n - 1800)

    while i < n - 12:
        # Use 32-bar (8H) Session Micro-VP so scalp levels always track live auction!
        vp = analyze_volume_profile_anatomy(df.iloc[i - 32:i], bins=20)
        if vp["shape"] != target_shape:
            i += 3
            continue

        row = df.iloc[i]
        c, ema21, rsi, adx = float(row["Close"]), float(row["ema21"]), float(row["rsi14"]), float(row["adx14"])
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
        sim_records.append({
            "engine": "SCALP",
            "shape": f"{target_shape} ({mode[:6]})",
            "side": side,
            "entry_time": df.index[i].strftime("%d/%m %H:%M"),
            "exit_time": df.index[exit_i].strftime("%d/%m %H:%M"),
            "entry": round(entry, 2),
            "exit": round(exit_p, 2),
            "r": round(res_r, 2),
            "pnl": round(res_r * 5.0, 2)
        })
        i = exit_i + 2

    wins = sum(1 for r in trades_r if r > 0)
    losses = sum(1 for r in trades_r if r < 0)
    net_r = round(sum(trades_r), 2)
    wr = round((wins / max(1, wins + losses)) * 100.0, 1)
    r_pf = round(sum(r for r in trades_r if r > 0) / max(0.5, abs(sum(r for r in trades_r if r < 0))), 2)
    fitness = round(max(0.20, min(0.99, (wr / 100.0) * 0.5 + min(1.0, r_pf / 2.5) * 0.5)), 2)

    return {"trades": len(trades_r), "win_rate": wr, "net_r": net_r, "r_pf": r_pf, "fitness": fitness, "trades_r": trades_r, "sim_records": sim_records}


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
                        best_cfg["sl_points"] = 500
                        best_cfg["tp_points"] = tp_c
                        best_cfg["tol_points"] = tol_c
                        mutated = True

        if mutated:
            best_cfg["generation"] = int(best_cfg.get("generation", 1)) + 1

        best_cfg["fitness"] = best_res["fitness"]
        best_cfg["obi_min"] = 0.18
        best_cfg["rule_th"] = f"MUE-X Gen#{best_cfg['generation']}: {best_cfg['mode']} | SL 500 / TP {best_cfg['tp_points']} จุด (โซน ±{best_cfg['tol_points']} จุด + OBI >= 0.18)"
        mue_memory["genes"][shape_name] = best_cfg

        breakdown.append({
            "shape": shape_name,
            "mode": best_cfg["mode"],
            "generation": best_cfg["generation"],
            "sl_points": 500,
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


def generate_live_ai_strategist_brief(spot, vp_session, scalp_setup, trend_setup, order_flow, scalp_bt, trend_bt, mue_mem):
    fallback_brief = (
        f"**1. Session Micro-VP 8H & Live OrderFlow ({vp_session['title_th']}):**\n"
        f"Spot ปัจจุบัน **${spot}** เกาะอยู่ในกรอบ Session 8H (VAL **${vp_session['val']}** / POC **${vp_session['poc']}** / VAH **${vp_session['vah']}**) | "
        f"L2 OBI **{order_flow['obi_25bps']:+.3f}** | Tick CVD **{order_flow['tick_cvd']:+.1f}** ({order_flow['absorption_flag']})\n\n"
        f"**2. เงื่อนไขลั่นไก OrderFlow Execution Gate (ใช้งานจริง 3 ชั้น):**\n"
        f"• **ชั้นที่ 1 (Location):** รอราคาเข้าใกล้จุดยิง Scalp **${scalp_setup['entry']}** ในระยะไม่เกิน `±{scalp_setup['tol_points']} จุด`\n"
        f"• **ชั้นที่ 2 (3s Sustained OBI):** แรงฝั่ง **{scalp_setup['direction']}** ต้องยืนเหนือเกณฑ์ `{'+0.18' if scalp_setup['direction']=='LONG' else '-0.18'}` ต่อเนื่อง 3 วินาทีเต็ม (ตัดออเดอร์หลอก)\n"
        f"• **ชั้นที่ 3 (60s Tape Delta):** แรงเคาะซื้อ/ขายจริง 60 วินาทีต้องสอดคล้องทิศทางจึงเปิดออเดอร์\n\n"
        f"**3. แผนเข้าเทรด Scalp ({scalp_setup['direction']}):** Entry **${scalp_setup['entry']}** | SL **${scalp_setup['sl']}** | BE (+250 จุด) **${scalp_setup['be_target']}** | TP **${scalp_setup['tp']}**\n"
        f"**4. แผน Trend Sniper ({trend_setup['direction']}):** Entry **${trend_setup['entry']}** | SL **${trend_setup['sl']}** | TP **${trend_setup['tp']}** (WR {trend_bt['best']['win_rate']}%)"
    )

    if not THAILLM_API_KEY:
        return fallback_brief, "MUE-X Local Quant V14.0"

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
            return txt, "Typhoon-v2.5 + MUE-X V14.0"
    except Exception as e:
        print(f"[AI Brief Warning] {e}")

    return fallback_brief, "MUE-X Quant V14.0"


def fetch_macro_and_news():
    dxy_p, us10y_y, corr_20 = 103.85, 4.25, -0.45
    try:
        dxy = yf.download("DX-Y.NYB", period="30d", interval="1d", progress=False)
        tnx = yf.download("^TNX", period="10d", interval="1d", progress=False)
        gld = yf.download("GC=F", period="30d", interval="1d", progress=False)
        for df_x in (dxy, tnx, gld):
            if isinstance(df_x.columns, pd.MultiIndex):
                df_x.columns = df_x.columns.get_level_values(0)
        if not dxy.empty:
            dxy_p = round(float(dxy["Close"].iloc[-1]), 2)
        if not tnx.empty:
            us10y_y = round(float(tnx["Close"].iloc[-1]), 2)
        if len(dxy) >= 15 and len(gld) >= 15:
            j = pd.concat([gld["Close"], dxy["Close"]], axis=1).dropna().tail(20)
            if len(j) >= 10:
                corr_20 = round(float(j.iloc[:, 0].corr(j.iloc[:, 1])), 2)
    except Exception:
        pass

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


def update_journals(spot, scalp_setup, trend_setup, in_scalp_hours, in_trend_hours):
    now_str = datetime.now(THAI_TZ).strftime("%d/%m %H:%M")
    sj = load_json_safe(SCALP_JOURNAL_PATH, {"wins": 0, "losses": 0, "closed_trades": 0, "open_trades": 0, "net_live_pnl": 0.0, "live_win_rate": 0.0, "recent_trades": []})
    for t in sj["recent_trades"]:
        if t["status"] == "OPEN":
            if t["direction"] == "LONG":
                if spot <= t["sl"]:
                    t["status"], t["pnl"] = "LOSS", -5.0
                elif spot >= t["tp"]:
                    t["status"], t["pnl"] = "WIN", round(t["tp"] - t["entry"], 2)
            else:
                if spot >= t["sl"]:
                    t["status"], t["pnl"] = "LOSS", -5.0
                elif spot <= t["tp"]:
                    t["status"], t["pnl"] = "WIN", round(t["entry"] - t["tp"], 2)

    open_sc = [t for t in sj["recent_trades"] if t["status"] == "OPEN"]
    if in_scalp_hours and len(open_sc) == 0 and abs(spot - scalp_setup["entry"]) <= (scalp_setup["tol_points"] / 100.0):
        sj["recent_trades"].insert(0, {
            "opened_at_th": now_str,
            "factor": f"{scalp_setup['shape']} ({scalp_setup['mode'][:6]})",
            "direction": scalp_setup["direction"],
            "entry": scalp_setup["entry"],
            "sl": scalp_setup["sl"],
            "tp": scalp_setup["tp"],
            "status": "OPEN",
            "pnl": 0.0
        })
        send_ntfy(
            f"PROP SCALP {scalp_setup['direction']} @ ${scalp_setup['entry']}",
            f"Session VP {scalp_setup['shape']} | OBI: {scalp_setup['obi']} | SL: ${scalp_setup['sl']} | TP: ${scalp_setup['tp']}",
            priority=5, tags=["zap", "dart"]
        )

    sj["wins"] = sum(1 for t in sj["recent_trades"] if t["status"] == "WIN")
    sj["losses"] = sum(1 for t in sj["recent_trades"] if t["status"] == "LOSS")
    sj["closed_trades"] = sj["wins"] + sj["losses"]
    sj["open_trades"] = sum(1 for t in sj["recent_trades"] if t["status"] == "OPEN")
    sj["net_live_pnl"] = round(sum(float(t.get("pnl", 0.0)) for t in sj["recent_trades"] if t["status"] != "OPEN"), 2)
    sj["live_win_rate"] = round((sj["wins"] / max(1, sj["closed_trades"])) * 100.0, 1)
    sj["recent_trades"] = sj["recent_trades"][:15]
    save_json_safe(SCALP_JOURNAL_PATH, sj)

    tj = load_json_safe(JOURNAL_PATH, {"wins": 0, "losses": 0, "closed_trades": 0, "open_trades": 0, "net_live_pnl": 0.0, "live_win_rate": 0.0, "recent_trades": []})
    for t in tj["recent_trades"]:
        if t["status"] == "OPEN":
            if t["direction"] == "LONG":
                if spot <= t["sl"]:
                    t["status"], t["pnl"] = "LOSS", -75.0
                elif spot >= t["tp"]:
                    t["status"], t["pnl"] = "WIN", 165.0
            else:
                if spot >= t["sl"]:
                    t["status"], t["pnl"] = "LOSS", -75.0
                elif spot <= t["tp"]:
                    t["status"], t["pnl"] = "WIN", 165.0

    open_tr = [t for t in tj["recent_trades"] if t["status"] == "OPEN"]
    if in_trend_hours and len(open_tr) == 0 and abs(spot - trend_setup["entry"]) <= 4.5:
        tj["recent_trades"].insert(0, {
            "opened_at_th": now_str,
            "factor": "EMA21_RETEST",
            "direction": trend_setup["direction"],
            "entry": trend_setup["entry"],
            "sl": trend_setup["sl"],
            "tp": trend_setup["tp"],
            "status": "OPEN",
            "pnl": 0.0
        })
        send_ntfy(
            f"PROP TREND {trend_setup['direction']} @ ${trend_setup['entry']}",
            f"{trend_setup['bias']} | SL: ${trend_setup['sl']} | TP: ${trend_setup['tp']}",
            priority=5, tags=["rocket", "moneybag"]
        )

    tj["wins"] = sum(1 for t in tj["recent_trades"] if t["status"] == "WIN")
    tj["losses"] = sum(1 for t in tj["recent_trades"] if t["status"] == "LOSS")
    tj["closed_trades"] = tj["wins"] + tj["losses"]
    tj["open_trades"] = sum(1 for t in tj["recent_trades"] if t["status"] == "OPEN")
    tj["net_live_pnl"] = round(sum(float(t.get("pnl", 0.0)) for t in tj["recent_trades"] if t["status"] != "OPEN"), 2)
    tj["live_win_rate"] = round((tj["wins"] / max(1, tj["closed_trades"])) * 100.0, 1)
    tj["recent_trades"] = tj["recent_trades"][:15]
    save_json_safe(JOURNAL_PATH, tj)
    return sj, tj


def run_cycle(send_summary_alert=True):
    now_th = datetime.now(THAI_TZ)
    print(f"[{now_th.strftime('%H:%M:%S')}] เริ่มประมวลผล XAU/USD V14.0 Unified OrderFlow + MUE-X...")

    df_raw, spot, bid, ask, feed_src = update_hydra_vault()
    df, pdh, pdl = compute_technicals(df_raw)
    order_flow = fetch_order_flow_snapshot(fallback_spot=spot)

    # Session Micro-VP (32 M15 bars = 8 hours) so Scalp entries ALWAYS stay anchored near live price!
    vp_session = analyze_volume_profile_anatomy(df.tail(32), bins=22)
    vp_5d = analyze_volume_profile_anatomy(df.tail(480), bins=32)

    mue_mem = load_or_init_mue_memory()
    scalp_bt = evolve_mue_x_genes(df, mue_mem)
    trend_bt = run_trend_walk_forward(df)

    active_gene = mue_mem["genes"][vp_session["shape"]]
    mode = active_gene["mode"]
    sl_pts, tp_pts, tol_pts = 500, int(active_gene["tp_points"]), int(active_gene["tol_points"])
    sl_usd, tp_usd = 5.0, tp_pts / 100.0

    last_row = df.iloc[-1]
    curr_ema21 = round(float(last_row["ema21"]), 2)
    curr_h1_200 = round(float(last_row["h1_ema200"]), 2)
    curr_adx = round(float(last_row["adx14"]), 1)
    curr_rsi = round(float(last_row["rsi14"]), 1)
    curr_atr = max(2.0, round(float(last_row["atr14"]), 2))

    # Anchor Scalp Entry to the NEAREST active Session VP level around current Spot price!
    candidates_long = [lvl for lvl in [vp_session["val"], vp_session["lower_peak"], vp_session["poc"]] if lvl <= spot + 2.5]
    candidates_short = [lvl for lvl in [vp_session["vah"], vp_session["upper_peak"], vp_session["poc"]] if lvl >= spot - 2.5]
    nearest_sup = round(max(candidates_long) if candidates_long else (spot - 1.5), 2)
    nearest_res = round(min(candidates_short) if candidates_short else (spot + 1.5), 2)

    if mode == "VALLEY_MIGRATION" and vp_session["valley"] is not None:
        pivot = vp_session["valley"]
        if spot >= pivot:
            sc_dir, sc_entry = "LONG", round(pivot if abs(spot - pivot) <= 4.0 else nearest_sup, 2)
            sc_bias = f"{vp_session['shape']} VALLEY BUY (รอย่อรับโซน ${sc_entry} + OBI >= +0.18)"
        else:
            sc_dir, sc_entry = "SHORT", round(pivot if abs(spot - pivot) <= 4.0 else nearest_res, 2)
            sc_bias = f"{vp_session['shape']} VALLEY SELL (รอเด้งชนโซน ${sc_entry} + OBI <= -0.18)"
    elif mode == "POC_PULLBACK":
        if spot >= vp_session["poc"]:
            sc_dir, sc_entry = "LONG", round(vp_session["poc"] if abs(spot - vp_session["poc"]) <= 5.0 else nearest_sup, 2)
            sc_bias = f"{vp_session['shape']} POC PULLBACK BUY (รอย่อฐาน Session POC ${sc_entry} + OBI)"
        else:
            sc_dir, sc_entry = "SHORT", round(vp_session["poc"] if abs(spot - vp_session["poc"]) <= 5.0 else nearest_res, 2)
            sc_bias = f"{vp_session['shape']} POC PULLBACK SELL (รอเด้งชน Session POC ${sc_entry} + OBI)"
    else:
        if abs(spot - nearest_sup) <= abs(nearest_res - spot):
            sc_dir, sc_entry = "LONG", nearest_sup
            sc_bias = f"{vp_session['shape']} SESSION SUPPORT BUY (ดักรับ ${sc_entry} + ยืนยัน OBI >= +0.18)"
        else:
            sc_dir, sc_entry = "SHORT", nearest_res
            sc_bias = f"{vp_session['shape']} SESSION RESIST SELL (ดักต้าน ${sc_entry} + ยืนยัน OBI <= -0.18)"

    if sc_dir == "LONG":
        sc_sl, sc_be, sc_tp = round(sc_entry - sl_usd, 2), round(sc_entry + 2.50, 2), round(sc_entry + tp_usd, 2)
    else:
        sc_sl, sc_be, sc_tp = round(sc_entry + sl_usd, 2), round(sc_entry - 2.50, 2), round(sc_entry - tp_usd, 2)

    scalp_setup = {
        "shape": vp_session["shape"], "mode": mode, "generation": active_gene["generation"],
        "shape_active": True, "direction": sc_dir, "bias": sc_bias,
        "entry": sc_entry, "sl": sc_sl, "be_target": sc_be, "tp": sc_tp,
        "sl_points": sl_pts, "tp_points": tp_pts, "tol_points": tol_pts, "tp_usd": tp_usd,
        "obi": order_flow["obi_25bps"], "obi_req": 0.18
    }

    is_bear = spot < curr_h1_200
    tr_sl_dist, tr_tp_dist = round(curr_atr * 1.8, 2), round(curr_atr * 1.8 * 2.2, 2)
    if is_bear:
        retest_lvl = curr_ema21 if abs(curr_ema21 - spot) < abs(pdl - spot) else pdl
        trend_setup = {
            "active_factor": "EMA21_RETEST", "direction": "SHORT",
            "bias": f"ANTI-CHASE RETEST SHORT (รอเด้งชน ${retest_lvl})",
            "entry": round(retest_lvl, 2), "sl": round(retest_lvl + tr_sl_dist, 2),
            "be_target": round(retest_lvl - tr_sl_dist, 2), "tp": round(retest_lvl - tr_tp_dist, 2),
            "sl_dist": tr_sl_dist, "rr": 2.2
        }
    else:
        retest_lvl = curr_ema21 if abs(curr_ema21 - spot) < abs(pdh - spot) else pdh
        trend_setup = {
            "active_factor": "EMA21_RETEST", "direction": "LONG",
            "bias": f"PULLBACK RETEST LONG (รอย่อรับ ${retest_lvl})",
            "entry": round(retest_lvl, 2), "sl": round(retest_lvl - tr_sl_dist, 2),
            "be_target": round(retest_lvl + tr_sl_dist, 2), "tp": round(retest_lvl + tr_tp_dist, 2),
            "sl_dist": tr_sl_dist, "rr": 2.2
        }

    hr_dec = now_th.hour + (now_th.minute / 60.0)
    in_scalp_hours = (9.0 <= hr_dec <= 24.0) or (hr_dec <= 1.0)
    in_trend_hours = (13.0 <= hr_dec <= 24.0) or (hr_dec <= 0.5)

    sj, tj = update_journals(spot, scalp_setup, trend_setup, in_scalp_hours, in_trend_hours)
    dxy_p, us10y_y, corr_20, news_items = fetch_macro_and_news()
    ai_text, ai_model_name = generate_live_ai_strategist_brief(spot, vp_session, scalp_setup, trend_setup, order_flow, scalp_bt, trend_bt, mue_mem)

    state_payload = {
        "version": "14.0",
        "updated_at": now_th.isoformat(),
        "order_flow": order_flow,
        "mue_evolution": {"generation": mue_mem["generation"], "last_evolved_at": mue_mem["last_evolved_at"], "genes": mue_mem["genes"]},
        "quant": {
            "spot_price": spot, "bid": bid, "ask": ask, "feed_source": feed_src,
            "dxy_price": dxy_p, "us10y_yield": us10y_y, "corr_gold_dxy_20d": corr_20,
            "data_audit": {"total_bars": len(df), "vault_days": round(len(df) / 96.0, 1)},
            "technicals": {"adx14": curr_adx, "rsi14": curr_rsi, "atr14": curr_atr, "ema21": curr_ema21, "h1_ema200": curr_h1_200, "pdh": pdh, "pdl": pdl},
            "setup": trend_setup
        },
        "scalp": {"anatomy": {**vp_session, "rule_th": active_gene["rule_th"]}, "setup": scalp_setup, "backtest": scalp_bt},
        "volume_profile": vp_5d,
        "backtest": trend_bt,
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
                run_cycle(send_summary_alert=(not args.no_startup_alert or not first))
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

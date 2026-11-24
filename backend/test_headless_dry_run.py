#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dry-run test ของ Headless Cloud Executor (backend/engine.py) — ไม่แตะข้อมูลจริง/ไม่ยิง network

รัน:  python backend/test_headless_dry_run.py
วิธี: monkeypatch requests.get/requests.patch + send_ntfy ในหน่วยความจำเท่านั้น (ไม่เขียน Gist จริง)
ครอบคลุม: Heartbeat Handover (รวมกรณีหน้า Mirror), price alignment, ปิด TP/SL/TIME_EXIT_12B/Auto-BE,
          LIMIT fill/cancel, เปิดไม้ใหม่ตามสูตรผู้ชนะ + guard ครบ, Daily rollover (dedupe), EOD report
"""
import copy
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import engine  # noqa: E402

THAI = timezone(timedelta(hours=7))
NOW = datetime(2026, 11, 24, 16, 0, 0, tzinfo=THAI)          # 16:00 BKK = ช่วง LONDON
NOW_EOD = datetime(2026, 11, 25, 5, 5, 0, tzinfo=THAI)       # 05:05 BKK = หน้าต่างรายงานปิดวัน

PASSED, FAILED = [], []
def check(name, cond, extra=""):
    (PASSED if cond else FAILED).append(name)
    print(("PASS  " if cond else "FAIL  ") + name + (("  | " + str(extra)) if extra else ""))

# ---------- dry-run stubs (no network / no real gist) ----------
_orig_requests_get = engine.requests.get
_orig_requests_patch = engine.requests.patch
_alerts = []
_gist_pushes = []

class _FakeResp:
    def __init__(self, status_code=200, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}
    def json(self):
        return self._payload

def _fake_patch(url, headers=None, json=None, timeout=None):
    _gist_pushes.append(json)
    return _FakeResp(200)

engine.requests.patch = _fake_patch
engine.send_ntfy = lambda title, message, priority=4, tags=None: _alerts.append({"title": title, "message": message})
engine.GIST_TOKEN = "TEST_TOKEN_FOR_DRYRUN"
engine.THAILLM_API_KEY = ""  # บังคับใช้ระบบสรุปในตัว — ไม่ยิง ThaiLLM ในเทสต์


def _last_push_payload():
    if not _gist_pushes:
        return None
    content = _gist_pushes[-1]["files"][engine.GIST_FILENAME]["content"]
    return json.loads(content)


def base_snapshot(browser_age_sec=600, auto_arm=True, active=None, cooldown_until=0, browser_role="MASTER"):
    now_ms = int(NOW.timestamp() * 1000)
    return {
        "revision": 1, "schema": 2,
        "masterConfig": {
            "mt5Offset": 7.51, "mt5Locked": True, "lockedSpotOffset": 7.51,
            "spreadBuffer": 47, "mt5BufferPts": 47, "maxSpreadPts": 65,
            "accountBalance": 1000, "riskPct": 1.0, "lotMode": "auto_prop",
            "autoBeEnabled": True, "ghToken": "SECRET_TOKEN_X", "thaillmApiKey": "SECRET_LLM_X",
        },
        "orderState": {
            "currentDate": NOW.strftime("%Y-%m-%d"), "autoArm": auto_arm,
            "activeOrder": active, "todayClosed": [], "learningArchive": [],
            "cooldownUnlockAt": cooldown_until, "cooldownReason": "LOSS" if cooldown_until else "",
            "sessionStart": now_ms - 3_600_000,
        },
        "aiBrain": {"sessionStats": {}, "formulaWeights": {}, "learnedInsights": []},
        "browserLastSeenMs": now_ms - browser_age_sec * 1000,
        "browserRole": browser_role,
        "livePriceSpace": {"spotMid": 3584.0, "spotBasisOffset": 7.49, "mt5Offset": 7.51, "at": now_ms},
    }


def open_order(direction="LONG", entry=3580.0, sl=3575.0, tp=3595.0, lot=0.02, be_rule="NO_BE",
               be_info=None, created_ago_min=10, mode="OPEN", origin="AUTO"):
    created_ms = int(time.time() * 1000) - created_ago_min * 60 * 1000
    if be_info is None:
        be_info = engine.headless_be_plan(entry, sl, tp, direction, be_rule)
    return {
        "id": "ORDT-" + direction, "createdTs": created_ms, "planType": "CLOUD-LONDON",
        "formula": "LIQUIDITY_SWEEP", "dateStr": NOW.strftime("%Y-%m-%d"), "openTime": "15:45:00",
        "mode": mode, "origin": origin, "dir": direction,
        "shape": "CLOUD-LONDON (LIQUIDITY_SWEEP)",
        "entry": entry, "sl": sl, "tp": tp, "slPts": int(abs(entry - sl) * 100), "tpPts": int(abs(tp - entry) * 100),
        "beRule": be_rule, "beInfo": be_info, "beTriggered": False,
        "lot": lot, "riskUsd": round(lot * abs(entry - sl) * 100, 2),
    }


WINNER = {"formula": "LIQUIDITY_SWEEP", "sl_atr_mult": 0.75, "rr": 2.0, "be_rule": "BE_AT_1R",
          "stability_score": 0.4587, "p05_worst_r": 6.41, "win_rate_pct": 55.6, "trades": 36}
WEAK_WINNER = dict(WINNER, formula="VP_EDGE_FADE", stability_score=-0.0937)


def session_stable(winner):
    return {"status": "OK", "sessions": {"LONDON": {"winner": winner}, "LATE_NY": {"winner": winner}}}


def run_exec(snap, bid=3584.0, ask=3584.6, atr=5.0, ss=None, scalp=None, news=None, now_th=NOW):
    _alerts.clear()
    return engine.run_headless_cloud_executor(
        snap, (bid + ask) / 2.0, bid, ask, atr,
        ss if ss is not None else session_stable(WINNER),
        scalp if scalp is not None else {"direction": "LONG", "entry": 3585.0, "sl": 3580.0, "tp": 3595.0},
        news or [], now_th)


# ============================== TESTS ==============================
print("=== 1) Heartbeat handover / price alignment ===")
# heartbeat ต้องอิงเวลาจริง (is_browser_online ใช้เวลาปัจจุบันของเครื่อง ไม่ใช่ NOW ของเทสต์)
def fresh_ms(seconds_ago):
    return int(time.time() * 1000) - seconds_ago * 1000

snap = base_snapshot(browser_age_sec=60)
snap["browserLastSeenMs"] = fresh_ms(60)  # อายุ 60 วิจริง < 180 → ผู้ใช้เปิดเว็บอยู่
_alerts.clear(); _gist_pushes.clear()
changed = run_exec(snap)
check("ผู้ใช้เปิดเว็บ (heartbeat 60 วิ) → engine ข้ามการเปิด/ปิดออเดอร์", changed is False and not _gist_pushes and not _alerts)

check("is_browser_online: 179 วิ → ออนไลน์", engine.is_browser_online(base_snapshot(179), now_ms=int(NOW.timestamp() * 1000)) is True)
check("is_browser_online: 181 วิ → ปิดเว็บ", engine.is_browser_online(base_snapshot(181), now_ms=int(NOW.timestamp() * 1000)) is False)
check("is_browser_online: ไม่มี heartbeat → ปิดเว็บ", engine.is_browser_online({"masterConfig": {}}) is False)
snap_mir = base_snapshot(browser_role="MIRROR"); snap_mir["browserLastSeenMs"] = fresh_ms(10)
check("is_browser_online: หน้า MIRROR เต้นสดอยู่ → ไม่กัน Cloud", engine.is_browser_online(snap_mir) is False)
snap_mst = base_snapshot(); snap_mst["browserLastSeenMs"] = fresh_ms(10)
check("is_browser_online: หน้า MASTER เต้นสดอยู่ → ออนไลน์", engine.is_browser_online(snap_mst) is True)

check("price delta = mt5Offset - basis (7.51 - 7.49)", engine.resolve_price_delta(base_snapshot()) == 0.02)
snap_nb = base_snapshot(); snap_nb["livePriceSpace"] = {}
check("ไม่มี basis → delta 0 (กันบวกซ้ำ)", engine.resolve_price_delta(snap_nb) == 0.0)
snap_cl = base_snapshot(); snap_cl["masterConfig"]["mt5Offset"] = 20.0; snap_cl["masterConfig"]["lockedSpotOffset"] = 20.0
check("delta เกินเพดาน → clamp ±5", engine.resolve_price_delta(snap_cl) == 5.0)

print("=== 2) ปิดออเดอร์: TP / SL / TIME_EXIT_12B / Auto-BE ===")
# TP hit (WIN)
snap = base_snapshot(active=open_order("LONG", 3580.0, 3575.0, 3585.5, 0.02))
_gist_pushes.clear()
changed = run_exec(snap, bid=3585.6, ask=3585.8)   # adj_bid 3585.62 >= tp 3585.50
os_ = snap["orderState"]
check("TP ชน → ปิดไม้ทันที", changed is True and os_["activeOrder"] is None)
rec = os_["todayClosed"][0] if os_["todayClosed"] else {}
check("PnL ถูกต้อง (WIN +$11.00 / +1.10R)", rec.get("outcome") == "WIN" and rec.get("pnlUsd") == 11.0 and rec.get("pnlR") == 1.1, str(rec))
check("บันทึกเข้า todayTrades + journalTrades (learningArchive)", len(os_["todayClosed"]) == 1 and len(os_["learningArchive"]) == 1 and os_["learningArchive"][0].get("tradeId") == rec.get("tradeId"))
check("cooldown ชนะ = จบแท่ง M15 + 15 นาที (สูตรตรง)", os_["cooldownUnlockAt"] == engine.compute_m15_cooldown_unlock_at(os_["lastCloseTs"], True) and os_["cooldownReason"] == "WIN")
check("ntfy แจ้งปิดไม้ + push Gist 1 ครั้ง", len(_alerts) == 1 and "ชน TP" in _alerts[0]["title"] and len(_gist_pushes) == 1)
check("ตั้งธง cloudExecutorActive + สรุป action สำหรับแบนเนอร์หน้าเว็บ", snap.get("cloudExecutorActive") is True and isinstance(snap.get("cloudExecutorLastActionMs"), int) and "ปิดไม้" in str(snap.get("cloudExecutorLastAction")))
pl = _last_push_payload()
check("snapshot ที่ push: token/api key ถูกตัดออกเสมอ", pl["masterConfig"]["ghToken"] == "" and pl["masterConfig"]["thaillmApiKey"] == "" and pl["cloudExecutor"]["mode"] == "HEADLESS")
check("aiBrain อัปเดตสถิติ LONDON + insight", snap["aiBrain"]["sessionStats"].get("LONDON", {}).get("wins") == 1 and len(snap["aiBrain"]["learnedInsights"]) == 1)

# SL hit (LOSS)
snap = base_snapshot(active=open_order("LONG", 3580.0, 3575.0, 3595.0, 0.02))
run_exec(snap, bid=3574.40, ask=3574.60)           # adj_bid 3574.42 <= sl-buffer 3574.53
os_ = snap["orderState"]
rec = os_["todayClosed"][0]
check("SL ชน (รวม buffer 47 จุด) → ปิดขาดทุน", rec.get("outcome") == "LOSS" and rec.get("pnlUsd") == -10.0 and "SL HIT" in rec.get("closeTag", ""))
check("cooldown แพ้ = จบแท่ง M15 + 30 นาที (สูตรตรง)", os_["cooldownUnlockAt"] == engine.compute_m15_cooldown_unlock_at(os_["lastCloseTs"], False) and os_["cooldownReason"] == "LOSS")

# TIME_EXIT_12B (3 ชั่วโมง)
snap = base_snapshot(active=open_order("LONG", 3580.0, 3575.0, 3595.0, 0.02, be_rule="TIME_EXIT_12B", created_ago_min=185))
run_exec(snap, bid=3582.0, ask=3582.2)
rec = snap["orderState"]["todayClosed"][0]
check("TIME_EXIT_12B ครบ 3 ชม. → ปิดที่ราคาตลาด", "TIME EXIT 12B" in rec.get("closeTag", "") and rec.get("exit") == 3582.02, rec.get("closeTag"))
snap = base_snapshot(active=open_order("LONG", 3580.0, 3575.0, 3595.0, 0.02, be_rule="TIME_EXIT_12B", created_ago_min=60))
run_exec(snap, bid=3582.0, ask=3582.2)
check("TIME_EXIT_12B ยังไม่ครบ 3 ชม. → ถือต่อ", snap["orderState"]["activeOrder"] is not None)

# Auto-BE move → BE stop
snap = base_snapshot(active=open_order("LONG", 3580.0, 3575.0, 3595.0, 0.02, be_rule="BE_AT_1R"))
run_exec(snap, bid=3585.10, ask=3585.30)           # adj_bid 3585.12 >= trigger 1R = 3585.00
ao = snap["orderState"]["activeOrder"]
check("Auto-BE: วิ่งถึง 1R → เลื่อน SL มาบังทุน (3580.20)", ao is not None and ao.get("beTriggered") is True and ao.get("sl") == 3580.20, ao.get("sl"))
check("Auto-BE แจ้ง ntfy", any("บังทุน" in a["title"] for a in _alerts))
run_exec(snap, bid=3579.60, ask=3579.80)           # adj_bid 3579.62 <= beSL-buffer 3579.73
rec = snap["orderState"]["todayClosed"][0]
check("หลังบังทุน SL ใหม่โดนชน → BE STOP", "BE STOP HIT" in rec.get("closeTag", "") and rec.get("exit") == 3580.20)

# LIMIT fill / cancel far
snap = base_snapshot(active=open_order("LONG", 3583.0, 3578.0, 3593.0, 0.02, mode="LIMIT"))
run_exec(snap, bid=3582.3, ask=3582.5)             # adj_ask 3582.52 <= entry 3583 → fill
ao = snap["orderState"]["activeOrder"]
check("LIMIT แมตช์ → เปลี่ยนเป็น OPEN + แจ้ง ntfy", ao is not None and ao.get("mode") == "OPEN" and any("แมตช์" in a["title"] for a in _alerts))
snap = base_snapshot(active=open_order("LONG", 3500.0, 3495.0, 3510.0, 0.02, mode="LIMIT", origin="AUTO"))
run_exec(snap, bid=3584.0, ask=3584.6)
check("LIMIT AUTO ไกลราคา > 160 จุด → ยกเลิกอัตโนมัติ", snap["orderState"]["activeOrder"] is None and any("ยกเลิก" in a["title"] for a in _alerts))

print("=== 3) เปิดไม้ใหม่ตามสูตรผู้ชนะ + guard ครบ ===")
snap = base_snapshot()
_gist_pushes.clear()
changed = run_exec(snap)
ao = snap["orderState"]["activeOrder"]
check("ผู้ใช้ปิดเว็บ + สูตรผู้ชนะผ่านเกณฑ์ → เปิดไม้เสมือนใหม่", changed is True and isinstance(ao, dict) and ao.get("mode") == "OPEN")
if ao:
    exp_entry = round(3584.6 + 0.02, 2)
    exp_sl = round(exp_entry - 3.75, 2)
    exp_lot = max(0.01, round((1000 * 0.01) / 375.0, 2))
    check("Entry = Ask + delta | SL = ATR×0.75 clamp | TP = SL×2.0 (สูตรผู้ชนะ LONDON)",
          ao.get("entry") == exp_entry and ao.get("sl") == exp_sl and ao.get("tp") == round(exp_entry + 7.5, 2),
          f"entry={ao.get('entry')} sl={ao.get('sl')} tp={ao.get('tp')}")
    check("Lot = Auto Prop 1% ของพอร์ตตามระยะ SL จริง", ao.get("lot") == exp_lot and ao.get("riskUsd") == round(exp_lot * 375, 2), f"lot={ao.get('lot')} risk={ao.get('riskUsd')}")
    check("มี beInfo ตามกฎสูตร (BE_AT_1R)", isinstance(ao.get("beInfo"), dict) and ao["beInfo"].get("rule") == "BE_AT_1R" and ao["beInfo"].get("shouldBe") is True)
check("ntfy แจ้งเปิดไม้ + push Gist 1 ครั้ง", any("เปิดไม้ Cloud" in a["title"] for a in _alerts) and len(_gist_pushes) == 1)
check("action summary บอกเปิดไม้ใหม่ (สำหรับแบนเนอร์)", "เปิดไม้ใหม่" in str(snap.get("cloudExecutorLastAction")) and snap.get("cloudExecutorActive") is True)

# effectiveRiskPct 0.5% (Discipline Guard จากหน้าเว็บ): Lot ต้องลดตามค่าที่ซิงก์มา
snap = base_snapshot(); snap["masterConfig"]["effectiveRiskPct"] = 0.5
run_exec(snap)
ao_rr = snap["orderState"]["activeOrder"]
check("effectiveRiskPct 0.5% → Lot ลดเหลือ 0.01 (เสี่ยง $3.75)", isinstance(ao_rr, dict) and ao_rr.get("lot") == 0.01 and ao_rr.get("riskUsd") == 3.75, ao_rr)

guards = [
    ("คูลดาวน์ยังไม่พ้น → ไม่เปิดไม้ใหม่", base_snapshot(cooldown_until=int(time.time() * 1000) + 10 * 60 * 1000), {}),
    ("ข่าว High Impact ±15 นาที → ไม่เปิดไม้ใหม่", base_snapshot(), {"news": [{"impact": "High", "hours_away": 0.2, "title": "NFP"}]}),
    ("สเปรดเกินเพดาน 65 จุด → ไม่เปิดไม้ใหม่", base_snapshot(), {"bid": 3580.0, "ask": 3582.0}),
    ("สูตรผู้ชนะ stability ไม่ผ่านเกณฑ์ → ไม่เปิดไม้ใหม่", base_snapshot(), {"ss": session_stable(WEAK_WINNER)}),
    ("Auto-Limit ปิด (autoArm=false) → ไม่เปิดไม้ใหม่", base_snapshot(auto_arm=False), {}),
]
for label, s, kw in guards:
    _gist_pushes.clear()
    changed = run_exec(s, **kw)
    check(label, changed is False and s["orderState"]["activeOrder"] is None and not _gist_pushes)

print("=== 4) Daily rollover (dedupe tradeId) / EOD report / Gist pull ===")
snap = base_snapshot()
snap["orderState"]["currentDate"] = "2026-11-23"
snap["orderState"]["todayClosed"] = [{"tradeId": "A", "pnlR": 1.0, "dateStr": "2026-11-24"}, {"tradeId": "B", "pnlR": -1.0, "dateStr": "2026-11-24"}]
snap["orderState"]["learningArchive"] = [{"tradeId": "B", "pnlR": -1.0, "dateStr": "2026-11-23"}]
_gist_pushes.clear()
run_exec(snap, bid=3584.0, ask=3584.6)
ids = [r.get("tradeId") for r in snap["orderState"]["learningArchive"]]
check("Rollover ย้ายสถิติเข้าสมุดเรียนรู้ + dedupe ไม่ซ้ำ", snap["orderState"]["todayClosed"] == [] and ids.count("A") == 1 and ids.count("B") == 1 and snap["orderState"]["currentDate"] == NOW.strftime("%Y-%m-%d"), str(ids))

# EOD
snap_eod = base_snapshot()
snap_eod["orderState"]["todayClosed"] = [
    {"tradeId": "E1", "dateStr": "2026-11-24", "closeTime": "20:10:00", "outcome": "WIN", "pnlR": 2.0, "pnlUsd": 20.0, "dir": "LONG", "formula": "LIQUIDITY_SWEEP", "closeTag": "TP HIT"},
    {"tradeId": "E2", "dateStr": "2026-11-24", "closeTime": "21:30:00", "outcome": "LOSS", "pnlR": -1.0, "pnlUsd": -10.0, "dir": "SHORT", "formula": "VP_EDGE_FADE", "closeTag": "SL HIT"},
]
_alerts.clear(); _gist_pushes.clear()
sent = engine.maybe_send_eod_report(snap_eod, NOW_EOD)
check("EOD 04:45-05:25 BKK → ส่งรายงานสรุป (1 ไม้ชนะ/1 แพ้, Net +1R/+$10)", sent is True and len(_alerts) == 1 and "สรุปปิดวัน 2026-11-24" in _alerts[0]["title"])
check("ข้อความ EOD ครบ: WR + Net R + Net $ + AI Post-Mortem ในตัว", all(x in _alerts[0]["message"] for x in ["Win Rate 50.0%", "+1.00R", "+10.00", "สรุปในตัว"]))
check("บันทึก lastEodReportDate ลง snapshot + push Gist (ตัดรหัสลับ)", snap_eod.get("lastEodReportDate") == "2026-11-24" and _last_push_payload()["lastEodReportDate"] == "2026-11-24" and _last_push_payload()["masterConfig"]["ghToken"] == "")
_alerts.clear()
check("เรียกซ้ำวันเดิม → ไม่ส่งซ้ำ", engine.maybe_send_eod_report(snap_eod, NOW_EOD) is False and not _alerts)
check("นอกหน้าต่างเวลา (16:00) → ไม่ส่ง", engine.maybe_send_eod_report(base_snapshot(), NOW) is False)

# Token guard: ไม่มี GIST_TOKEN → ไม่แตะออเดอร์/ไม่ส่ง EOD (กันทำงานซ้ำซ้อน)
_saved_token = engine.GIST_TOKEN
engine.GIST_TOKEN = ""
_alerts.clear(); _gist_pushes.clear()
snap_tg = base_snapshot(active=open_order("LONG", 3580.0, 3575.0, 3585.5, 0.02))
changed = run_exec(snap_tg, bid=3585.6, ask=3585.8)
check("ไม่มี GIST_TOKEN → ข้ามการจัดการออเดอร์ (ออเดอร์เดิมไม่ถูกแตะ)", changed is False and snap_tg["orderState"]["activeOrder"] is not None and not _gist_pushes and not _alerts)
_alerts.clear()
check("ไม่มี GIST_TOKEN → ข้าม EOD report", engine.maybe_send_eod_report(base_snapshot(), NOW_EOD) is False and not _alerts)
engine.GIST_TOKEN = _saved_token

# Gist pull: API 200 / API fail → raw fallback
sample = json.dumps(base_snapshot())
engine.requests.get = lambda url, headers=None, params=None, timeout=None: (
    _FakeResp(200, payload={"files": {engine.GIST_FILENAME: {"content": sample}}}) if "api.github.com" in url else _FakeResp(404)
)
got = engine.pull_canonical_snapshot()
check("pull: API ทางแรกสำเร็จ → source=api", isinstance(got, dict) and got.get("_source") == "api")
engine.requests.get = lambda url, headers=None, params=None, timeout=None: (
    _FakeResp(403) if "api.github.com" in url else _FakeResp(200, text=sample)
)
got = engine.pull_canonical_snapshot()
check("pull: API ตอบ 403 → fallback raw public สำเร็จ", isinstance(got, dict) and got.get("_source") == "raw")

engine.requests.get = _orig_requests_get
engine.requests.patch = _orig_requests_patch

print()
print(f"SUMMARY: {len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("FAILED:", FAILED)
    sys.exit(1)
print("HEADLESS DRY-RUN: ALL PASS (ไม่มีการเขียนข้อมูลจริง/ไม่ยิง network)")

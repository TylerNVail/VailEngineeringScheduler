# FirstApp.py — ClearConnect (Daytime-only 24h + Adaptive Splitting ≥5h)
# Updates in this version:
# - 24h clients: daytime-only (07:00–22:00). Nights ignored for now.
# - Removed duplicate 24h checkbox from Shifts tab (only in Client List).
# - Availability rows with blank "Availability Type" => treat as Available (hard).
# - NEW: Adaptive splitting at availability edges when a block can’t be placed:
#     * Sub-blocks >= 5h: auto-attempt placement.
#     * Sub-blocks < 5h: require approval before attempting; shows a clear exception row.
# - All prior rules preserved (travel, city preferences/penalties, day off, daily >9h -> approval, weekly prefs, etc.)

import streamlit as st
import pandas as pd
import os, io, zipfile, random, uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# ---------------- Page / layout ----------------
st.set_page_config(page_title="ClearConnect — Homecare Scheduler", layout="wide")
st.markdown(
    """
    <style>
      .block-container { max-width: 100% !important; padding-left: 1.5rem; padding-right: 1.5rem; }
      .app-title { text-align:center; font-size: 38px; font-weight: 700; margin: .5rem 0 1rem 0; }
      .app-footer { text-align:center; color:#666; padding: 1rem 0 0.5rem 0; }

      /* Zebra striping for ALL tables (dataframe + data_editor) */
      [data-testid="stDataFrame"] table tbody tr:nth-child(even),
      [data-testid="stDataEditor"] table tbody tr:nth-child(even) { background-color: #f5fbff !important; }
      table { font-family: Arial, Helvetica, sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="app-title">ClearConnect</div>', unsafe_allow_html=True)

# ---------------- CSV files ----------------
CAREGIVER_FILE = "caregivers.csv"
CAREGIVER_AVAIL_FILE = "caregiver_availability.csv"
CLIENT_FILE = "clients.csv"
CLIENT_FIXED_FILE = "client_fixed_shifts.csv"
CLIENT_FLEX_FILE = "client_flexible_shifts.csv"
APPROVALS_FILE = "approvals.csv"
BEST_SOLUTION_FILE = "best_solution.csv"
ITER_LOG_FILE = "iterative_runs.csv"
MANUAL_SHIFTS_FILE = "manual_shifts.csv"

def ensure_csv(path, cols):
    if not os.path.exists(path):
        pd.DataFrame(columns=cols).to_csv(path, index=False)

def load_csv_safe(path, cols):
    ensure_csv(path, cols)
    df = pd.read_csv(path, dtype=str).fillna("")
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()

def save_csv_safe(path, df):
    df2 = df.dropna(how="all").reset_index(drop=True).astype(str)
    df2.to_csv(path, index=False)

def ensure_ui_csvs():
    ensure_csv(CAREGIVER_FILE, ["Name","Base Location","Notes","SkipForWeek","Min Hours (week)","Max Hours (week)","AsManyHours"])
    ensure_csv(CAREGIVER_AVAIL_FILE, ["Caregiver Name","Day","Start","End","Availability Type","Notes"])
    ensure_csv(CLIENT_FILE, ["Name","Base Location","Importance","Scheduling Mode","Preferred Caregivers","Not Permitted Caregivers","Notes","24_Hour","SkipForWeek"])
    ensure_csv(CLIENT_FIXED_FILE, ["Client Name","Day","Start","End","SkipForWeek","Notes"])
    ensure_csv(CLIENT_FLEX_FILE, ["Client Name","Length (hrs)","Number of Shifts","Start Day","End Day","Start Time","End Time","Consecutive Days","SkipForWeek","Notes"])
    ensure_csv(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"])
    ensure_csv(ITER_LOG_FILE, ["iteration","score","timestamp","notes"])
    ensure_csv(MANUAL_SHIFTS_FILE, ["Client Name","Caregiver Name","Day","Start","End"])
    if not os.path.exists(BEST_SOLUTION_FILE):
        pd.DataFrame(columns=["block_id","client_id","caregiver_id","day","start_time","end_time","assignment_status"]).to_csv(BEST_SOLUTION_FILE, index=False)
ensure_ui_csvs()

def load_ui_csvs():
    return {
        "caregivers": load_csv_safe(CAREGIVER_FILE, ["Name","Base Location","Notes","SkipForWeek","Min Hours (week)","Max Hours (week)","AsManyHours"]),
        "caregiver_avail": load_csv_safe(CAREGIVER_AVAIL_FILE, ["Caregiver Name","Day","Start","End","Availability Type","Notes"]),
        "clients": load_csv_safe(CLIENT_FILE, ["Name","Base Location","Importance","Scheduling Mode","Preferred Caregivers","Not Permitted Caregivers","Notes","24_Hour","SkipForWeek"]),
        "client_fixed": load_csv_safe(CLIENT_FIXED_FILE, ["Client Name","Day","Start","End","SkipForWeek","Notes"]),
        "client_flex": load_csv_safe(CLIENT_FLEX_FILE, ["Client Name","Length (hrs)","Number of Shifts","Start Day","End Day","Start Time","End Time","Consecutive Days","SkipForWeek","Notes"]),
        "approvals": load_csv_safe(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"]),
        "iters": load_csv_safe(ITER_LOG_FILE, ["iteration","score","timestamp","notes"]),
        "manual": load_csv_safe(MANUAL_SHIFTS_FILE, ["Client Name","Caregiver Name","Day","Start","End"]),
        "best": pd.read_csv(BEST_SOLUTION_FILE, dtype=str).fillna(""),
    }
dfs = load_ui_csvs()

# Defaults
if "solver_iters" not in st.session_state:
    st.session_state["solver_iters"] = 500   # default 500
if "solver_time" not in st.session_state:
    st.session_state["solver_time"] = 10

# ---------------- Utilities ----------------
DAYS_FULL = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
DAY_SHORT = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
UI_TO_SHORT = dict(zip(DAYS_FULL, DAY_SHORT))
SHORT_TO_UI = dict(zip(DAY_SHORT, DAYS_FULL))

def time_15m_options():
    t = datetime(2000,1,1,0,0); opts=[]
    for _ in range(96):
        opts.append(t.strftime("%H:%M")); t += timedelta(minutes=15)
    return opts
TIME_OPTS = time_15m_options()

def parse_time_to_minutes(t:str)->int:
    if t == "24:00": return 24*60
    h,m = map(int, t.split(":")); return h*60+m

def time_to_slot(t:str)->int: return parse_time_to_minutes(t)//15
def slot_to_time(s:int)->str:
    m = s*15
    return "24:00" if m>=24*60 else f"{m//60:02d}:{m%60:02d}"

def drop_empty_rows(df):
    if df is None or df.empty: return pd.DataFrame(columns=df.columns)
    mask = ~(df.replace("", pd.NA).isna().all(axis=1))
    return df[mask].reset_index(drop=True)

def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

def sort_by_last_name(names:List[str])->List[str]:
    def key(n):
        n = (n or "").strip()
        parts = n.split()
        return (parts[-1].lower(), n.lower()) if parts else ("", "")
    return sorted(names, key=key)

# ---------------- Domain ----------------
@dataclass
class Caregiver:
    caregiver_id: str
    name: str
    base_location: str
    min_week_hours: float = 0.0
    max_week_hours: float = 0.0
    as_many_hours: bool = False
    availability: Dict[str, List[Dict]] = field(default_factory=dict)
    notes: str = ""
    work_log: Dict[str, List[Tuple[int,int,str]]] = field(default_factory=lambda: defaultdict(list))
    daily_city_trips: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

@dataclass
class Client:
    client_id: str
    name: str
    base_location: str
    priority: int
    scheduling_mode: str
    top_caregivers: List[str]
    banned_caregivers: List[str]
    requests: List[Dict]
    notes: str = ""

@dataclass
class ScheduleEntry:
    block_id: str
    client_id: str
    caregiver_id: str
    day: str
    start_time: str
    end_time: str
    assignment_status: str
    details: Dict = field(default_factory=dict)

@dataclass
class ExceptionOption:
    exception_id: str
    client_id: str
    caregiver_id: str  # "*" for policy exceptions (like split<5h)
    day: str
    start_time: str
    end_time: str
    exception_type: str
    details: Dict = field(default_factory=dict)

@dataclass
class SolverResult:
    schedule: List[ScheduleEntry]
    pending_exceptions: List[ExceptionOption]
    diagnostics: Dict = field(default_factory=dict)

# ---------------- Rules helpers ----------------
def travel_buffer_mins(city_a:str, city_b:str)->int:
    if not city_a or not city_b or city_a==city_b: return 30
    pair = {city_a, city_b}
    if pair == {"Paradise","Chico"}: return 30
    return 60

def is_pc(a,b)->bool: return {a,b} == {"Paradise","Chico"}

def travel_exception_type(city_a:str, city_b:str, pc_crossings_today:int)->Optional[str]:
    if not city_a or not city_b or city_a==city_b: return None
    pair = {city_a, city_b}
    if "Oroville" in pair: return "Travel Oroville"
    if pair == {"Paradise","Chico"} and pc_crossings_today >= 1: return "Paradise↔Chico (2nd+)"
    return None

DAILY_AUTO_LIMIT_MIN = 9*60
AS_MANY_WEEK_CAP_MIN = 50*60
MIN_AUTO_SUBBLOCK_SLOTS = 5*4  # 5 hours (in 15-min slots)

def is_available(cg:Caregiver, day_full:str, start:str, end:str)->Tuple[bool,bool]:
    """
    Returns (hard_ok, soft_prefer_used)
    Treat BLANK Availability Type as Available (hard). 'Preferred Unavailable' is soft (assignable).
    """
    day_short = UI_TO_SHORT.get(day_full, day_full)
    segs = cg.availability.get(day_short, [])
    s = time_to_slot(start); e = time_to_slot(end)
    if e<=s: return (False, False)
    if not segs:
        # No rows at all -> Not available for that day
        return (False, False)
    span = range(s,e)
    covered = [False]*(e-s); prefer=[False]*(e-s)
    for seg in segs:
        st = time_to_slot(seg.get("start","00:00")); en = time_to_slot(seg.get("end","00:00"))
        stype = (seg.get("state","") or "").lower().replace(" ", "_")
        # Blank => treat as available
        if stype == "" or stype == "available":
            is_cov = True; is_pref = False
        elif stype == "preferred_unavailable":
            is_cov = True; is_pref = True
        else:
            # Unknown types do NOT count as coverage
            is_cov = False; is_pref = False
        if not is_cov:
            continue
        for i,sl in enumerate(span):
            if st<=sl<en:
                covered[i] = True
                if is_pref:
                    prefer[i] = True
    if not all(covered): return (False, False)
    return (True, any(prefer))

def would_break_day_off(cg:Caregiver, day_full:str, start_slot:int, end_slot:int)->bool:
    days_worked = {d for d,blocks in cg.work_log.items() if blocks}
    prospective = set(days_worked)
    if start_slot < end_slot: prospective.add(day_full)
    return len(prospective) > 6

def has_overlap(cg:Caregiver, day:str, start_slot:int, end_slot:int)->bool:
    for (s,e,_) in cg.work_log.get(day, []):
        if not (end_slot <= s or e <= start_slot):
            return True
    return False

def current_daily_minutes(cg:Caregiver, day:str)->int:
    return sum((e-s)*15 for s,e,_ in cg.work_log.get(day, []))

def current_week_minutes(cg:Caregiver)->int:
    return sum((e-s)*15 for d in cg.work_log for s,e,_ in cg.work_log[d])

_city_lookup: Dict[str,str] = {}
def cl_city_for(client_id:str)->str: return _city_lookup.get(client_id, "")

def week_hour_pref_penalty(cg:Caregiver)->float:
    mins = current_week_minutes(cg); hrs = mins/60.0
    if cg.as_many_hours:
        return 0.5*max(0.0, hrs-50)
    lo = cg.min_week_hours or 0.0; hi = cg.max_week_hours or 0.0
    if hi and hrs>hi: return (hrs-hi) * 0.8
    if lo and hrs<lo: return (lo-hrs) * 0.3
    return 0.0

def per_caregiver_city_changes(assignments:List[ScheduleEntry])->int:
    changes = 0
    per = defaultdict(lambda: defaultdict(list))
    for a in assignments:
        per[a.caregiver_id][a.day].append((time_to_slot(a.start_time), time_to_slot(a.end_time), cl_city_for(a.client_id)))
    for _, days in per.items():
        for segs in days.values():
            segs.sort(key=lambda x:x[0])
            for i in range(1,len(segs)):
                if segs[i-1][2] != segs[i][2]:
                    changes += 1
    return changes

def client_gap_score(day_blocks:List[Tuple[int,int]])->int:
    day_blocks = sorted(day_blocks)
    score = 0
    for i in range(1, len(day_blocks)):
        gap = day_blocks[i][0] - day_blocks[i-1][1]
        if gap <= 2: score += 10       # ≤30m
        elif gap <= 4: score += 5      # ≤60m
        else: score += 1
    return score

def score_solution(assignments:List[ScheduleEntry], client_priority:Dict[str,int], cg_map:Dict[str,"Caregiver"])->float:
    score = 0.0
    for a in assignments:
        dur = (time_to_slot(a.end_time)-time_to_slot(a.start_time))*0.25
        pr = client_priority.get(a.client_id,0)
        score += dur * (1 + pr/10.0)
        if "Suggested" in a.assignment_status:
            score -= 2.0
    per_client_day = defaultdict(lambda: defaultdict(list))
    for a in assignments:
        per_client_day[a.client_id][a.day].append((time_to_slot(a.start_time), time_to_slot(a.end_time)))
    for _, days in per_client_day.items():
        for segs in days.values():
            score += client_gap_score(segs) * 0.5
    score -= per_caregiver_city_changes(assignments) * 2.0
    for cg in cg_map.values():
        score -= week_hour_pref_penalty(cg)
    return score

# ---------------- Split fixed into chunks ----------------
def split_fixed_block(day:str, start:str, end:str)->List[Tuple[str,str]]:
    """
    Deterministic, long-chunk splitting with 15-min precision.
    For daytime 07:00–22:00 windows, produce:
      07:00–13:00 (6h), 13:00–19:00 (6h), 19:00–22:00 (3h)
    For arbitrary windows, cut into ~7h pieces (clamped to the window).
    """
    s = time_to_slot(start); e = time_to_slot(end)
    if e <= s:
        return []
    chunks=[]
    DAY_S = time_to_slot("07:00"); DAY_E = time_to_slot("22:00")
    if s >= DAY_S and e <= DAY_E:
        anchors = [s, max(s, time_to_slot("13:00")), max(s, time_to_slot("19:00")), e]
        anchors = [max(s, min(a, e)) for a in anchors]
        seq = [anchors[0]]
        for a in anchors[1:]:
            if a > seq[-1]:
                seq.append(a)
        for i in range(len(seq)-1):
            if seq[i] < seq[i+1]:
                chunks.append((seq[i], seq[i+1]))
    else:
        step = 28  # 7 hours
        cur = s
        while cur < e:
            cut = min(e, cur + step)
            chunks.append((cur, cut))
            cur = cut

    # merge touching
    merged=[]
    for seg in chunks:
        if not merged: merged=[seg]
        elif merged[-1][1]==seg[0]: merged[-1]=(merged[-1][0], seg[1])
        else: merged.append(seg)
    return [(slot_to_time(x[0]), slot_to_time(x[1])) for x in merged]

# ---------------- Build client blocks ----------------
def expand_client_requests(clients:List[Client])->Tuple[List[Dict], List[Dict]]:
    """
    If client 24_Hour=True, we only generate daytime blocks 07:00–22:00 per day.
    Nights ignored for now.
    """
    blocks = []; flex_specs = []
    DAY_S = "07:00"; DAY_E = "22:00"
    for c in clients:
        is_24 = any(r.get("type")=="24flag" for r in c.requests)
        for req in c.requests:
            if req.get("type") == "fixed":
                st = max(req["start"], DAY_S) if is_24 else req["start"]
                en = min(req["end"], DAY_E)   if is_24 else req["end"]
                if is_24 and (en <= DAY_S or st >= DAY_E):
                    continue
                parts = split_fixed_block(req["day"], st, en)
                for stt,enn in parts:
                    blocks.append({
                        "block_id": f"B_{uuid.uuid4().hex[:8]}",
                        "client_id": c.client_id,
                        "day": req["day"],
                        "start": stt,
                        "end": enn,
                        "allow_split": False,   # base fixed piece
                        "flex": False
                    })
            elif req.get("type") == "flexible":
                flex_specs.append({
                    "client_id": c.client_id,
                    "blocks": int(req["blocks"]),
                    "duration": float(req["duration"]),
                    "days": list(req["days"]),
                    "window_start": max(req["window_start"], DAY_S) if is_24 else req["window_start"],
                    "window_end":   min(req["window_end"], DAY_E)   if is_24 else req["window_end"],
                    "consecutive": str(req.get("consecutive","")).strip().lower() in ("true","1","t","yes","y")
                })
            elif req.get("type") == "24flag":
                pass
    return blocks, flex_specs

def place_flex_blocks_one_plan(flex_specs:List[Dict], rng:random.Random)->List[Dict]:
    placed=[]
    d2i = {d:i for i,d in enumerate(DAYS_FULL)}
    for spec in flex_specs:
        dur_slots = int(round(spec["duration"]*4))
        ws = time_to_slot(spec["window_start"]); we = time_to_slot(spec["window_end"])
        allowed_days = list(spec["days"]); rng.shuffle(allowed_days)
        want = int(spec["blocks"]); used_days=[]; tries=0
        def ok_gap(newd):
            if spec["consecutive"]: return True
            ni = d2i[newd]
            return all(min((ni - d2i[u]) % 7, (d2i[u] - ni) % 7) >= 2 for u in used_days)
        while want>0 and tries<600:
            tries+=1
            if not allowed_days: break
            d = allowed_days[tries % len(allowed_days)]
            if not ok_gap(d): continue
            valids=[]
            for s in range(ws, max(ws, we - dur_slots) + 1):
                e = s + dur_slots
                if not (time_to_slot("07:00") <= s < time_to_slot("22:00")): continue
                if not (time_to_slot("07:00") < e <= time_to_slot("22:00")): continue
                valids.append(s)
            rng.shuffle(valids)
            if not valids: continue
            s0 = valids[0]; e0 = s0 + dur_slots
            placed.append({
                "block_id": f"B_{uuid.uuid4().hex[:8]}",
                "client_id": spec["client_id"],
                "day": d,
                "start": slot_to_time(s0),
                "end": slot_to_time(e0),
                "allow_split": True,
                "flex": True
            })
            used_days.append(d); want -= 1
    return placed

# ---------------- Travel / adjacency ----------------
def travel_buffer_slots(city_a:str, city_b:str)->int:
    return travel_buffer_mins(city_a, city_b)//15

def will_violate_travel(cg:Caregiver, day:str, start_slot:int, end_slot:int, new_city:str)->Tuple[bool, Optional[str]]:
    blocks = sorted(cg.work_log.get(day, []), key=lambda x:x[0])
    left=None; right=None
    for s,e,clid in blocks:
        if e <= start_slot: left=(s,e,clid)
        if s >= end_slot and right is None: right=(s,e,clid)
    if left:
        ls, le, lclid = left; lcity = cl_city_for(lclid)
        gap = start_slot - le; buf = travel_buffer_slots(lcity, new_city)
        if gap < buf: return (True, "Travel buffer")
        ex = travel_exception_type(lcity, new_city, cg.daily_city_trips.get(day,0))
        if ex: return (True, ex)
    if right:
        rs, re, rclid = right; rcity = cl_city_for(rclid)
        gap = rs - end_slot; buf = travel_buffer_slots(new_city, rcity)
        if gap < buf: return (True, "Travel buffer")
        ex = travel_exception_type(new_city, rcity, cg.daily_city_trips.get(day,0))
        if ex: return (True, ex)
    return (False, None)

def add_assignment(cg:Caregiver, day:str, start_slot:int, end_slot:int, client_id:str):
    cg.work_log[day].append((start_slot, end_slot, client_id))
    cg.work_log[day].sort(key=lambda x:x[0])
    blocks = cg.work_log[day]
    idx = [i for i,b in enumerate(blocks) if b==(start_slot,end_slot,client_id)][0]
    if idx>0:
        _,_,lclid = blocks[idx-1]
        if is_pc(cl_city_for(lclid), cl_city_for(client_id)):
            cg.daily_city_trips[day] += 1
    if idx < len(blocks)-1:
        _,_,rclid = blocks[idx+1]
        if is_pc(cl_city_for(rclid), cl_city_for(client_id)):
            cg.daily_city_trips[day] += 1

# ---------------- Exceptions CSV (human-readable) ----------------
def append_pending_exceptions_to_csv(pending:List[ExceptionOption]):
    if not pending: return
    exist = load_ui_csvs()["approvals"]
    keys_existing = set((r["client_name"], r["caregiver_name"], r["day"], r["start"], r["end"], r["constraint_type"]) for _,r in exist.iterrows())
    friendly = {
        "Travel buffer": "not enough travel time between adjacent assignments",
        "Travel Oroville": "Oroville crossing requires approval",
        "Paradise↔Chico (2nd+)": "more than one Paradise↔Chico trip in the same day requires approval",
        "Day off": "would violate caregiver’s 1-day-off rule",
        "Overlap": "overlaps an existing assignment",
        "Unavailable": "caregiver not available for the entire block",
        "Daily hours >9": "would push daily total above 9 hours",
        "Weekly hours >50 (as-many)": "would push weekly total above 50 hours (as-many-hours cap)",
        "Split <5h requires approval": "auto-splitting produced a sub-block shorter than 5 hours",
    }
    new_rows=[]
    for ex in pending:
        key = (ex.client_id, ex.caregiver_id, ex.day, ex.start_time, ex.end_time, ex.exception_type)
        if key in keys_existing: continue
        extra = friendly.get(ex.exception_type, "constraint would be violated")
        # caregiver_id may be "*" for split-policy approvals
        cg_name = ex.caregiver_id if ex.caregiver_id else "*"
        summary = f"{cg_name} could cover {ex.client_id} on {ex.day} {ex.start_time}-{ex.end_time}, but this would break: {ex.exception_type} ({extra})."
        new_rows.append({
            "approval_id": f"A_{uuid.uuid4().hex[:8]}",
            "client_name": ex.client_id,
            "caregiver_name": cg_name,
            "day": ex.day,
            "start": ex.start_time,
            "end": ex.end_time,
            "constraint_type": summary,
            "decision": "",
            "timestamp": datetime.now().isoformat(),
            "notes": ""
        })
    if new_rows:
        updated = pd.concat([exist, pd.DataFrame(new_rows)], ignore_index=True)
        save_csv_safe(APPROVALS_FILE, updated)

# ---------------- Build objects from CSVs ----------------
def build_client_objects_from_dfs(dfs):
    clients=[]
    clients_rows = dfs["clients"].copy()
    clients_rows["__sort"] = clients_rows["Name"].apply(lambda n: (n.split()[-1].lower(), n.lower()) if isinstance(n,str) and n.strip() else ("", ""))
    clients_rows = clients_rows.sort_values(["__sort"], kind="stable").drop(columns=["__sort"])
    for _, r in clients_rows.iterrows():
        if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"): continue
        reqs=[]
        if str(r.get("24_Hour","")).strip().lower() in ("true","1","t","yes","y"):
            for d in DAYS_FULL:
                reqs.append({"type":"fixed","day": d, "start":"07:00", "end":"22:00"})
            reqs.append({"type":"24flag"})
        fixed_rows = dfs["client_fixed"][dfs["client_fixed"]["Client Name"]==r["Name"]]
        for _, fr in fixed_rows.iterrows():
            if str(fr.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"): continue
            day=fr.get("Day",""); start=fr.get("Start",""); end=fr.get("End","")
            if day and start and end: reqs.append({"type":"fixed","day": day, "start": start, "end": end})
        flex_rows = dfs["client_flex"][dfs["client_flex"]["Client Name"]==r["Name"]]
        for _, fx in flex_rows.iterrows():
            if str(fx.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"): continue
            try:
                ln=float(fx.get("Length (hrs)","") or 0); nm=int(float(fx.get("Number of Shifts","") or 0))
            except: 
                continue
            if ln<=0 or nm<=0: continue
            sday=fx.get("Start Day",""); eday=fx.get("End Day","")
            stime=fx.get("Start Time","") or "00:00"; etime=fx.get("End Time","") or "24:00"
            if sday in DAYS_FULL and eday in DAYS_FULL:
                si=DAYS_FULL.index(sday); ei=DAYS_FULL.index(eday)
                allowed=DAYS_FULL[si:ei+1] if si<=ei else DAYS_FULL[si:]+DAYS_FULL[:ei+1]
            elif sday in DAYS_FULL: allowed=[sday]
            else: allowed=DAYS_FULL.copy()
            reqs.append({
                "type":"flexible","blocks":nm,"duration":ln,"days":allowed,
                "window_start":stime,"window_end":etime,
                "consecutive": fx.get("Consecutive Days","")
            })
        clients.append(Client(
            client_id=r["Name"], name=r["Name"], base_location=r.get("Base Location",""),
            priority=int(r.get("Importance",0) or 0),
            scheduling_mode=r.get("Scheduling Mode","Maximize Client Preference"),
            top_caregivers=[p.strip() for p in str(r.get("Preferred Caregivers","")).split(",") if p.strip()],
            banned_caregivers=[p.strip() for p in str(r.get("Not Permitted Caregivers","")).split(",") if p.strip()],
            requests=reqs, notes=r.get("Notes","")
        ))
    return clients

def build_caregiver_objects_from_dfs(dfs):
    caregivers=[]
    rows = dfs["caregivers"].copy()
    rows["__sort"] = rows["Name"].apply(lambda n: (n.split()[-1].lower(), n.lower()) if isinstance(n,str) and n.strip() else ("", ""))
    rows = rows.sort_values(["__sort"], kind="stable").drop(columns=["__sort"])
    for _, r in rows.iterrows():
        if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"): continue
        name = r["Name"]
        av_rows = dfs["caregiver_avail"][dfs["caregiver_avail"]["Caregiver Name"]==name]
        av_map = defaultdict(list)
        for _, a in av_rows.iterrows():
            dshort = UI_TO_SHORT.get(a.get("Day",""), a.get("Day",""))
            if dshort:
                av_map[dshort].append({
                    "start": a.get("Start",""), "end": a.get("End",""),
                    "state": (a.get("Availability Type","") or "")  # blank treated as Available later
                })
        min_h = float(r.get("Min Hours (week)","") or 0)
        max_h = float(r.get("Max Hours (week)","") or 0)
        as_many = str(r.get("AsManyHours","")).strip().lower() in ("true","1","t","yes","y")
        caregivers.append(Caregiver(
            caregiver_id=name, name=name, base_location=r.get("Base Location",""),
            min_week_hours=min_h, max_week_hours=max_h, as_many_hours=as_many,
            availability=av_map, notes=r.get("Notes","")
        ))
    return caregivers

# ---------------- Uncovered overlays ----------------
def compute_all_requested_blocks(clients:List[Client]):
    fixed_blocks, flex_specs = expand_client_requests(clients)
    return fixed_blocks, flex_specs

def representative_slot_for_flex(spec)->Optional[Tuple[str,str,str]]:
    dur_slots = int(round(spec["duration"]*4))
    ws = time_to_slot(spec["window_start"]); we = time_to_slot(spec["window_end"])
    for d in spec["days"]:
        for s in range(ws, max(ws, we - dur_slots) + 1):
            e = s + dur_slots
            if not (time_to_slot("07:00") <= s < time_to_slot("22:00")): continue
            if not (time_to_slot("07:00") < e <= time_to_slot("22:00")): continue
            return d, slot_to_time(s), slot_to_time(e)
    return None

def compute_uncovered(dfs)->Tuple[pd.DataFrame, Dict[str, List[Dict]]]:
    clients = build_client_objects_from_dfs(dfs)
    fixed_blocks, flex_specs = compute_all_requested_blocks(clients)
    assigned = dfs["best"]
    assigned_keys = set((r["client_id"], r["day"], r["start_time"], r["end_time"]) for _,r in assigned.iterrows())
    uncovered_fixed = [b for b in fixed_blocks if (b["client_id"], b["day"], b["start"], b["end"]) not in assigned_keys]
    display_flex = []
    for spec in flex_specs:
        for _ in range(int(spec["blocks"])):
            rep = representative_slot_for_flex(spec)
            if rep:
                d, st, en = rep
                display_flex.append({"client_id": spec["client_id"], "day": d, "start": st, "end": en})
    mat = pd.DataFrame(index=TIME_OPTS, columns=DAYS_FULL).fillna("")
    for b in uncovered_fixed:
        s=time_to_slot(b["start"]); e=time_to_slot(b["end"])
        for sl in range(s,e):
            t=slot_to_time(sl)
            val = mat.at[t, b["day"]]
            lab = f'{b["client_id"]}'
            mat.at[t, b["day"]] = (val + " | " if val else "") + lab
    for f in display_flex:
        s=time_to_slot(f["start"]); e=time_to_slot(f["end"])
        for sl in range(s,e):
            t=slot_to_time(sl)
            val = mat.at[t, f["day"]]
            lab = f'{f["client_id"]} (F)'
            mat.at[t, f["day"]] = (val + " | " if val else "") + lab
    per_client_unfilled = defaultdict(list)
    for b in uncovered_fixed:
        per_client_unfilled[b["client_id"]].append({"day": b["day"], "start": b["start"], "end": b["end"], "label": "Not covered"})
    for f in display_flex:
        per_client_unfilled[f["client_id"]].append({"day": f["day"], "start": f["start"], "end": f["end"], "label": "Not covered (F)"})
    return mat, per_client_unfilled

# ---------------- Solver (with Adaptive Splitting ≥5h) ----------------
def solve_week(
    caregivers:List[Caregiver],
    clients:List[Client],
    approvals_df:pd.DataFrame,
    iterations:int=1,
    per_iter_time:int=10,
    random_seed:int=0,
    locked_assignments:Optional[pd.DataFrame]=None,
    respect_locks:bool=True,
    manual_locks:Optional[pd.DataFrame]=None,
)->SolverResult:

    global _city_lookup
    _city_lookup = {c.client_id: c.base_location for c in clients}

    locks=[]
    if respect_locks and locked_assignments is not None and not locked_assignments.empty:
        for _, r in locked_assignments.iterrows():
            locks.append((r["client_id"], r["day"], r["start_time"], r["end_time"], r["caregiver_id"], "Assigned (Locked)"))
    if manual_locks is not None and not manual_locks.empty:
        for _, r in manual_locks.iterrows():
            locks.append((r["Client Name"], r["Day"], r["Start"], r["End"], r["Caregiver Name"], "Assigned (Manual)"))

    approved_overrides=[]; declined_blacklist=set(); approved_small_splits=set()
    if approvals_df is not None and not approvals_df.empty:
        for _, r in approvals_df.iterrows():
            key = (r["client_name"], r["day"], r["start"], r["end"], r["caregiver_name"])
            dec = str(r.get("decision","")).lower()
            ct = str(r.get("constraint_type",""))
            if dec == "approved":
                if r["caregiver_name"] == "*" and "Split <5h" in ct:
                    approved_small_splits.add((r["client_name"], r["day"], r["start"], r["end"]))
                else:
                    approved_overrides.append(key)
            elif dec == "declined":
                declined_blacklist.add(key)

    rng = random.Random(random_seed)
    fixed_blocks, flex_specs = expand_client_requests(clients)
    pr_map = {c.client_id: c.priority for c in clients}
    client_map = {c.client_id: c for c in clients}
    cg_map = {c.caregiver_id: c for c in caregivers}

    # compose blocks
    all_blocks = fixed_blocks + place_flex_blocks_one_plan(flex_specs, rng)

    # index by day
    blocks_by_day = defaultdict(list)
    for b in all_blocks:
        blocks_by_day[b["day"]].append(b)

    # Remove any that are locked; we’ll force-place locks first
    lock_keys = set((clid, d, st, en) for (clid, d, st, en, _, _) in locks)
    for day in DAYS_FULL:
        blocks_by_day[day] = [b for b in blocks_by_day[day] if (b["client_id"], b["day"], b["start"], b["end"]) not in lock_keys]

    # round-robin order within a day
    def day_round_order(day):
        blocks = blocks_by_day.get(day, [])
        pref = defaultdict(list); nonpref = defaultdict(list)
        for b in blocks:
            cl = client_map[b["client_id"]]
            (pref if cl.top_caregivers else nonpref)[cl.client_id].append(b)
        for dct in (pref, nonpref):
            for cid in dct:
                dct[cid].sort(key=lambda x: time_to_slot(x["start"]))
        order=[]
        for bucket in (pref, nonpref):
            keys = sorted(bucket.keys(), key=lambda cid: -client_map[cid].priority)
            while any(bucket.values()):
                for cid in keys:
                    if bucket[cid]:
                        order.append(bucket[cid].pop(0))
        return order

    def candidate_caregivers(cl, cl_city):
        cand = [c for c in caregivers if c.caregiver_id not in cl.banned_caregivers]
        cand.sort(key=lambda cg: 0 if cg.caregiver_id in cl.top_caregivers else 1)
        cand.sort(key=lambda cg: 0 if cg.base_location==cl_city else 1)
        return cand

    def can_place_single(cg:Caregiver, cl_city:str, day:str, s:int, e:int)->Tuple[bool, Optional[str], bool]:
        hard_ok, soft_prefer = is_available(cg, day, slot_to_time(s), slot_to_time(e))
        if not hard_ok: return (False, "Unavailable", soft_prefer)
        if would_break_day_off(cg, day, s, e): return (False, "Day off", soft_prefer)
        if has_overlap(cg, day, s, e): return (False, "Overlap", soft_prefer)
        if current_daily_minutes(cg, day) + (e-s)*15 > DAILY_AUTO_LIMIT_MIN: return (False, "Daily hours >9", soft_prefer)
        if cg.as_many_hours and (current_week_minutes(cg) + (e-s)*15 > AS_MANY_WEEK_CAP_MIN): return (False, "Weekly hours >50 (as-many)", soft_prefer)
        violates, ex_type = will_violate_travel(cg, day, s, e, cl_city)
        if violates: return (False, ex_type or "Travel", soft_prefer)
        return (True, None, soft_prefer)

    def force_place(clid, day, st, en, caregiver_id, status="Assigned (Approved/Locked)"):
        cg = cg_map.get(caregiver_id); 
        if not cg: return False
        s=time_to_slot(st); e=time_to_slot(en); cl_city = cl_city_for(clid)
        ok, err, _ = can_place_single(cg, cl_city, day, s, e)
        if not ok:
            exceptions.append(ExceptionOption(f"E_{uuid.uuid4().hex[:8]}", clid, caregiver_id, day, st, en, err or "Constraint", {})); 
            return False
        assignments.append(ScheduleEntry(f"B_{uuid.uuid4().hex[:8]}", clid, caregiver_id, day, st, en, status))
        add_assignment(cg, day, s, e, clid)
        return True

    def availability_edges_for_window(day:str, s:int, e:int)->List[int]:
        """Collect cut-points from all caregiver availability segment boundaries that fall within [s,e]."""
        pts = {s, e}
        for cg in caregivers:
            segs = cg.availability.get(UI_TO_SHORT.get(day, day), [])
            for seg in segs:
                st = time_to_slot(seg.get("start","00:00")); en = time_to_slot(seg.get("end","00:00"))
                # consider only segments that overlap window
                if st < e and en > s:
                    pts.add(max(s, st)); pts.add(min(e, en))
        pts = [p for p in pts if s <= p <= e]
        pts.sort()
        return pts

    def try_assign_segment(cl, cl_city, day, s, e):
        placed = False; local_excs=[]
        for cg in candidate_caregivers(cl, cl_city):
            if (cl.client_id, day, slot_to_time(s), slot_to_time(e), cg.caregiver_id) in declined_blacklist:
                local_excs.append((cg.caregiver_id, "Declined earlier")); continue
            ok, err, soft = can_place_single(cg, cl_city, day, s, e)
            if not ok:
                local_excs.append((cg.caregiver_id, err or "Constraint")); continue
            status = "Assigned" if not soft else "Suggested (Soft)"
            assignments.append(ScheduleEntry(f"B_{uuid.uuid4().hex[:8]}", cl.client_id, cg.caregiver_id, day, slot_to_time(s), slot_to_time(e), status))
            add_assignment(cg, day, s, e, cl.client_id)
            placed=True; break
        if not placed:
            used=set()
            for cg_id, ex_type in local_excs:
                if cg_id in used: continue
                used.add(cg_id)
                exceptions.append(ExceptionOption(
                    exception_id=f"E_{uuid.uuid4().hex[:8]}",
                    client_id=cl.client_id, caregiver_id=cg_id, day=day,
                    start_time=slot_to_time(s), end_time=slot_to_time(e), exception_type=ex_type or "Constraint", details={}
                ))
                if len(used)>=5: break
        return placed

    def adaptive_split_and_place(cl, day, s, e):
        """Auto-split at availability edges. >=5h segments are attempted; <5h require prior approval."""
        cl_city = cl.base_location
        pts = availability_edges_for_window(day, s, e)
        if len(pts) <= 2:
            return False  # nothing to split with
        any_placed = False
        for i in range(len(pts)-1):
            a, b = pts[i], pts[i+1]
            if a >= b: continue
            seg_len = b - a
            if seg_len >= MIN_AUTO_SUBBLOCK_SLOTS:
                # try place this segment normally
                placed = try_assign_segment(cl, cl_city, day, a, b)
                any_placed = any_placed or placed
            else:
                # sub-5h: require approval before attempting
                key = (cl.client_id, day, slot_to_time(a), slot_to_time(b))
                if key in approved_small_splits:
                    placed = try_assign_segment(cl, cl_city, day, a, b)
                    any_placed = any_placed or placed
                else:
                    exceptions.append(ExceptionOption(
                        exception_id=f"E_{uuid.uuid4().hex[:8]}",
                        client_id=cl.client_id, caregiver_id="*",
                        day=day, start_time=slot_to_time(a), end_time=slot_to_time(b),
                        exception_type="Split <5h requires approval", details={}
                    ))
        return any_placed

    best_assignments=[]; best_exceptions=[]; best_score=None

    for it in range(max(1, int(iterations))):
        rng.seed(random_seed + it*7919)
        for cg in caregivers:
            cg.work_log = defaultdict(list); cg.daily_city_trips = defaultdict(int)

        assignments=[]; exceptions=[]

        # apply locks and approvals first
        for clid, day, stt, enn, cg_id, label in locks:
            force_place(clid, day, stt, enn, cg_id, status=label)
        for (clid, day, stt, enn, cg_id) in approved_overrides:
            if any(a.client_id==clid and a.day==day and a.start_time==stt and a.end_time==enn for a in assignments): continue
            force_place(clid, day, stt, enn, cg_id, status="Assigned (Approved)")

        # work days in randomized order
        day_order = DAYS_FULL[:]; rng.shuffle(day_order)

        for day in day_order:
            seq = day_round_order(day)

            for b in seq:
                cl = client_map[b["client_id"]]; cl_city = cl.base_location
                s=time_to_slot(b["start"]); e=time_to_slot(b["end"])

                # First, try whole segment
                if try_assign_segment(cl, cl_city, day, s, e):
                    continue

                # If whole failed, and window is big (or flex), try adaptive split
                duration_slots = e - s
                if duration_slots >= MIN_AUTO_SUBBLOCK_SLOTS or b.get("flex", False):
                    adaptive_split_and_place(cl, day, s, e)
                else:
                    # Too small to place and not eligible for adaptive split auto-attempt
                    exceptions.append(ExceptionOption(
                        exception_id=f"E_{uuid.uuid4().hex[:8]}",
                        client_id=cl.client_id, caregiver_id="*",
                        day=day, start_time=b["start"], end_time=b["end"],
                        exception_type="Split <5h requires approval", details={}
                    ))

        sc = score_solution(assignments, pr_map, {c.caregiver_id:c for c in caregivers})
        if best_score is None or sc>best_score:
            best_score = sc; best_assignments = assignments; best_exceptions = exceptions

    pd.DataFrame([{
        "block_id": a.block_id, "client_id": a.client_id, "caregiver_id": a.caregiver_id,
        "day": a.day, "start_time": a.start_time, "end_time": a.end_time, "assignment_status": a.assignment_status
    } for a in best_assignments]).to_csv(BEST_SOLUTION_FILE, index=False)

    return SolverResult(best_assignments, best_exceptions, diagnostics={"score": best_score or 0.0})

# ---------------- Render helpers ----------------
def render_schedule_matrix(assignments_df:pd.DataFrame, mode:str, person:str)->pd.DataFrame:
    mat = pd.DataFrame(index=TIME_OPTS, columns=DAYS_FULL).fillna("")
    if assignments_df is None or assignments_df.empty or not person:
        return mat
    df = assignments_df.copy()
    if mode=="caregiver":
        df = df[df["caregiver_id"]==person]; lab = lambda r: f'{r["client_id"]}'
    else:
        df = df[df["client_id"]==person]; lab = lambda r: f'{r["caregiver_id"]}'
    for _, r in df.iterrows():
        day=r["day"]; st=time_to_slot(r["start_time"]); en=time_to_slot(r["end_time"]); txt=lab(r)
        for s in range(st,en):
            t=slot_to_time(s)
            prev = mat.at[t, day]
            mat.at[t, day] = (prev + " | " if prev else "") + txt
    return mat

def empty_week_matrix(): return pd.DataFrame(index=TIME_OPTS, columns=DAYS_FULL).fillna("")

def manual_matrix_from_csv(dfs, client_name:str)->pd.DataFrame:
    mat = empty_week_matrix()
    m = dfs["manual"]; m = m[m["Client Name"]==client_name]
    for _, r in m.iterrows():
        day=r["Day"]; st=time_to_slot(r["Start"]); en=time_to_slot(r["End"]); cg=r["Caregiver Name"]
        for s in range(st,en):
            t=slot_to_time(s)
            prev = mat.at[t, day]
            mat.at[t, day] = (prev + " | " if prev else "") + cg
    return mat

def parse_manual_matrix_to_blocks(mat:pd.DataFrame, client_name:str)->List[Dict]:
    out=[]
    for day in DAYS_FULL:
        cur_name=None; run_start=None
        for idx, t in enumerate(TIME_OPTS):
            cell = (mat.at[t, day] or "").strip()
            if "|" in cell: cell = cell.split("|")[0].strip()
            if cell:
                if cur_name is None:
                    cur_name = cell; run_start = idx
                elif cell != cur_name:
                    out.append({"Client Name": client_name, "Caregiver Name": cur_name, "Day": day, "Start": TIME_OPTS[run_start], "End": TIME_OPTS[idx]})
                    cur_name = cell; run_start = idx
            else:
                if cur_name is not None:
                    out.append({"Client Name": client_name, "Caregiver Name": cur_name, "Day": day, "Start": TIME_OPTS[run_start], "End": TIME_OPTS[idx]})
                    cur_name=None; run_start=None
        if cur_name is not None:
            out.append({"Client Name": client_name, "Caregiver Name": cur_name, "Day": day, "Start": TIME_OPTS[run_start], "End": "24:00"})
    return out

# ---------------- UI ----------------
tabs = st.tabs(["Caregivers","Clients","Schedules","Exceptions","Settings"])

# CAREGIVERS
with tabs[0]:
    st.header("Caregivers")
    cg_sub = st.tabs(["Caregiver List (core profile)", "Availability"])

    with cg_sub[0]:
        st.subheader("Caregiver List (core profile)")
        cg_df = dfs["caregivers"].copy()
        cg_df["__sort"] = cg_df["Name"].apply(lambda n: (n.split()[-1].lower(), n.lower()) if isinstance(n,str) and n.strip() else ("", ""))
        cg_df = cg_df.sort_values(["__sort"], kind="stable").drop(columns=["__sort"])
        edited = st.data_editor(
            cg_df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Name": st.column_config.TextColumn("Name"),
                "Base Location": st.column_config.TextColumn("Base Location"),
                "Notes": st.column_config.TextColumn("Notes"),
                "SkipForWeek": st.column_config.TextColumn("SkipForWeek (True | False)"),
                "Min Hours (week)": st.column_config.TextColumn("Min Hours (week)"),
                "Max Hours (week)": st.column_config.TextColumn("Max Hours (week)"),
                "AsManyHours": st.column_config.TextColumn("AsManyHours (True | False)"),
            },
            key="cg_list_editor",
        )
        if st.button("💾 Save Caregiver List"):
            cleaned = drop_empty_rows(edited)
            save_csv_safe(CAREGIVER_FILE, cleaned)
            # purge orphan availability
            keep = set(cleaned["Name"].tolist())
            avail = dfs["caregiver_avail"]
            avail = avail[avail["Caregiver Name"].isin(keep)].reset_index(drop=True)
            save_csv_safe(CAREGIVER_AVAIL_FILE, avail)
            st.success("Caregiver list saved (and orphan availability purged).")
            dfs = load_ui_csvs()

    with cg_sub[1]:
        st.subheader("Caregiver Availability")
        cg_names = sort_by_last_name(dfs["caregivers"]["Name"].tolist()) if not dfs["caregivers"].empty else []
        sel = st.selectbox("Select Caregiver", options=[""]+cg_names, index=0, key="avail_select")
        if sel:
            cur = dfs["caregivers"].loc[dfs["caregivers"]["Name"]==sel, "SkipForWeek"]
            cur = (str(cur.iloc[0]).strip().lower() in ("true","1","t","yes","y")) if len(cur)>0 else False
            skip_val = st.checkbox("Skip for the week", value=cur, key=f"skip_cg_{sel}")
            if st.button("Save Skip flag for caregiver"):
                dfc = dfs["caregivers"].copy()
                dfc.loc[dfc["Name"]==sel, "SkipForWeek"] = "True" if skip_val else "False"
                save_csv_safe(CAREGIVER_FILE, dfc)
                st.success("Skip flag saved."); dfs = load_ui_csvs(); do_rerun()

            sub_av = dfs["caregiver_avail"][dfs["caregiver_avail"]["Caregiver Name"]==sel].copy()
            if sub_av.empty:
                sub_av = pd.DataFrame([{"Caregiver Name": sel, "Day":"", "Start":"", "End":"", "Availability Type":"", "Notes":""}])
            edited_av = st.data_editor(
                sub_av,
                num_rows="dynamic",
                width="stretch",
                column_config={
                    "Caregiver Name": st.column_config.TextColumn("Caregiver Name"),
                    "Day": st.column_config.TextColumn("Day"),
                    "Start": st.column_config.TextColumn("Start"),
                    "End": st.column_config.TextColumn("End"),
                    "Availability Type": st.column_config.TextColumn("Availability Type (Available | Preferred Unavailable)"),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                key=f"av_{sel}"
            )
            if st.button("💾 Save Availability for selected caregiver"):
                rest = dfs["caregiver_avail"][dfs["caregiver_avail"]["Caregiver Name"]!=sel].copy()
                save_csv_safe(CAREGIVER_AVAIL_FILE, pd.concat([rest, drop_empty_rows(edited_av)], ignore_index=True))
                st.success(f"Availability saved for {sel}."); dfs = load_ui_csvs()

# CLIENTS
with tabs[1]:
    st.header("Clients")
    cl_sub = st.tabs(["Client List (core profile)", "Shifts"])

    with cl_sub[0]:
        st.subheader("Client List (core profile)")
        cl_df = dfs["clients"].copy()
        cl_df["__sort"] = cl_df["Name"].apply(lambda n: (n.split()[-1].lower(), n.lower()) if isinstance(n,str) and n.strip() else ("", ""))
        cl_df = cl_df.sort_values(["__sort"], kind="stable").drop(columns=["__sort"])
        try:
            cl_df["Importance"] = pd.to_numeric(cl_df["Importance"].replace("", "0"), errors="coerce").fillna(0).astype(int)
        except: cl_df["Importance"] = 0
        edited = st.data_editor(
            cl_df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Name": st.column_config.TextColumn("Name"),
                "Base Location": st.column_config.TextColumn("Base Location"),
                "Importance": st.column_config.NumberColumn("Importance", min_value=0, max_value=10, step=1),
                "Scheduling Mode": st.column_config.TextColumn("Scheduling Mode (Maximize Client Preference | Maximize Fairness)"),
                "Preferred Caregivers": st.column_config.TextColumn("Preferred Caregivers (comma separated)"),
                "Not Permitted Caregivers": st.column_config.TextColumn("Not Permitted Caregivers (comma separated)"),
                "Notes": st.column_config.TextColumn("Notes"),
                "24_Hour": st.column_config.TextColumn("24_Hour (True | False)"),
                "SkipForWeek": st.column_config.TextColumn("SkipForWeek (True | False)"),
            },
            key="cl_list_editor",
        )
        if st.button("💾 Save Client List"):
            cleaned = drop_empty_rows(edited)
            cleaned["Importance"] = pd.to_numeric(cleaned["Importance"].replace("", "0"), errors="coerce").fillna(0).astype(int)
            save_csv_safe(CLIENT_FILE, cleaned.astype(str))
            # purge orphan shift rows
            keep = set(cleaned["Name"].tolist())
            fx = dfs["client_fixed"]; fx = fx[fx["Client Name"].isin(keep)].reset_index(drop=True)
            fl = dfs["client_flex"]; fl = fl[fl["Client Name"].isin(keep)].reset_index(drop=True)
            save_csv_safe(CLIENT_FIXED_FILE, fx); save_csv_safe(CLIENT_FLEX_FILE, fl)
            st.success("Client list saved (and orphan shifts purged).")
            dfs = load_ui_csvs()

    with cl_sub[1]:
        st.subheader("Client Shifts")
        names = sort_by_last_name(dfs["clients"]["Name"].tolist()) if not dfs["clients"].empty else []
        sel = st.selectbox("Select Client", options=[""]+names, index=0, key="shift_select")
        if sel:
            # Removed 24h toggle here per your request; it now lives only on Client List.
            st.markdown("**Fixed Shifts**")
            sub_fx = dfs["client_fixed"][dfs["client_fixed"]["Client Name"]==sel].copy()
            if sub_fx.empty:
                sub_fx = pd.DataFrame([{"Client Name": sel, "Day":"", "Start":"", "End":"", "SkipForWeek":"", "Notes":""}])
            edited_fx = st.data_editor(sub_fx, num_rows="dynamic", width="stretch", key=f"fx_{sel}",
                column_config={
                    "Client Name": st.column_config.TextColumn("Client Name"),
                    "Day": st.column_config.TextColumn("Day"),
                    "Start": st.column_config.TextColumn("Start"),
                    "End": st.column_config.TextColumn("End"),
                    "SkipForWeek": st.column_config.TextColumn("SkipForWeek (True | False)"),
                    "Notes": st.column_config.TextColumn("Notes"),
                }
            )

            st.markdown("**Flexible Shifts**")
            sub_fl = dfs["client_flex"][dfs["client_flex"]["Client Name"]==sel].copy()
            if sub_fl.empty:
                sub_fl = pd.DataFrame([{"Client Name": sel, "Length (hrs)":"", "Number of Shifts":"", "Start Day":"", "End Day":"", "Start Time":"", "End Time":"", "Consecutive Days":"", "SkipForWeek":"", "Notes":""}])
            edited_fl = st.data_editor(sub_fl, num_rows="dynamic", width="stretch", key=f"fl_{sel}",
                column_config={
                    "Client Name": st.column_config.TextColumn("Client Name"),
                    "Length (hrs)": st.column_config.TextColumn("Length (hrs)"),
                    "Number of Shifts": st.column_config.TextColumn("Number of Shifts"),
                    "Start Day": st.column_config.TextColumn("Start Day"),
                    "End Day": st.column_config.TextColumn("End Day"),
                    "Start Time": st.column_config.TextColumn("Start Time"),
                    "End Time": st.column_config.TextColumn("End Time"),
                    "Consecutive Days": st.column_config.TextColumn("Consecutive Days (True | False)"),
                    "SkipForWeek": st.column_config.TextColumn("SkipForWeek (True | False)"),
                    "Notes": st.column_config.TextColumn("Notes"),
                }
            )

            if st.button("💾 Save Shifts for selected client"):
                rest_fx = dfs["client_fixed"][dfs["client_fixed"]["Client Name"]!=sel].copy()
                rest_fl = dfs["client_flex"][dfs["client_flex"]["Client Name"]!=sel].copy()
                save_csv_safe(CLIENT_FIXED_FILE, pd.concat([rest_fx, drop_empty_rows(edited_fx)], ignore_index=True))
                save_csv_safe(CLIENT_FLEX_FILE, pd.concat([rest_fl, drop_empty_rows(edited_fl)], ignore_index=True))
                st.success(f"Shifts saved for {sel}."); dfs = load_ui_csvs()

# SCHEDULES
with tabs[2]:
    st.header("Schedules")

    if st.button("▶️ Solve Schedules (run solver)"):
        caregivers = build_caregiver_objects_from_dfs(dfs)
        clients = build_client_objects_from_dfs(dfs)
        manual_df = dfs["manual"]
        seed = random.randint(1,1_000_000)
        result = solve_week(
            caregivers=caregivers, clients=clients, approvals_df=dfs["approvals"],
            iterations=int(st.session_state["solver_iters"]), per_iter_time=int(st.session_state["solver_time"]),
            random_seed=seed, locked_assignments=None, respect_locks=False,
            manual_locks=manual_df
        )
        append_pending_exceptions_to_csv(result.pending_exceptions)
        it_df = dfs["iters"]
        it_df = pd.concat([it_df, pd.DataFrame([{
            "iteration": st.session_state["solver_iters"],
            "score": result.diagnostics.get("score",0.0),
            "timestamp": datetime.now().isoformat(),
            "notes": "daytime-only 24h + adaptive split ≥5h"
        }])], ignore_index=True)
        save_csv_safe(ITER_LOG_FILE, it_df)
        dfs = load_ui_csvs()
        st.success(f"Solve complete. Best score={result.diagnostics.get('score',0.0)}. Exceptions added: {len(result.pending_exceptions)}")

    sch_sub = st.tabs(["Caregivers","Clients","Manual Shift Assignment"])

    with sch_sub[0]:
        st.subheader("Caregiver Schedule Viewer")
        cg_names = sort_by_last_name(dfs["caregivers"]["Name"].tolist())
        sel_cg = st.selectbox("Select Caregiver", options=[""]+cg_names, key="sched_cg_select")
        mat = render_schedule_matrix(dfs["best"], mode="caregiver", person=sel_cg)
        st.dataframe(mat, use_container_width=True, height=2000)

    with sch_sub[1]:
        st.subheader("Client Schedule Viewer")
        cl_names = sort_by_last_name(dfs["clients"]["Name"].tolist())
        sel_cl = st.selectbox("Select Client", options=[""]+cl_names, key="sched_client_select")
        base = render_schedule_matrix(dfs["best"], mode="client", person=sel_cl)
        all_uncovered, per_client_unfilled = compute_uncovered(dfs)
        overlay_list = per_client_unfilled.get(sel_cl, [])
        if overlay_list:
            for item in overlay_list:
                day=item["day"]; s=time_to_slot(item["start"]); e=time_to_slot(item["end"])
                lab=item["label"]
                for sl in range(s,e):
                    t=slot_to_time(sl)
                    if base.at[t, day]:
                        base.at[t, day] = base.at[t, day] + " | " + lab
                    else:
                        base.at[t, day] = lab
        def red_map(val):
            if isinstance(val,str) and ("Not covered" in val):
                return "color: red; font-weight: 600;"
            return ""
        styled = base.style.applymap(red_map)
        def stripes(_):
            styles = pd.DataFrame('', index=base.index, columns=base.columns); styles.iloc[::2,:]='background-color:#f5fbff'; return styles
        styled = styled.apply(stripes, axis=None)
        st.dataframe(styled, use_container_width=True, height=2000)

    with sch_sub[2]:
        st.subheader("Manual Shift Assignment")
        cl_names = sort_by_last_name(dfs["clients"]["Name"].tolist())
        sel_client = st.selectbox("Select Client", options=[""]+cl_names, key="manual_select_client")
        if sel_client:
            mat = manual_matrix_from_csv(dfs, sel_client)
            edited = st.data_editor(
                mat,
                key=f"manual_editor_{sel_client}",
                num_rows="fixed",
                width="stretch",
                height=2000
            )
            if st.button("💾 Save Manual Shifts for selected client"):
                blocks = parse_manual_matrix_to_blocks(pd.DataFrame(edited), sel_client)
                rest = dfs["manual"][dfs["manual"]["Client Name"]!=sel_client].copy()
                new = pd.DataFrame(blocks, columns=["Client Name","Caregiver Name","Day","Start","End"])
                save_csv_safe(MANUAL_SHIFTS_FILE, pd.concat([rest, new], ignore_index=True))
                st.success("Manual shifts saved."); dfs = load_ui_csvs()

# EXCEPTIONS
with tabs[3]:
    st.header("Exceptions & Approvals")

    st.subheader("Uncovered Shifts — All Clients")
    global_uncovered, _per = compute_uncovered(dfs)
    def red_if_text(val):
        return "color: red; font-weight: 600;" if isinstance(val,str) and val else ""
    styled_uncovered = global_uncovered.style.applymap(red_if_text)
    def stripes(_):
        styles = pd.DataFrame('', index=global_uncovered.index, columns=global_uncovered.columns); styles.iloc[::2,:]='background-color:#f5fbff'; return styles
    styled_uncovered = styled_uncovered.apply(stripes, axis=None)
    st.dataframe(styled_uncovered, use_container_width=True, height=900)

    st.markdown("---")
    st.subheader("Pending Approvals (next up)")

    approvals = dfs["approvals"]
    pending = approvals[approvals["decision"].astype(str).str.strip()==""].copy()

    if pending.empty:
        st.info("No pending exceptions.")
    else:
        first = pending.iloc[0]
        st.subheader(f"Review: {first['constraint_type']}")
        st.caption(f"Approval ID: {first['approval_id']}")

        # Snapshot window (±2h)
        def parse_minutes(t:str):
            try: h,m=map(int,t.split(":")); return h*60+m
            except: return None
        start_min = parse_minutes(first["start"]); end_min = parse_minutes(first["end"])
        if start_min is not None and end_min is not None:
            window_start = max(0, start_min-120); window_end=min(24*60, end_min+120)
            rows=[]; t=window_start
            while t<=window_end: rows.append(f"{t//60:02d}:{t%60:02d}"); t+=15
            snap = pd.DataFrame(index=rows, columns=[first["day"]]).fillna("")
            for r in rows:
                mins = parse_minutes(r)
                if mins is not None and start_min <= mins < end_min:
                    snap.at[r, first["day"]] = f"⚠ {first['client_name']} (exception)"
            def stripes2(_):
                styles = pd.DataFrame('', index=snap.index, columns=snap.columns); styles.iloc[::2,:]='background-color:#f5fbff'; return styles
            snap_styled = snap.style.apply(stripes2, axis=None).applymap(lambda v: "color:red; font-weight:600;" if v else "")
            st.dataframe(snap_styled, use_container_width=True, height=420)
        else:
            st.warning("Invalid time — cannot render snapshot.")

        c1,c2 = st.columns(2)
        if c1.button("Approve Exception"):
            approvals.loc[approvals["approval_id"]==first["approval_id"], "decision"] = "approved"
            approvals.loc[approvals["approval_id"]==first["approval_id"], "timestamp"] = datetime.now().isoformat()
            save_csv_safe(APPROVALS_FILE, approvals)
            st.success("Approved. Re-solving incrementally…")

            dfs_local = load_ui_csvs()
            best_locked = dfs_local["best"]
            caregivers = build_caregiver_objects_from_dfs(dfs_local)
            clients = build_client_objects_from_dfs(dfs_local)

            result = solve_week(
                caregivers=caregivers, clients=clients, approvals_df=dfs_local["approvals"],
                iterations=1, per_iter_time=int(st.session_state["solver_time"]),
                random_seed=random.randint(1,1_000_000), locked_assignments=best_locked, respect_locks=True,
                manual_locks=dfs_local["manual"]
            )
            append_pending_exceptions_to_csv(result.pending_exceptions)
            st.success("Re-solve complete."); do_rerun()

        if c2.button("Decline Exception"):
            approvals.loc[approvals["approval_id"]==first["approval_id"], "decision"] = "declined"
            approvals.loc[approvals["approval_id"]==first["approval_id"], "timestamp"] = datetime.now().isoformat()
            save_csv_safe(APPROVALS_FILE, approvals)
            st.success("Declined. Re-solving incrementally…")

            dfs_local = load_ui_csvs()
            best_locked = dfs_local["best"]
            caregivers = build_caregiver_objects_from_dfs(dfs_local)
            clients = build_client_objects_from_dfs(dfs_local)

            result = solve_week(
                caregivers=caregivers, clients=clients, approvals_df=dfs_local["approvals"],
                iterations=1, per_iter_time=int(st.session_state["solver_time"]),
                random_seed=random.randint(1,1_000_000), locked_assignments=best_locked, respect_locks=True,
                manual_locks=dfs_local["manual"]
            )
            append_pending_exceptions_to_csv(result.pending_exceptions)
            st.success("Re-solve complete."); do_rerun()

    st.markdown("---")
    st.subheader("Approval History")
    st.dataframe(dfs["approvals"], use_container_width=True, height=400)

# SETTINGS
with tabs[4]:
    st.header("Settings")
    st.subheader("Solver Settings")
    st.session_state["solver_iters"] = st.number_input("Iterative solving (restarts)", min_value=1, value=st.session_state["solver_iters"], step=1, key="iterative_solving_count_settings")
    st.session_state["solver_time"] = st.number_input("Per-iteration time limit (seconds)", min_value=1, value=st.session_state["solver_time"], step=1, key="per_iter_time_settings")
    st.caption("Used when you click 'Solve Schedules' on the Schedules tab.")

    st.subheader("Data Export/Import")
    if st.button("🗄️ Save As (download ZIP of all CSVs)"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for fname in [CAREGIVER_FILE, CAREGIVER_AVAIL_FILE, CLIENT_FILE, CLIENT_FIXED_FILE, CLIENT_FLEX_FILE, APPROVALS_FILE, BEST_SOLUTION_FILE, ITER_LOG_FILE, MANUAL_SHIFTS_FILE]:
                if os.path.exists(fname): z.write(fname)
        buf.seek(0)
        st.download_button("Download backup ZIP", data=buf.getvalue(), file_name=f"clearconnect_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")

    upl = st.file_uploader("Load From File (upload a ZIP exported from Save As)", type=["zip"])
    if upl is not None:
        try:
            z = zipfile.ZipFile(upl); z.extractall(".")
            st.success("ZIP extracted. Files overwritten locally."); dfs = load_ui_csvs(); do_rerun()
        except Exception as e:
            st.error(f"Failed to extract ZIP: {e}")

# Footer
st.markdown('<div class="app-footer">By: Vail Engineering</div>', unsafe_allow_html=True)

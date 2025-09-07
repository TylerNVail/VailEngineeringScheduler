# FirstApp.py — Streamlit app (UI + solver)
# New in this build:
# 1) Paradise↔Chico: >1 same-day crossings => exception (approvable), Oroville any time => exception.
# 2) UI hints on text columns for discrete choices.
# 3) Client field "Not Permitted Caregivers" (hard exclusion).
# 4) Soft penalty for caregiver city changes (per-day).
# 5) Fixed shifts >8h are split into ~6–8h chunks; never split inside 22:00–07:00 window.

import streamlit as st
import pandas as pd
import os, io, zipfile, random, uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import math

# ---------- Page / layout ----------
st.set_page_config(page_title="Homecare Scheduler", layout="wide")
st.markdown(
    """
    <style>
      .block-container { max-width: 100% !important; padding-left: 1.5rem; padding-right: 1.5rem; }
      .exception-cell { color: red; font-weight: bold; }
      table { font-family: Arial, Helvetica, sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- CSV files ----------
CAREGIVER_FILE = "caregivers.csv"
CAREGIVER_AVAIL_FILE = "caregiver_availability.csv"
CLIENT_FILE = "clients.csv"
CLIENT_FIXED_FILE = "client_fixed_shifts.csv"
CLIENT_FLEX_FILE = "client_flexible_shifts.csv"
APPROVALS_FILE = "approvals.csv"
BEST_SOLUTION_FILE = "best_solution.csv"
ITER_LOG_FILE = "iterative_runs.csv"

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
    ensure_csv(CAREGIVER_FILE, ["Name","Base Location","Notes","SkipForWeek"])
    ensure_csv(CAREGIVER_AVAIL_FILE, ["Caregiver Name","Day","Start","End","Availability Type","Notes"])
    ensure_csv(CLIENT_FILE, ["Name","Base Location","Importance","Scheduling Mode","Preferred Caregivers","Not Permitted Caregivers","Notes","24_Hour","SkipForWeek"])
    ensure_csv(CLIENT_FIXED_FILE, ["Client Name","Day","Start","End","Notes"])
    ensure_csv(CLIENT_FLEX_FILE, ["Client Name","Length (hrs)","Number of Shifts","Start Day","End Day","Start Time","End Time","Notes"])
    ensure_csv(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"])
    ensure_csv(ITER_LOG_FILE, ["iteration","score","timestamp","notes"])
    if not os.path.exists(BEST_SOLUTION_FILE):
        pd.DataFrame(columns=["block_id","client_id","caregiver_id","day","start_time","end_time","assignment_status"]).to_csv(BEST_SOLUTION_FILE, index=False)

ensure_ui_csvs()

def load_ui_csvs():
    return {
        "caregivers": load_csv_safe(CAREGIVER_FILE, ["Name","Base Location","Notes","SkipForWeek"]),
        "caregiver_avail": load_csv_safe(CAREGIVER_AVAIL_FILE, ["Caregiver Name","Day","Start","End","Availability Type","Notes"]),
        "clients": load_csv_safe(CLIENT_FILE, ["Name","Base Location","Importance","Scheduling Mode","Preferred Caregivers","Not Permitted Caregivers","Notes","24_Hour","SkipForWeek"]),
        "client_fixed": load_csv_safe(CLIENT_FIXED_FILE, ["Client Name","Day","Start","End","Notes"]),
        "client_flex": load_csv_safe(CLIENT_FLEX_FILE, ["Client Name","Length (hrs)","Number of Shifts","Start Day","End Day","Start Time","End Time","Notes"]),
        "approvals": load_csv_safe(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"]),
        "iters": load_csv_safe(ITER_LOG_FILE, ["iteration","score","timestamp","notes"]),
        "best": pd.read_csv(BEST_SOLUTION_FILE, dtype=str).fillna(""),
    }

dfs = load_ui_csvs()

# Keep solver settings stable via session state
if "solver_iters" not in st.session_state:
    st.session_state["solver_iters"] = 1
if "solver_time" not in st.session_state:
    st.session_state["solver_time"] = 10

# ---------- Constants / helpers ----------
DAYS_FULL = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
DAY_SHORT = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
UI_TO_SHORT = dict(zip(DAYS_FULL, DAY_SHORT))
SHORT_TO_UI = dict(zip(DAY_SHORT, DAYS_FULL))

def time_30m_options():
    t = datetime(2000,1,1,0,0); opts=[]
    for _ in range(48):
        opts.append(t.strftime("%H:%M")); t += timedelta(minutes=30)
    return opts
TIME_OPTS = time_30m_options()

def parse_time_to_minutes(t:str)->int:
    if t == "24:00": return 24*60
    h,m = map(int, t.split(":")); return h*60+m

def time_to_slot(t:str)->int:
    return parse_time_to_minutes(t)//30

def slot_to_time(s:int)->str:
    m = s*30
    return "24:00" if m>=24*60 else f"{m//60:02d}:{m%60:02d}"

def ensure_min_rows(df, n, defaults):
    if df is None or df.empty:
        df = pd.DataFrame(columns=list(defaults.keys()))
    if len(df) < n:
        add = n - len(df)
        df = pd.concat([df, pd.DataFrame([defaults.copy() for _ in range(add)])], ignore_index=True)
    return df.reset_index(drop=True)

def drop_empty_rows(df):
    if df is None or df.empty: return pd.DataFrame(columns=df.columns)
    mask = ~(df.replace("", pd.NA).isna().all(axis=1))
    return df[mask].reset_index(drop=True)

def do_rerun():
    if hasattr(st, "rerun"): st.rerun()
    elif hasattr(st, "experimental_rerun"): st.experimental_rerun()

# ---------- Domain ----------
@dataclass
class Caregiver:
    caregiver_id: str
    name: str
    base_location: str
    availability: Dict[str, List[Dict]] = field(default_factory=dict)
    notes: str = ""
    work_log: Dict[str, List[Tuple[int,int,str]]] = field(default_factory=lambda: defaultdict(list))
    daily_city_trips: Dict[str, int] = field(default_factory=lambda: defaultdict(int))  # counts of Paradise↔Chico crossings only

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
    caregiver_id: str
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

# ---------- Travel / rules ----------
def travel_buffer_mins(city_a:str, city_b:str)->int:
    if not city_a or not city_b or city_a==city_b: return 30
    pair = {city_a, city_b}
    if pair == {"Paradise","Chico"}: return 30
    return 60  # Any Oroville crossing

def is_pc(a,b)->bool:
    return {a,b} == {"Paradise","Chico"}

def travel_exception_type(city_a:str, city_b:str, pc_crossings_today:int)->Optional[str]:
    if not city_a or not city_b or city_a==city_b:
        return None
    pair = {city_a, city_b}
    if "Oroville" in pair:
        return "Travel Oroville"
    if pair == {"Paradise","Chico"} and pc_crossings_today >= 1:
        # 1st PC crossing ok; 2nd+ require approval
        return "Paradise↔Chico (2nd+)"
    return None

# ---------- Availability ----------
def is_available(cg:Caregiver, day_full:str, start:str, end:str)->Tuple[bool,bool]:
    day_short = UI_TO_SHORT.get(day_full, day_full)
    segs = cg.availability.get(day_short, [])
    s = time_to_slot(start); e = time_to_slot(end)
    if e<=s: return (False, False)
    if not segs: return (False, False)
    span = range(s,e)
    covered = [False]*(e-s); prefer=[False]*(e-s)
    for seg in segs:
        st = time_to_slot(seg.get("start","00:00"))
        en = time_to_slot(seg.get("end","00:00"))
        state = (seg.get("state","") or "").lower()
        for i,sl in enumerate(span):
            if st<=sl<en:
                covered[i] = True
                if state.startswith("prefer"):
                    prefer[i] = True
    if not all(covered): return (False, False)
    return (True, any(prefer))

def would_break_day_off(cg:Caregiver, day_full:str, start_slot:int, end_slot:int)->bool:
    days_worked = {d for d,blocks in cg.work_log.items() if blocks}
    prospective = set(days_worked)
    if start_slot < end_slot: prospective.add(day_full)
    return len(prospective) > 6  # must have ≥1 day off

def has_overlap(cg:Caregiver, day:str, start_slot:int, end_slot:int)->bool:
    for (s,e,_) in cg.work_log.get(day, []):
        if not (end_slot <= s or e <= start_slot):
            return True
    return False

# ---------- Gaps / scoring ----------
def client_gap_score(day_blocks:List[Tuple[int,int]])->int:
    day_blocks = sorted(day_blocks)
    score = 0
    for i in range(1, len(day_blocks)):
        gap = day_blocks[i][0] - day_blocks[i-1][1]
        if gap <= 1: score += 10
        elif gap == 2: score += 5
        else: score += 1
    return score

# Global city lookup for scoring transitions
_city_lookup: Dict[str,str] = {}
def cl_city_for(client_id:str)->str:
    return _city_lookup.get(client_id, "")

def per_caregiver_city_changes(assignments:List[ScheduleEntry])->int:
    # Count city transitions per caregiver per day
    changes = 0
    per = defaultdict(lambda: defaultdict(list))  # cg -> day -> [(s,e,city)]
    for a in assignments:
        per[a.caregiver_id][a.day].append((time_to_slot(a.start_time), time_to_slot(a.end_time), cl_city_for(a.client_id)))
    for cg, days in per.items():
        for day, segs in days.items():
            segs.sort(key=lambda x:x[0])
            for i in range(1,len(segs)):
                if segs[i-1][2] != segs[i][2]:
                    changes += 1
    return changes

def score_solution(assignments:List[ScheduleEntry], client_priority:Dict[str,int])->float:
    score = 0.0
    for a in assignments:
        dur = (time_to_slot(a.end_time)-time_to_slot(a.start_time))*0.5
        pr = client_priority.get(a.client_id,0)
        score += dur * (1 + pr/10.0)
        if "Suggested" in a.assignment_status:
            score -= 2.0
    # gap rewards per client/day
    per_client_day = defaultdict(lambda: defaultdict(list))
    for a in assignments:
        per_client_day[a.client_id][a.day].append((time_to_slot(a.start_time), time_to_slot(a.end_time)))
    for _, days in per_client_day.items():
        for segs in days.values():
            score += client_gap_score(segs) * 0.5
    # soft penalty for city changes
    changes = per_caregiver_city_changes(assignments)
    score -= changes * 2.0  # soft penalty
    return score

# ---------- Splitting long fixed blocks ----------
NIGHT_START = time_to_slot("22:00")  # 22:00
NIGHT_END   = time_to_slot("07:00")  # 07:00

def spans_night(s:int, e:int)->bool:
    # whether [s,e) intersects 22:00..24:00 or 00:00..07:00
    return (s < NIGHT_START < e) or (s < 48 and e > 48) or (s < NIGHT_END)

def split_fixed_block(day:str, start:str, end:str)->List[Tuple[str,str]]:
    """Split a fixed block (>8h) into ~6–8h chunks; never create a boundary inside 22:00..07:00."""
    s = time_to_slot(start); e = time_to_slot(end)
    total = e - s
    if total <= 16:  # <=8h
        return [(start, end)]
    # target size ~14 slots (7h), within [12..16] (6..8h)
    chunks = []
    cur = s
    while e - cur > 16:
        cut = cur + 14
        # Avoid cutting inside night window
        if NIGHT_START <= cut < 48:
            cut = min(NIGHT_START, cur + 16)
        if 0 <= cut < NIGHT_END:
            cut = max(NIGHT_END, cur + 12)
        cut = max(cur+12, min(cut, cur+16))  # clamp
        chunks.append((cur, cut))
        cur = cut
    chunks.append((cur, e))
    # Final pass: ensure we didn't create a cut *inside* 22:00..07:00
    sanitized=[]
    prev_s=None
    for i,(cs,ce) in enumerate(chunks):
        if NIGHT_START < cs < 48:  # start inside night -> pull back to NIGHT_START
            cs = NIGHT_START
        if 0 < cs < NIGHT_END:
            cs = NIGHT_END
        if NIGHT_START < ce < 48:  # end inside night -> push to NIGHT_START
            ce = NIGHT_START
        if 0 < ce < NIGHT_END:
            ce = NIGHT_END
        cs = max(s, min(cs, e))
        ce = max(s, min(ce, e))
        if cs<ce:
            sanitized.append((cs,ce))
    # merge any tiny fragments accidentally made equal by clamps
    merged=[]
    for seg in sanitized:
        if not merged: merged=[seg]; continue
        if merged[-1][1]==seg[0]:
            merged[-1]=(merged[-1][0], seg[1])
        else:
            merged.append(seg)
    return [(slot_to_time(x[0]), slot_to_time(x[1])) for x in merged]

# ---------- Build blocks ----------
def expand_client_requests(clients:List[Client])->Tuple[List[Dict], List[Dict]]:
    blocks = []
    flex_specs = []
    for c in clients:
        for req in c.requests:
            if req.get("type") == "fixed":
                # split if >8h
                parts = split_fixed_block(req["day"], req["start"], req["end"])
                for (st,en) in parts:
                    blocks.append({
                        "block_id": f"B_{uuid.uuid4().hex[:8]}",
                        "client_id": c.client_id,
                        "day": req["day"],
                        "start": st,
                        "end": en,
                        "allow_split": False,
                        "flex": False
                    })
            elif req.get("type") == "flexible":
                flex_specs.append({
                    "client_id": c.client_id,
                    "blocks": int(req["blocks"]),
                    "duration": float(req["duration"]),
                    "days": list(req["days"]),
                    "window_start": req["window_start"],
                    "window_end": req["window_end"]
                })
    return blocks, flex_specs

def place_flex_blocks_one_plan(flex_specs:List[Dict], rng:random.Random)->List[Dict]:
    placed=[]
    for spec in flex_specs:
        days = spec["days"][:]; rng.shuffle(days)
        used_days=set()
        dur_slots = int(round(spec["duration"]*2))
        ws = time_to_slot(spec["window_start"]); we = time_to_slot(spec["window_end"])
        count = spec["blocks"]; tries = 0
        while count>0 and tries<300:
            tries+=1
            if not days: break
            d = days[tries % len(days)]
            if d in used_days: continue
            valid=[]
            for s in range(ws, max(ws, we - dur_slots) + 1):
                e = s + dur_slots
                # don't start/end inside 22:00..07:00
                if not (14 <= s < 44):  # start in 07:00..22:00
                    continue
                if not (14 < e <= 48):  # end not in night
                    continue
                valid.append(s)
            rng.shuffle(valid)
            if not valid: continue
            start_slot = valid[0]; end_slot = start_slot + dur_slots
            placed.append({
                "block_id": f"B_{uuid.uuid4().hex[:8]}",
                "client_id": spec["client_id"],
                "day": d,
                "start": slot_to_time(start_slot),
                "end": slot_to_time(end_slot),
                "allow_split": True,
                "flex": True
            })
            used_days.add(d); count -= 1
    return placed

# ---------- Travel / add assignment ----------
def will_violate_travel(cg:Caregiver, day:str, start_slot:int, end_slot:int, new_city:str)->Tuple[bool, Optional[str]]:
    # Check neighbors on day
    blocks = sorted(cg.work_log.get(day, []), key=lambda x:x[0])
    left=None; right=None
    for s,e,clid in blocks:
        if e <= start_slot: left=(s,e,clid)
        if s >= end_slot and right is None: right=(s,e,clid)
    # left
    if left:
        ls, le, lclid = left
        lcity = cl_city_for(lclid)
        gap = start_slot - le
        buf = travel_buffer_mins(lcity, new_city)//30
        if gap < buf:
            return (True, "Travel buffer")
        # hard/approvable exceptions
        ex = travel_exception_type(lcity, new_city, cg.daily_city_trips.get(day,0))
        if ex: return (True, ex)
    # right
    if right:
        rs, re, rclid = right
        rcity = cl_city_for(rclid)
        gap = rs - end_slot
        buf = travel_buffer_mins(new_city, rcity)//30
        if gap < buf:
            return (True, "Travel buffer")
        ex = travel_exception_type(new_city, rcity, cg.daily_city_trips.get(day,0))
        if ex: return (True, ex)
    return (False, None)

def add_assignment(cg:Caregiver, day:str, start_slot:int, end_slot:int, client_id:str):
    cg.work_log[day].append((start_slot, end_slot, client_id))
    cg.work_log[day].sort(key=lambda x:x[0])
    # update PC crossings at adjacency boundaries
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

# ---------- Exceptions persistence ----------
def append_pending_exceptions_to_csv(pending:List[ExceptionOption]):
    if not pending: return
    exist = load_csv_safe(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"])
    keys_existing = set((r["client_name"], r["caregiver_name"], r["day"], r["start"], r["end"], r["constraint_type"]) for _,r in exist.iterrows())
    new_rows=[]
    for ex in pending:
        key = (ex.client_id, ex.caregiver_id, ex.day, ex.start_time, ex.end_time, ex.exception_type)
        if key in keys_existing: continue
        # Human-friendly explanation
        reason = ex.exception_type
        if reason == "Travel buffer":
            extra = "not enough travel time between adjacent assignments"
        elif reason.startswith("Travel Oroville"):
            extra = "Oroville crossing requires approval"
        elif reason.startswith("Paradise↔Chico"):
            extra = "more than one Paradise↔Chico trip in the same day requires approval"
        elif reason == "Day off":
            extra = "would violate caregiver’s 1-day-off rule"
        elif reason == "Overlap":
            extra = "overlaps an existing assignment"
        elif reason == "Unavailable":
            extra = "caregiver not available for the entire block"
        else:
            extra = "constraint would be violated"
        summary = f"{ex.caregiver_id} could cover {ex.client_id} on {ex.day} {ex.start_time}-{ex.end_time}, but this would break: {reason} ({extra})."
        new_rows.append({
            "approval_id": f"A_{uuid.uuid4().hex[:8]}",
            "client_name": ex.client_id,
            "caregiver_name": ex.caregiver_id,
            "day": ex.day,
            "start": ex.start_time,
            "end": ex.end_time,
            "constraint_type": summary,  # store human summary in the row
            "decision": "",
            "timestamp": datetime.now().isoformat(),
            "notes": ""
        })
    if new_rows:
        updated = pd.concat([exist, pd.DataFrame(new_rows)], ignore_index=True)
        save_csv_safe(APPROVALS_FILE, updated)

# ---------- Solver ----------
def solve_week(
    caregivers:List[Caregiver],
    clients:List[Client],
    approvals_df:pd.DataFrame,
    iterations:int=1,
    per_iter_time:int=10,
    random_seed:int=0,
    locked_assignments:Optional[pd.DataFrame]=None,
    respect_locks:bool=True,
)->SolverResult:

    # city map for scoring/transit
    global _city_lookup
    _city_lookup = {c.client_id: c.base_location for c in clients}

    # Locks + approvals
    locks=[]
    if respect_locks and locked_assignments is not None and not locked_assignments.empty:
        for _, r in locked_assignments.iterrows():
            locks.append((r["client_id"], r["day"], r["start_time"], r["end_time"], r["caregiver_id"]))

    approved_overrides=[]; declined_blacklist=set()
    if approvals_df is not None and not approvals_df.empty:
        for _, r in approvals_df.iterrows():
            # note: constraint_type now contains summary; keys are still client/day/start/end/cg
            key = (r["client_name"], r["day"], r["start"], r["end"], r["caregiver_name"])
            dec = str(r.get("decision","")).lower()
            if dec == "approved": approved_overrides.append(key)
            elif dec == "declined": declined_blacklist.add(key)

    rng = random.Random(random_seed)
    fixed_blocks, flex_specs = expand_client_requests(clients)
    pr_map = {c.client_id: c.priority for c in clients}
    client_map = {c.client_id: c for c in clients}
    cg_map = {c.caregiver_id: c for c in caregivers}

    def blocks_key(b):
        cl = client_map.get(b["client_id"])
        if not cl: return (1, 999)
        mode_rank = 0 if str(cl.scheduling_mode).lower().startswith("maximize") else 1
        return (mode_rank, -cl.priority)

    best_assignments=[]; best_exceptions=[]; best_score=None

    for it in range(max(1, int(iterations))):
        rng.seed(random_seed + it*7919)
        # reset caregiver state
        for cg in caregivers:
            cg.work_log = defaultdict(list)
            cg.daily_city_trips = defaultdict(int)

        assignments=[]; exceptions=[]

        def force_place(client_id, day, start, end, caregiver_id, status="Assigned (Approved/Locked)"):
            cg = cg_map.get(caregiver_id)
            if not cg: return False
            s = time_to_slot(start); e=time_to_slot(end)
            # Keep obvious overlap as exception rather than forcing silently
            if has_overlap(cg, day, s, e):
                exceptions.append(ExceptionOption(
                    exception_id=f"E_{uuid.uuid4().hex[:8]}",
                    client_id=client_id, caregiver_id=caregiver_id, day=day,
                    start_time=start, end_time=end, exception_type="Overlap (locked/approved)", details={}
                ))
                return False
            assignments.append(ScheduleEntry(
                block_id=f"B_{uuid.uuid4().hex[:8]}",
                client_id=client_id, caregiver_id=caregiver_id, day=day,
                start_time=start, end_time=end, assignment_status=status
            ))
            add_assignment(cg, day, s, e, client_id)
            return True

        # apply locks first
        for clid, day, stt, enn, cg_id in locks:
            force_place(clid, day, stt, enn, cg_id, status="Assigned (Locked)")

        # apply approved overrides
        for (clid, day, stt, enn, cg_id) in approved_overrides:
            if any(a.client_id==clid and a.day==day and a.start_time==stt and a.end_time==enn for a in assignments):
                continue
            force_place(clid, day, stt, enn, cg_id, status="Assigned (Approved)")

        # build list of remaining blocks this iteration
        def already(a_block):
            return any(
                a.client_id==a_block["client_id"] and a.day==a_block["day"] and a.start_time==a_block["start"] and a.end_time==a_block["end"]
                for a in assignments
            )
        iter_blocks = [b for b in (fixed_blocks + place_flex_blocks_one_plan(flex_specs, rng)) if not already(b)]
        iter_blocks.sort(key=blocks_key)

        for b in iter_blocks:
            cl = client_map.get(b["client_id"]); 
            if not cl: continue
            day=b["day"]; start=b["start"]; end=b["end"]
            s=time_to_slot(start); e=time_to_slot(end)
            cl_city = cl.base_location

            # candidates: preferred first (if maximize), same-city first, exclude banned
            cand = [c for c in caregivers if c.caregiver_id not in cl.banned_caregivers]
            if str(cl.scheduling_mode).lower().startswith("maximize"):
                cand.sort(key=lambda cg: 0 if cg.caregiver_id in cl.top_caregivers else 1)
            else:
                rng.shuffle(cand)
            cand.sort(key=lambda cg: 0 if cg.base_location==cl_city else 1)

            placed=False; local_excs=[]
            for cg in cand:
                # Declined exact pair?
                if (cl.client_id, day, start, end, cg.caregiver_id) in declined_blacklist:
                    local_excs.append((cg.caregiver_id, "Declined earlier"))
                    continue
                hard_ok, soft_prefer = is_available(cg, day, start, end)
                if not hard_ok:
                    local_excs.append((cg.caregiver_id, "Unavailable")); continue
                if would_break_day_off(cg, day, s, e):
                    local_excs.append((cg.caregiver_id, "Day off")); continue
                if has_overlap(cg, day, s, e):
                    local_excs.append((cg.caregiver_id, "Overlap")); continue
                violates, ex_type = will_violate_travel(cg, day, s, e, cl_city)
                if violates:
                    # Make it approvable: record exception instead of assigning
                    local_excs.append((cg.caregiver_id, ex_type or "Travel")); continue

                status = "Assigned" if not soft_prefer else "Suggested (Soft)"
                assignments.append(ScheduleEntry(
                    block_id=b["block_id"], client_id=cl.client_id, caregiver_id=cg.caregiver_id,
                    day=day, start_time=start, end_time=end, assignment_status=status
                ))
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
                        start_time=start, end_time=end, exception_type=ex_type or "Constraint", details={}
                    ))
                    if len(used)>=5: break

        sc = score_solution(assignments, pr_map)
        if best_score is None or sc>best_score:
            best_score = sc
            best_assignments = assignments
            best_exceptions = exceptions

    # Persist best schedule
    pd.DataFrame([{
        "block_id": a.block_id, "client_id": a.client_id, "caregiver_id": a.caregiver_id,
        "day": a.day, "start_time": a.start_time, "end_time": a.end_time, "assignment_status": a.assignment_status
    } for a in best_assignments]).to_csv(BEST_SOLUTION_FILE, index=False)

    return SolverResult(best_assignments, best_exceptions, diagnostics={"score": best_score or 0.0})

# ---------- Rendering ----------
def stripe_styler(df: pd.DataFrame):
    # alternate row background for readability
    def _row_style(idx):
        return ['background-color: #f5fbff' if i%2==0 else '' for i in range(len(df.columns))]
    return df.style.apply(lambda _: _row_style(_), axis=1)

def render_schedule_matrix(assignments_df:pd.DataFrame, mode:str, person:str)->pd.DataFrame:
    mat = pd.DataFrame(index=TIME_OPTS, columns=DAYS_FULL).fillna("")
    if assignments_df is None or assignments_df.empty or not person:
        return mat
    df = assignments_df.copy()
    if mode=="caregiver":
        df = df[df["caregiver_id"]==person]
        def label(r): return f'{r["client_id"]}'
    else:
        df = df[df["client_id"]==person]
        def label(r): return f'{r["caregiver_id"]}'
    for _, r in df.iterrows():
        day=r["day"]; st=time_to_slot(r["start_time"]); en=time_to_slot(r["end_time"]); lab=label(r)
        for s in range(st,en):
            t=slot_to_time(s)
            prev = mat.at[t, day]
            if prev and lab not in prev: mat.at[t, day] = prev + " | " + lab
            else: mat.at[t, day] = lab
    return mat

# ---------- UI ----------
tabs = st.tabs(["Caregivers","Clients","Schedules","Exceptions","Settings"])

# CAREGIVERS
with tabs[0]:
    st.header("Caregivers")
    cg_sub = st.tabs(["Caregiver List (core profile)", "Availability"])

    with cg_sub[0]:
        st.subheader("Caregiver List (core profile)")
        cg_df = dfs["caregivers"].copy()
        edited = st.data_editor(
            cg_df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Name": st.column_config.TextColumn("Name"),
                "Base Location": st.column_config.TextColumn("Base Location"),
                "Notes": st.column_config.TextColumn("Notes"),
                "SkipForWeek": st.column_config.TextColumn("SkipForWeek (True | False)"),
            },
            key="cg_list_editor",
        )
        if st.button("💾 Save Caregiver List"):
            save_csv_safe(CAREGIVER_FILE, drop_empty_rows(edited))
            st.success("Caregiver list saved.")
            dfs["caregivers"] = load_ui_csvs()["caregivers"]

    with cg_sub[1]:
        st.subheader("Caregiver Availability")
        cg_names = dfs["caregivers"]["Name"].tolist() if not dfs["caregivers"].empty else []
        sel = st.selectbox("Select Caregiver", options=[""]+cg_names, index=0, key="avail_select")
        if sel:
            # Skip flag quick toggle
            cur = dfs["caregivers"].loc[dfs["caregivers"]["Name"]==sel, "SkipForWeek"]
            cur = (str(cur.iloc[0]).strip().lower() in ("true","1","t","yes","y")) if len(cur)>0 else False
            skip_val = st.checkbox("Skip for the week", value=cur, key=f"skip_cg_{sel}")
            if st.button("Save Skip flag for caregiver"):
                dfs["caregivers"].loc[dfs["caregivers"]["Name"]==sel, "SkipForWeek"] = "True" if skip_val else "False"
                save_csv_safe(CAREGIVER_FILE, dfs["caregivers"]); st.success("Skip flag saved."); dfs["caregivers"]=load_ui_csvs()["caregivers"]; do_rerun()

            sub_av = dfs["caregiver_avail"][dfs["caregiver_avail"]["Caregiver Name"]==sel].copy()
            sub_av = ensure_min_rows(sub_av, 3, {"Caregiver Name": sel, "Day":"", "Start":"", "End":"", "Availability Type":"", "Notes":""})
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
                st.success(f"Availability saved for {sel}."); dfs["caregiver_avail"]=load_ui_csvs()["caregiver_avail"]

# CLIENTS
with tabs[1]:
    st.header("Clients")
    cl_sub = st.tabs(["Client List (core profile)", "Shifts"])

    with cl_sub[0]:
        st.subheader("Client List (core profile)")
        cl_df = dfs["clients"].copy()
        # Ensure numeric Importance
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
            st.success("Client list saved."); dfs["clients"]=load_ui_csvs()["clients"]

    with cl_sub[1]:
        st.subheader("Client Shifts")
        names = dfs["clients"]["Name"].tolist() if not dfs["clients"].empty else []
        sel = st.selectbox("Select Client", options=[""]+names, index=0, key="shift_select")
        if sel:
            cur_24 = dfs["clients"].loc[dfs["clients"]["Name"]==sel, "24_Hour"]
            cur_24 = (str(cur_24.iloc[0]).strip().lower() in ("true","1","t","yes","y")) if len(cur_24)>0 else False
            cur_skip = dfs["clients"].loc[dfs["clients"]["Name"]==sel, "SkipForWeek"]
            cur_skip = (str(cur_skip.iloc[0]).strip().lower() in ("true","1","t","yes","y")) if len(cur_skip)>0 else False
            c24 = st.checkbox("24-Hour Client", value=cur_24, key=f"c24_{sel}")
            skip_val = st.checkbox("Skip for the week", value=cur_skip, key=f"skip_client_{sel}")
            if st.button("Save 24-Hour / Skip flags for client"):
                dfs["clients"].loc[dfs["clients"]["Name"]==sel, "24_Hour"] = "True" if c24 else "False"
                dfs["clients"].loc[dfs["clients"]["Name"]==sel, "SkipForWeek"] = "True" if skip_val else "False"
                save_csv_safe(CLIENT_FILE, dfs["clients"]); st.success("Client flags saved."); dfs["clients"]=load_ui_csvs()["clients"]; do_rerun()

            st.markdown("**Fixed Shifts**")
            sub_fx = dfs["client_fixed"][dfs["client_fixed"]["Client Name"]==sel].copy()
            sub_fx = ensure_min_rows(sub_fx, 2, {"Client Name": sel, "Day":"", "Start":"", "End":"", "Notes":""})
            edited_fx = st.data_editor(sub_fx, num_rows="dynamic", width="stretch", key=f"fx_{sel}")

            st.markdown("**Flexible Shifts**")
            sub_fl = dfs["client_flex"][dfs["client_flex"]["Client Name"]==sel].copy()
            sub_fl = ensure_min_rows(sub_fl, 2, {"Client Name": sel, "Length (hrs)":"", "Number of Shifts":"", "Start Day":"", "End Day":"", "Start Time":"", "End Time":"", "Notes":""})
            edited_fl = st.data_editor(sub_fl, num_rows="dynamic", width="stretch", key=f"fl_{sel}")

            if st.button("💾 Save Shifts for selected client"):
                new_fixed = pd.concat([dfs["client_fixed"][dfs["client_fixed"]["Client Name"]!=sel], drop_empty_rows(edited_fx)], ignore_index=True)
                new_flex = pd.concat([dfs["client_flex"][dfs["client_flex"]["Client Name"]!=sel], drop_empty_rows(edited_fl)], ignore_index=True)
                save_csv_safe(CLIENT_FIXED_FILE, new_fixed); save_csv_safe(CLIENT_FLEX_FILE, new_flex)
                st.success(f"Shifts saved for {sel}."); dfs["client_fixed"]=load_ui_csvs()["client_fixed"]; dfs["client_flex"]=load_ui_csvs()["client_flex"]

# SCHEDULES
with tabs[2]:
    st.header("Schedules")

    # Solve button at top
    if st.button("▶️ Solve Schedules (run solver)"):
        # Build caregivers
        caregivers=[]
        for _, r in dfs["caregivers"].iterrows():
            if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"): continue
            name = r["Name"]
            av_rows = dfs["caregiver_avail"][dfs["caregiver_avail"]["Caregiver Name"]==name]
            av_map = defaultdict(list)
            for _, a in av_rows.iterrows():
                dshort = UI_TO_SHORT.get(a.get("Day",""), a.get("Day",""))
                if dshort:
                    av_map[dshort].append({"start": a.get("Start",""), "end": a.get("End",""), "state": (a.get("Availability Type","") or "").lower().replace(" ","_")})
            caregivers.append(Caregiver(
                caregiver_id=name, name=name, base_location=r.get("Base Location",""),
                availability=av_map, notes=r.get("Notes","")
            ))

        # Build clients + requests
        clients=[]
        for _, r in dfs["clients"].iterrows():
            if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"): continue
            reqs=[]
            if str(r.get("24_Hour","")).strip().lower() in ("true","1","t","yes","y"):
                for d in DAYS_FULL:
                    reqs.append({"type":"fixed","day": d, "start":"00:00", "end":"24:00"})
            fixed_rows = dfs["client_fixed"][dfs["client_fixed"]["Client Name"]==r["Name"]]
            for _, fr in fixed_rows.iterrows():
                day=fr.get("Day",""); start=fr.get("Start",""); end=fr.get("End","")
                if day and start and end: reqs.append({"type":"fixed","day": day, "start": start, "end": end})
            flex_rows = dfs["client_flex"][dfs["client_flex"]["Client Name"]==r["Name"]]
            for _, fx in flex_rows.iterrows():
                try:
                    ln=float(fx.get("Length (hrs)","") or 0); nm=int(float(fx.get("Number of Shifts","") or 0))
                except: continue
                if ln<=0 or nm<=0: continue
                sday=fx.get("Start Day",""); eday=fx.get("End Day","")
                stime=fx.get("Start Time","") or "00:00"; etime=fx.get("End Time","") or "24:00"
                if sday in DAYS_FULL and eday in DAYS_FULL:
                    si=DAYS_FULL.index(sday); ei=DAYS_FULL.index(eday)
                    allowed=DAYS_FULL[si:ei+1] if si<=ei else DAYS_FULL[si:]+DAYS_FULL[:ei+1]
                elif sday in DAYS_FULL: allowed=[sday]
                else: allowed=DAYS_FULL.copy()
                reqs.append({"type":"flexible","blocks":nm,"duration":ln,"days":allowed,"window_start":stime,"window_end":etime})
            clients.append(Client(
                client_id=r["Name"], name=r["Name"], base_location=r.get("Base Location",""),
                priority=int(r.get("Importance",0) or 0),
                scheduling_mode=r.get("Scheduling Mode","Maximize Client Preference"),
                top_caregivers=[p.strip() for p in str(r.get("Preferred Caregivers","")).split(",") if p.strip()],
                banned_caregivers=[p.strip() for p in str(r.get("Not Permitted Caregivers","")).split(",") if p.strip()],
                requests=reqs, notes=r.get("Notes","")
            ))

        seed = random.randint(1,1_000_000)
        result = solve_week(
            caregivers=caregivers, clients=clients, approvals_df=dfs["approvals"],
            iterations=int(st.session_state["solver_iters"]), per_iter_time=int(st.session_state["solver_time"]),
            random_seed=seed, locked_assignments=None, respect_locks=False
        )
        # Save pending exceptions
        from itertools import islice
        append_pending_exceptions_to_csv(result.pending_exceptions)
        # Log
        it_df = dfs["iters"]
        it_df = pd.concat([it_df, pd.DataFrame([{
            "iteration": st.session_state["solver_iters"],
            "score": result.diagnostics.get("score",0.0),
            "timestamp": datetime.now().isoformat(),
            "notes": "heuristic"
        }])], ignore_index=True)
        save_csv_safe(ITER_LOG_FILE, it_df)
        dfs = load_ui_csvs()
        st.success(f"Solve complete. Best score={result.diagnostics.get('score',0.0)}. Exceptions added: {len(result.pending_exceptions)}")

    sch_sub = st.tabs(["Caregivers","Clients"])
    with sch_sub[0]:
        st.subheader("Caregiver Schedule Viewer")
        cg_names = dfs["caregivers"]["Name"].tolist()
        sel_cg = st.selectbox("Select Caregiver", options=[""]+cg_names, key="sched_cg_select")
        mat = render_schedule_matrix(dfs["best"], mode="caregiver", person=sel_cg)
        st.dataframe(stripe_styler(mat), use_container_width=True, height=1500)

    with sch_sub[1]:
        st.subheader("Client Schedule Viewer")
        cl_names = dfs["clients"]["Name"].tolist()
        sel_cl = st.selectbox("Select Client", options=[""]+cl_names, key="sched_client_select")
        mat = render_schedule_matrix(dfs["best"], mode="client", person=sel_cl)
        st.dataframe(stripe_styler(mat), use_container_width=True, height=1500)

# EXCEPTIONS
with tabs[3]:
    st.header("Exceptions & Approvals")
    approvals = dfs["approvals"]
    pending = approvals[approvals["decision"].astype(str).str.strip()==""].copy()

    if pending.empty:
        st.info("No pending exceptions.")
    else:
        first = pending.iloc[0]
        st.subheader(f"Review: {first['constraint_type']}")
        st.caption(f"Approval ID: {first['approval_id']}")

        # Render a small local snapshot window (±2h)
        def parse_minutes(t:str):
            try: h,m=map(int,t.split(":")); return h*60+m
            except: return None
        start_min = parse_minutes(first["start"]); end_min = parse_minutes(first["end"])
        if start_min is not None and end_min is not None:
            window_start = max(0, start_min-120); window_end=min(24*60, end_min+120)
            rows=[]; t=window_start
            while t<=window_end: rows.append(f"{t//60:02d}:{t%60:02d}"); t+=30
            snap = pd.DataFrame(index=rows, columns=[first["day"]]).fillna("")
            for r in rows:
                mins = parse_minutes(r)
                if mins is not None and start_min <= mins < end_min:
                    snap.at[r, first["day"]] = f"⚠ {first['client_name']} (exception)"
            st.dataframe(stripe_styler(snap), use_container_width=True, height=420)
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

            # rebuild objects
            caregivers=[]; clients=[]
            for _, r in dfs_local["caregivers"].iterrows():
                if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"): continue
                name=r["Name"]
                av_rows = dfs_local["caregiver_avail"][dfs_local["caregiver_avail"]["Caregiver Name"]==name]
                av_map = defaultdict(list)
                for _, a in av_rows.iterrows():
                    dshort = UI_TO_SHORT.get(a.get("Day",""), a.get("Day",""))
                    if dshort:
                        av_map[dshort].append({"start":a.get("Start",""),"end":a.get("End",""),"state":(a.get("Availability Type","") or "").lower().replace(" ","_")})
                caregivers.append(Caregiver(caregiver_id=name, name=name, base_location=r.get("Base Location",""), availability=av_map))
            for _, r in dfs_local["clients"].iterrows():
                if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"): continue
                reqs=[]
                if str(r.get("24_Hour","")).strip().lower() in ("true","1","t","yes","y"):
                    for d in DAYS_FULL: reqs.append({"type":"fixed","day":d,"start":"00:00","end":"24:00"})
                fixed_rows = dfs_local["client_fixed"][dfs_local["client_fixed"]["Client Name"]==r["Name"]]
                for _, fr in fixed_rows.iterrows():
                    if fr.get("Day","") and fr.get("Start","") and fr.get("End",""):
                        reqs.append({"type":"fixed","day":fr["Day"],"start":fr["Start"],"end":fr["End"]})
                flex_rows = dfs_local["client_flex"][dfs_local["client_flex"]["Client Name"]==r["Name"]]
                for _, fx in flex_rows.iterrows():
                    try:
                        ln=float(fx.get("Length (hrs)","") or 0); nm=int(float(fx.get("Number of Shifts","") or 0))
                    except: continue
                    if ln<=0 or nm<=0: continue
                    sday=fx.get("Start Day",""); eday=fx.get("End Day","")
                    stime=fx.get("Start Time","") or "00:00"; etime=fx.get("End Time","") or "24:00"
                    if sday in DAYS_FULL and eday in DAYS_FULL:
                        si=DAYS_FULL.index(sday); ei=DAYS_FULL.index(eday)
                        allowed=DAYS_FULL[si:ei+1] if si<=ei else DAYS_FULL[si:]+DAYS_FULL[:ei+1]
                    elif sday in DAYS_FULL: allowed=[sday]
                    else: allowed=DAYS_FULL.copy()
                    reqs.append({"type":"flexible","blocks":nm,"duration":ln,"days":allowed,"window_start":stime,"window_end":etime})
                clients.append(Client(
                    client_id=r["Name"], name=r["Name"], base_location=r.get("Base Location",""),
                    priority=int(r.get("Importance",0) or 0), scheduling_mode=r.get("Scheduling Mode","Maximize Client Preference"),
                    top_caregivers=[p.strip() for p in str(r.get("Preferred Caregivers","")).split(",") if p.strip()],
                    banned_caregivers=[p.strip() for p in str(r.get("Not Permitted Caregivers","")).split(",") if p.strip()],
                    requests=reqs
                ))

            result = solve_week(
                caregivers=caregivers, clients=clients, approvals_df=load_ui_csvs()["approvals"],
                iterations=1, per_iter_time=int(st.session_state["solver_time"]),
                random_seed=random.randint(1,1_000_000), locked_assignments=best_locked, respect_locks=True
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
            caregivers=[]; clients=[]
            for _, r in dfs_local["caregivers"].iterrows():
                if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"): continue
                name=r["Name"]
                av_rows = dfs_local["caregiver_avail"][dfs_local["caregiver_avail"]["Caregiver Name"]==name]
                av_map = defaultdict(list)
                for _, a in av_rows.iterrows():
                    dshort = UI_TO_SHORT.get(a.get("Day",""), a.get("Day",""))
                    if dshort:
                        av_map[dshort].append({"start":a.get("Start",""),"end":a.get("End",""),"state":(a.get("Availability Type","") or "").lower().replace(" ","_")})
                caregivers.append(Caregiver(caregiver_id=name, name=name, base_location=r.get("Base Location",""), availability=av_map))
            for _, r in dfs_local["clients"].iterrows():
                if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"): continue
                reqs=[]
                if str(r.get("24_Hour","")).strip().lower() in ("true","1","t","yes","y"):
                    for d in DAYS_FULL: reqs.append({"type":"fixed","day":d,"start":"00:00","end":"24:00"})
                fixed_rows = dfs_local["client_fixed"][dfs_local["client_fixed"]["Client Name"]==r["Name"]]
                for _, fr in fixed_rows.iterrows():
                    if fr.get("Day","") and fr.get("Start","") and fr.get("End",""):
                        reqs.append({"type":"fixed","day":fr["Day"],"start":fr["Start"],"end":fr["End"]})
                flex_rows = dfs_local["client_flex"][dfs_local["client_flex"]["Client Name"]==r["Name"]]
                for _, fx in flex_rows.iterrows():
                    try:
                        ln=float(fx.get("Length (hrs)","") or 0); nm=int(float(fx.get("Number of Shifts","") or 0))
                    except: continue
                    if ln<=0 or nm<=0: continue
                    sday=fx.get("Start Day",""); eday=fx.get("End Day","")
                    stime=fx.get("Start Time","") or "00:00"; etime=fx.get("End Time","") or "24:00"
                    if sday in DAYS_FULL and eday in DAYS_FULL:
                        si=DAYS_FULL.index(sday); ei=DAYS_FULL.index(eday)
                        allowed=DAYS_FULL[si:ei+1] if si<=ei else DAYS_FULL[si:]+DAYS_FULL[:ei+1]
                    elif sday in DAYS_FULL: allowed=[sday]
                    else: allowed=DAYS_FULL.copy()
                    reqs.append({"type":"flexible","blocks":nm,"duration":ln,"days":allowed,"window_start":stime,"window_end":etime})
                clients.append(Client(
                    client_id=r["Name"], name=r["Name"], base_location=r.get("Base Location",""),
                    priority=int(r.get("Importance",0) or 0), scheduling_mode=r.get("Scheduling Mode","Maximize Client Preference"),
                    top_caregivers=[p.strip() for p in str(r.get("Preferred Caregivers","")).split(",") if p.strip()],
                    banned_caregivers=[p.strip() for p in str(r.get("Not Permitted Caregivers","")).split(",") if p.strip()],
                    requests=reqs
                ))

            result = solve_week(
                caregivers=caregivers, clients=clients, approvals_df=load_ui_csvs()["approvals"],
                iterations=1, per_iter_time=int(st.session_state["solver_time"]),
                random_seed=random.randint(1,1_000_000), locked_assignments=best_locked, respect_locks=True
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
            for fname in [CAREGIVER_FILE, CAREGIVER_AVAIL_FILE, CLIENT_FILE, CLIENT_FIXED_FILE, CLIENT_FLEX_FILE, APPROVALS_FILE, BEST_SOLUTION_FILE, ITER_LOG_FILE]:
                if os.path.exists(fname): z.write(fname)
        buf.seek(0)
        st.download_button("Download backup ZIP", data=buf.getvalue(), file_name=f"homecare_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")

    upl = st.file_uploader("Load From File (upload a ZIP exported from Save As)", type=["zip"])
    if upl is not None:
        try:
            z = zipfile.ZipFile(upl); z.extractall(".")
            st.success("ZIP extracted. Files overwritten locally."); dfs = load_ui_csvs(); do_rerun()
        except Exception as e:
            st.error(f"Failed to extract ZIP: {e}")

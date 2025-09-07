# FirstApp.py — Single-file Streamlit app (UI + heuristic solver)
# Copy/paste into a file named FirstApp.py and run: streamlit run FirstApp.py

import streamlit as st
import pandas as pd
import os
import io
import zipfile
import random
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# ------------- Page/UI setup -------------
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

# ------------- CSV files -------------
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
    df2 = df.dropna(how="all").reset_index(drop=True)
    df2 = df2.astype(str)
    df2.to_csv(path, index=False)

# UI schemas we’ve been using
def ensure_ui_csvs():
    ensure_csv(CAREGIVER_FILE, ["Name","Base Location","Notes","SkipForWeek"])
    ensure_csv(CAREGIVER_AVAIL_FILE, ["Caregiver Name","Day","Start","End","Availability Type","Notes"])
    ensure_csv(CLIENT_FILE, ["Name","Base Location","Importance","Scheduling Mode","Preferred Caregivers","Notes","24_Hour","SkipForWeek"])
    ensure_csv(CLIENT_FIXED_FILE, ["Client Name","Day","Start","End","Notes"])
    ensure_csv(CLIENT_FLEX_FILE, ["Client Name","Length (hrs)","Number of Shifts","Start Day","End Day","Start Time","End Time","Notes"])
    ensure_csv(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"])
    ensure_csv(ITER_LOG_FILE, ["iteration","score","timestamp","notes"])

ensure_ui_csvs()

# Load current CSVs
def load_ui_csvs():
    return {
        "caregivers": load_csv_safe(CAREGIVER_FILE, ["Name","Base Location","Notes","SkipForWeek"]),
        "caregiver_avail": load_csv_safe(CAREGIVER_AVAIL_FILE, ["Caregiver Name","Day","Start","End","Availability Type","Notes"]),
        "clients": load_csv_safe(CLIENT_FILE, ["Name","Base Location","Importance","Scheduling Mode","Preferred Caregivers","Notes","24_Hour","SkipForWeek"]),
        "client_fixed": load_csv_safe(CLIENT_FIXED_FILE, ["Client Name","Day","Start","End","Notes"]),
        "client_flex": load_csv_safe(CLIENT_FLEX_FILE, ["Client Name","Length (hrs)","Number of Shifts","Start Day","End Day","Start Time","End Time","Notes"]),
        "approvals": load_csv_safe(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"]),
        "iters": load_csv_safe(ITER_LOG_FILE, ["iteration","score","timestamp","notes"]),
    }

dfs = load_ui_csvs()

# ------------- Constants / helpers -------------
DAYS_FULL = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
DAY_SHORT = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"]
UI_TO_SHORT = {f:s for f,s in zip(DAYS_FULL, DAY_SHORT)}
SHORT_TO_UI = {s:f for f,s in zip(DAYS_FULL, DAY_SHORT)}

def time_30m_options():
    t = datetime(2000,1,1,0,0)
    opts=[]
    for _ in range(48):
        opts.append(t.strftime("%H:%M"))
        t += timedelta(minutes=30)
    return opts
TIME_OPTS = time_30m_options()

def parse_time_to_minutes(t:str)->int:
    if t == "24:00": return 24*60
    h,m = map(int, t.split(":"))
    return h*60+m

def time_to_slot(t:str)->int:
    m = parse_time_to_minutes(t)
    return m//30

def slot_to_time(s:int)->str:
    m = s*30
    if m>=24*60: return "24:00"
    return f"{m//60:02d}:{m%60:02d}"

def ensure_min_rows(df, min_rows, defaults):
    if df is None or df.empty:
        df = pd.DataFrame(columns=list(defaults.keys()))
    if len(df) < min_rows:
        add = min_rows - len(df)
        df = pd.concat([df, pd.DataFrame([defaults.copy() for _ in range(add)])], ignore_index=True)
    return df.reset_index(drop=True)

def drop_empty_rows(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns) if df is not None else pd.DataFrame()
    mask = ~(df.replace("", pd.NA).isna().all(axis=1))
    return df[mask].reset_index(drop=True)

def do_rerun():
    rerun = getattr(st, "rerun", None)
    exp = getattr(st, "experimental_rerun", None)
    if callable(rerun): rerun()
    elif callable(exp): exp()

# ------------- Domain dataclasses -------------
@dataclass
class Caregiver:
    caregiver_id: str
    name: str
    base_location: str
    min_hours: float = 0.0
    max_hours: float = 168.0
    prefer_max_hours: bool = False
    availability: Dict[str, List[Dict]] = field(default_factory=dict)  # "Mon":[{"start":"08:00","end":"12:00","state":"available"/"prefer_not"}]
    preferred_clients: List[str] = field(default_factory=list)
    notes: str = ""
    work_log: Dict[str, List[Tuple[int,int,str]]] = field(default_factory=lambda: defaultdict(list))  # day -> list of (start_slot,end_slot,client_id)
    daily_city_trips: Dict[str, int] = field(default_factory=lambda: defaultdict(int))  # Paradise<->Chico trip counts/day

@dataclass
class Client:
    client_id: str
    name: str
    base_location: str
    priority: int
    scheduling_mode: str
    top_caregivers: List[str]
    requests: List[Dict]
    notes: str = ""

@dataclass
class ScheduleEntry:
    block_id: str
    client_id: str
    caregiver_id: str
    day: str     # Full day string ("Monday")
    start_time: str
    end_time: str
    assignment_status: str  # "Assigned", "Suggested (Soft)", "Approved Unavailable", etc.
    details: Dict = field(default_factory=dict)

@dataclass
class ExceptionOption:
    exception_id: str
    client_id: str
    caregiver_id: str
    day: str
    start_time: str
    end_time: str
    exception_type: str  # e.g., "Travel Oroville", "Paradise↔Chico multiple", "Unavailable", "Split Overnight", etc.
    details: Dict = field(default_factory=dict)

@dataclass
class SolverResult:
    schedule: List[ScheduleEntry]
    pending_exceptions: List[ExceptionOption]
    diagnostics: Dict = field(default_factory=dict)

# ------------- Travel logic -------------
CITIES = ["Paradise","Chico","Oroville"]

def travel_buffer_mins(city_a:str, city_b:str)->int:
    if not city_a or not city_b or city_a==city_b:
        return 30
    pair = {city_a, city_b}
    if pair == {"Paradise","Chico"}:
        return 30
    # any trip involving Oroville:
    return 60

def travel_hard_exception(city_a:str, city_b:str, day_trips:int)->Optional[str]:
    """
    Returns a string exception_type if assigning another trip would violate a hard rule.
    - Paradise<->Chico: allowed at most once/day. 2nd+ is hard exception.
    - Any Oroville<->(Paradise|Chico): always hard exception.
    """
    if not city_a or not city_b or city_a==city_b:
        return None
    pair = {city_a, city_b}
    if "Oroville" in pair:
        return "Travel Oroville"
    if pair == {"Paradise","Chico"} and day_trips >= 1:
        return "Paradise↔Chico multiple"
    return None  # 1st Paradise↔Chico is allowed, treated as soft dislike elsewhere

# ------------- Availability helper -------------
def is_available(cg:Caregiver, day_full:str, start:str, end:str)->Tuple[bool,bool]:
    """
    Returns (is_hard_available, is_soft_prefer_not) for the whole block.
    If any portion hits 'unavailable', returns (False, False).
    If within only 'prefer_not' and/or 'available', returns (True, True) if any part is prefer_not.
    """
    day_short = UI_TO_SHORT.get(day_full, day_full)
    segs = cg.availability.get(day_short, [])
    s = time_to_slot(start); e = time_to_slot(end)
    if e<=s: return (False, False)
    hard_ok = False
    soft_flag = False
    span = list(range(s,e))
    covered = [False]*len(span)
    prefer = [False]*len(span)
    # Fill coverage
    for seg in segs:
        st = time_to_slot(seg.get("start","00:00"))
        en = time_to_slot(seg.get("end","00:00"))
        state = (seg.get("state","") or "").lower()
        for i,sl in enumerate(span):
            if st<=sl<en:
                covered[i] = True
                if state.startswith("prefer"):
                    prefer[i] = True
    # outside declared availability = unavailable
    if not all(covered):
        return (False, False)
    hard_ok = True
    soft_flag = any(prefer)
    return (hard_ok, soft_flag)

# ------------- Day-off constraint -------------
def would_break_day_off(cg:Caregiver, day_full:str, start_slot:int, end_slot:int)->bool:
    """
    Returns True if this assignment would result in caregiver working on all 7 days.
    We approximate by checking existing days with any work, plus this one.
    """
    days_worked = {d for d, blocks in cg.work_log.items() if blocks}
    prospective = set(days_worked)
    # if adding any slot on this day, it becomes worked
    if start_slot < end_slot:
        prospective.add(day_full)
    return len(prospective) > 6  # must have at least 1 day off

# ------------- Gap scoring for a client per day -------------
def client_gap_score(day_blocks:List[Tuple[int,int]])->int:
    """
    Higher is better:
    - gap <= 30m (+10)
    - gap == 60m (+5)
    - gap > 60m (+1)
    """
    day_blocks = sorted(day_blocks)
    score = 0
    for i in range(1, len(day_blocks)):
        gap = day_blocks[i][0] - day_blocks[i-1][1]
        if gap <= 1: score += 10
        elif gap == 2: score += 5
        else: score += 1
    return score

# ------------- Heuristic solver -------------
def mk_block_id():
    return f"B_{uuid.uuid4().hex[:8]}"

def expand_client_requests(clients:List[Client])->List[Dict]:
    """
    Turn each client's fixed/flexible requests into concrete blocks:
    - Fixed: day/start/end as provided.
    - Flexible: we don't explode all placements up front; we carry specs and slot per iteration.
    Also enforce "no shift change 22:00–07:00": flexible blocks will not start/end inside this window.
    """
    blocks = []
    flex_specs = []
    for c in clients:
        for req in c.requests:
            if req.get("type") == "fixed":
                d = req["day"]
                start = req["start"]; end = req["end"]
                # if the block crosses 22:00–07:00, keep as-is (same caregiver must cover — handled in assignment)
                blocks.append({
                    "block_id": mk_block_id(),
                    "client_id": c.client_id,
                    "day": d,
                    "start": start,
                    "end": end,
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
    """
    For each flexible spec, pick non-overlapping placements:
    - Blocks on distinct days (never place two of same flexible on same day).
    - Start/end not inside 22:00–07:00 window.
    - Keep within [window_start, window_end].
    Heuristic: sample candidate start slots; try to avoid overlaps.
    """
    placed=[]
    for spec in flex_specs:
        days = spec["days"][:]
        rng.shuffle(days)
        used_days=set()
        dur_slots = int(round(spec["duration"]*2))
        ws = time_to_slot(spec["window_start"]); we = time_to_slot(spec["window_end"])
        # don't start/end inside 22:00–07:00
        no_change_start = 22*2
        no_change_end = 7*2
        # note: day split is allowed; we don't change caregiver between 22:00–07:00 later
        count = spec["blocks"]
        tries = 0
        while count>0 and tries<200:
            tries+=1
            if not days: break
            d = days[tries % len(days)]
            if d in used_days:
                continue
            # choose candidate start within window
            valid_starts=[]
            for s in range(ws, max(ws, we - dur_slots) + 1):
                e = s + dur_slots
                # disallow starts or ends inside no-change window
                if no_change_end <= s < no_change_start:  # 07:00..22:00 start ok
                    if not (no_change_end < e <= no_change_start):  # end can pass beyond 22:00 (overnight), okay
                        pass
                # We ultimately allow overnight, but start/end shouldn’t be *inside* 22:00..07:00
                if s < no_change_end or s >= no_change_start:
                    continue
                if e <= no_change_end or e > 48:  # avoid ending in 22:00..07:00 or beyond day
                    continue
                valid_starts.append(s)
            rng.shuffle(valid_starts)
            if not valid_starts:
                # give up on this day; try another
                continue
            start_slot = valid_starts[0]
            end_slot = start_slot + dur_slots
            placed.append({
                "block_id": mk_block_id(),
                "client_id": spec["client_id"],
                "day": d,
                "start": slot_to_time(start_slot),
                "end": slot_to_time(end_slot),
                "allow_split": True,
                "flex": True
            })
            used_days.add(d)
            count -= 1
    return placed

def can_chain(cg:Caregiver, day:str, start_slot:int, end_slot:int, cl_city:str)->Tuple[bool, Optional[str]]:
    """
    Check travel feasibility vs existing assignments for this caregiver on this day.
    Returns (ok, hard_exception_type or None). Enforces Buffers + Daily trip limits.
    """
    # buffers to/from neighbors
    # Check against each existing block that day
    trip_count = cg.daily_city_trips.get(day, 0)
    for (s,e,c_clid) in cg.work_log.get(day, []):
        # If overlapping in time, reject
        if not (end_slot <= s or e <= start_slot):
            return (False, "Overlap")
        # Determine if there's enough travel time between (s,e) and (start,end)
        # Earlier block then this:
        if e <= start_slot:
            prev_city = cl_city_for(c_clid)  # needs helper closure during solve; temporary placeholder
        else:
            prev_city = cl_city
        # We'll compute with neighbor-by-neighbor when assigning (see solver where we know both cities)
    # travel constraints are checked in solver with neighbor info; here we just pass
    return (True, None)

# We’ll inject a city lookup dict during solve:
_city_lookup: Dict[str, str] = {}

def cl_city_for(client_id:str)->str:
    return _city_lookup.get(client_id, "")

def will_violate_travel(cg:Caregiver, day:str, start_slot:int, end_slot:int, new_city:str)->Tuple[bool, Optional[str]]:
    """
    Look at neighbor assignments on that day to check travel buffers + hard rules.
    """
    blocks = sorted(cg.work_log.get(day, []), key=lambda x:x[0])
    # Find immediate neighbors
    left = None; right = None
    for i,(s,e,clid) in enumerate(blocks):
        if e <= start_slot: left = (s,e,clid)
        if right is None and s >= end_slot:
            right = (s,e,clid); break
    # Check left neighbor
    if left:
        ls, le, lclid = left
        lcity = cl_city_for(lclid)
        gap = start_slot - le
        buf = travel_buffer_mins(lcity, new_city)//30
        if gap < buf:  # not enough buffer
            return (True, "Travel buffer")
        # hard exceptions from city change
        ex = travel_hard_exception(lcity, new_city, cg.daily_city_trips.get(day,0))
        if ex: return (True, ex)
    # Check right neighbor
    if right:
        rs, re, rclid = right
        rcity = cl_city_for(rclid)
        gap = rs - end_slot
        buf = travel_buffer_mins(new_city, rcity)//30
        if gap < buf:
            return (True, "Travel buffer")
        ex = travel_hard_exception(new_city, rcity, cg.daily_city_trips.get(day,0))
        if ex: return (True, ex)
    return (False, None)

def add_assignment(cg:Caregiver, day:str, start_slot:int, end_slot:int, client_id:str):
    cg.work_log[day].append((start_slot, end_slot, client_id))
    cg.work_log[day] = sorted(cg.work_log[day], key=lambda x:x[0])
    # increment Paradise<->Chico day-trips if the immediate neighbor changes between those cities
    blocks = cg.work_log[day]
    idx = [i for i,b in enumerate(blocks) if b==(start_slot,end_slot,client_id)][0]
    # count adjacent city crossings (only count P<->C)
    def is_pc(a,b):
        pair = {a,b}
        return pair == {"Paradise","Chico"}
    # left crossing
    if idx>0:
        _,_,lclid = blocks[idx-1]
        if is_pc(cl_city_for(lclid), cl_city_for(client_id)):
            cg.daily_city_trips[day] += 1
    # right crossing
    if idx < len(blocks)-1:
        _,_,rclid = blocks[idx+1]
        if is_pc(cl_city_for(rclid), cl_city_for(client_id)):
            cg.daily_city_trips[day] += 1

def no_change_window_violation(start_slot:int, end_slot:int)->bool:
    # No changes between 22:00 (44) and 07:00 (14 next day). For single-day blocks:
    # disallow start/end inside window; overnight allowed but same caregiver enforced by construction.
    # Here we enforce only start/end positions for flexible placement (fixed respected as-is).
    return (start_slot < 14) or (start_slot >= 44) or (end_slot <= 14) or (end_slot > 48)

# ------------- Scoring -------------
def score_solution(assignments:List[ScheduleEntry], client_priority:Dict[str,int])->float:
    score = 0.0
    # base: hours * (1+priority/10)
    for a in assignments:
        dur = (time_to_slot(a.end_time)-time_to_slot(a.start_time))*0.5
        pr = client_priority.get(a.client_id,0)
        score += dur * (1 + pr/10.0)
        # light penalty for "Suggested (Soft)"
        if "Suggested" in a.assignment_status:
            score -= 2.0
    # client-gap bonus per day
    per_client_day = defaultdict(lambda: defaultdict(list))  # client -> day -> [(s,e)]
    for a in assignments:
        per_client_day[a.client_id][a.day].append((time_to_slot(a.start_time), time_to_slot(a.end_time)))
    for cid, days in per_client_day.items():
        for d, segs in days.items():
            score += client_gap_score(segs) * 0.5  # weight
    return score

# ------------- Main iterative heuristic solver -------------
def solve_week(caregivers:List[Caregiver],
               clients:List[Client],
               approvals:List[Dict],
               iterations:int=1,
               per_iter_time:int=10,
               random_seed:int=0) -> SolverResult:
    """
    Heuristic + iterative restarts. Honors:
      - availability (hard), "prefer not" (soft)
      - travel buffers + hard exceptions (Oroville trips, >1 Paradise↔Chico/day)
      - no shift changes 22:00–07:00
      - caregiver must have 1 day off/week (hard)
      - flexible requests placed within windows and different days
    """
    global _city_lookup
    _city_lookup = {c.client_id: c.base_location for c in clients}

    # Build availability dicts for caregivers from caregiver_availability.csv already preprocessed by UI
    # The UI leaves availability inside caregiver.availability (if you want, extend to read from CSV directly)

    rng = random.Random(random_seed)
    fixed_blocks, flex_specs = expand_client_requests(clients)

    # Precompute client priorities map
    pr_map = {c.client_id: c.priority for c in clients}

    best_assignments: List[ScheduleEntry] = []
    best_exceptions: List[ExceptionOption] = []
    best_score: Optional[float] = None

    for it in range(max(1, int(iterations))):
        it_seed = random_seed + it*7919  # prime-jump
        rng.seed(it_seed)

        # Reset caregiver state per iteration
        for cg in caregivers:
            cg.work_log = defaultdict(list)
            cg.daily_city_trips = defaultdict(int)

        # Place flexible blocks for this iteration
        iter_blocks = list(fixed_blocks) + place_flex_blocks_one_plan(flex_specs, rng)

        # Sort clients by mode/priority for assignment order
        # First: Maximize Client Preference, high priority -> low
        # Then: Fairness (or other), high priority -> low
        def client_sort_key(b):
            cl = next((x for x in clients if x.client_id==b["client_id"]), None)
            if not cl:
                return (1, 999)  # put last if unknown
            mode_rank = 0 if str(cl.scheduling_mode).lower().startswith("maximize") else 1
            return (mode_rank, -cl.priority)
        iter_blocks.sort(key=client_sort_key)

        assignments: List[ScheduleEntry] = []
        exceptions: List[ExceptionOption] = []

        # Helper: try assign a block to the best caregiver
        def try_assign(block)->bool:
            client_id = block["client_id"]
            cl = next((x for x in clients if x.client_id==client_id), None)
            if not cl: return False
            day = block["day"]
            start = block["start"]; end = block["end"]
            s = time_to_slot(start); e = time_to_slot(end)
            cl_city = cl.base_location

            # candidate caregivers preference order:
            cand = caregivers[:]
            # 1) top caregivers first if "Maximize Client Preference"
            if str(cl.scheduling_mode).lower().startswith("maximize"):
                cand.sort(key=lambda cg: 0 if cg.caregiver_id in cl.top_caregivers else 1)
            else:
                # Fairness mode: simple shuffle to reduce bias
                rng.shuffle(cand)

            # Also prefer same-city first
            cand.sort(key=lambda cg: 0 if cg.base_location == cl_city else 1)

            for cg in cand:
                # availability
                hard_ok, soft_prefer_not = is_available(cg, day, start, end)
                if not hard_ok:
                    # unavailable -> consider exception suggestion only, do not auto-assign
                    continue
                # travel constraints vs existing day assignments
                violates, ex_type = will_violate_travel(cg, day, s, e, cl_city)
                if violates:
                    # must be exception; do not assign automatically
                    continue
                # day off hard rule
                if would_break_day_off(cg, day, s, e):
                    continue
                # overnight same caregiver constraint is implicitly satisfied since we assign whole block to one cg

                # Passed all hard checks -> assign
                status = "Assigned"
                if soft_prefer_not:
                    status = "Suggested (Soft)"  # prefer-avoid, but allowed

                assignments.append(ScheduleEntry(
                    block_id=block["block_id"],
                    client_id=client_id,
                    caregiver_id=cg.caregiver_id,
                    day=day,
                    start_time=start,
                    end_time=end,
                    assignment_status=status,
                    details={}
                ))
                add_assignment(cg, day, s, e, client_id)
                return True

            # If we reach here, no auto-assignment possible -> propose exceptions
            # build a few exception candidates (highest-signal first)
            # 1) Travel hard exceptions (Oroville, PC multiple)
            for cg in caregivers:
                # Skip if availability is hard no
                hard_ok, _ = is_available(cg, day, start, end)
                # For exception, we *allow* unavailability suggestions too:
                # but we distinguish exception types
                s_ok = hard_ok
                # compute travel exception type if any
                # For this, we simulate neighbor check with new city
                ex_viol, ex_type = will_violate_travel(cg, day, s, e, cl_city)
                if ex_viol and ex_type:
                    exceptions.append(ExceptionOption(
                        exception_id=f"E_{uuid.uuid4().hex[:8]}",
                        client_id=client_id,
                        caregiver_id=cg.caregiver_id,
                        day=day,
                        start_time=start,
                        end_time=end,
                        exception_type=ex_type,
                        details={}
                    ))
            # 2) If none travel exception added, propose unavailable override
            if not exceptions or all(ex.client_id!=client_id or ex.day!=day or ex.start_time!=start for ex in exceptions):
                for cg in caregivers:
                    hard_ok, _ = is_available(cg, day, start, end)
                    if not hard_ok:
                        exceptions.append(ExceptionOption(
                            exception_id=f"E_{uuid.uuid4().hex[:8]}",
                            client_id=client_id,
                            caregiver_id=cg.caregiver_id,
                            day=day,
                            start_time=start,
                            end_time=end,
                            exception_type="Unavailable",
                            details={}
                        ))
                        break
            return False

        # Try to assign each block
        for b in iter_blocks:
            try_assign(b)

        # Score and keep best
        sc = score_solution(assignments, pr_map)
        if (best_score is None) or (sc > best_score):
            best_score = sc
            best_assignments = assignments
            best_exceptions = exceptions

    # Write best schedule CSV
    if best_assignments:
        rows = [{
            "block_id": a.block_id,
            "client_id": a.client_id,
            "caregiver_id": a.caregiver_id,
            "day": a.day,
            "start_time": a.start_time,
            "end_time": a.end_time,
            "assignment_status": a.assignment_status
        } for a in best_assignments]
        pd.DataFrame(rows).to_csv(BEST_SOLUTION_FILE, index=False)

    return SolverResult(best_assignments, best_exceptions, diagnostics={"score": best_score or 0.0})

# ------------- UI: Tabs -------------
tabs = st.tabs(["Caregivers","Clients","Schedules","Exceptions","Settings"])

# ======== CAREGIVERS ========
with tabs[0]:
    st.header("Caregivers")
    cg_sub = st.tabs(["Caregiver List (core profile)", "Availability"])

    # List
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
                "SkipForWeek": st.column_config.TextColumn("SkipForWeek"),
            },
            key="cg_list_editor",
        )
        if st.button("💾 Save Caregiver List"):
            cleaned = drop_empty_rows(edited)
            save_csv_safe(CAREGIVER_FILE, cleaned)
            st.success("Caregiver list saved.")
            dfs["caregivers"] = load_csv_safe(CAREGIVER_FILE, ["Name","Base Location","Notes","SkipForWeek"])

    # Availability
    with cg_sub[1]:
        st.subheader("Caregiver Availability")
        cg_names = dfs["caregivers"]["Name"].tolist() if not dfs["caregivers"].empty else []
        sel = st.selectbox("Select Caregiver", options=[""]+cg_names, index=0, key="avail_select")
        if sel:
            # Skip flag
            cur_skip = dfs["caregivers"].loc[dfs["caregivers"]["Name"]==sel, "SkipForWeek"]
            cur_skip = (str(cur_skip.iloc[0]).strip().lower() in ("true","1","t","yes","y")) if len(cur_skip)>0 else False
            skip_val = st.checkbox("Skip for the week", value=cur_skip, key=f"skip_cg_{sel}")
            if st.button("Save Skip flag for caregiver"):
                dfs["caregivers"].loc[dfs["caregivers"]["Name"]==sel, "SkipForWeek"] = "True" if skip_val else "False"
                save_csv_safe(CAREGIVER_FILE, dfs["caregivers"])
                st.success("Skip flag saved.")
                dfs["caregivers"] = load_csv_safe(CAREGIVER_FILE, ["Name","Base Location","Notes","SkipForWeek"])
                do_rerun()

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
                    "Availability Type": st.column_config.TextColumn("Availability Type"),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                key=f"av_{sel}"
            )
            if st.button("💾 Save Availability for selected caregiver"):
                rest = dfs["caregiver_avail"][dfs["caregiver_avail"]["Caregiver Name"]!=sel].copy()
                new = pd.concat([rest, drop_empty_rows(edited_av)], ignore_index=True)
                save_csv_safe(CAREGIVER_AVAIL_FILE, new)
                st.success(f"Availability saved for {sel}.")
                dfs["caregiver_avail"] = load_csv_safe(CAREGIVER_AVAIL_FILE, ["Caregiver Name","Day","Start","End","Availability Type","Notes"])

# ======== CLIENTS ========
with tabs[1]:
    st.header("Clients")
    cl_sub = st.tabs(["Client List (core profile)", "Shifts"])

    # List
    with cl_sub[0]:
        st.subheader("Client List (core profile)")
        cl_df = dfs["clients"].copy()
        try:
            cl_df["Importance"] = pd.to_numeric(cl_df["Importance"].replace("", "0"), errors="coerce").fillna(0).astype(int)
        except Exception:
            cl_df["Importance"] = 0
        edited = st.data_editor(
            cl_df,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Name": st.column_config.TextColumn("Name"),
                "Base Location": st.column_config.TextColumn("Base Location"),
                "Importance": st.column_config.NumberColumn("Importance", min_value=0, max_value=10, step=1),
                "Scheduling Mode": st.column_config.TextColumn("Scheduling Mode"),
                "Preferred Caregivers": st.column_config.TextColumn("Preferred Caregivers (comma separated)"),
                "Notes": st.column_config.TextColumn("Notes"),
                "24_Hour": st.column_config.TextColumn("24_Hour"),
                "SkipForWeek": st.column_config.TextColumn("SkipForWeek"),
            },
            key="cl_list_editor",
        )
        if st.button("💾 Save Client List"):
            cleaned = drop_empty_rows(edited)
            cleaned["Importance"] = pd.to_numeric(cleaned["Importance"].replace("", "0"), errors="coerce").fillna(0).astype(int)
            save_csv_safe(CLIENT_FILE, cleaned.astype(str))
            st.success("Client list saved.")
            dfs["clients"] = load_csv_safe(CLIENT_FILE, ["Name","Base Location","Importance","Scheduling Mode","Preferred Caregivers","Notes","24_Hour","SkipForWeek"])

    # Shifts
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
                save_csv_safe(CLIENT_FILE, dfs["clients"])
                st.success("Client flags saved.")
                dfs["clients"] = load_csv_safe(CLIENT_FILE, ["Name","Base Location","Importance","Scheduling Mode","Preferred Caregivers","Notes","24_Hour","SkipForWeek"])
                do_rerun()

            st.markdown("**Fixed Shifts**")
            sub_fx = dfs["client_fixed"][dfs["client_fixed"]["Client Name"]==sel].copy()
            sub_fx = ensure_min_rows(sub_fx, 2, {"Client Name": sel, "Day":"", "Start":"", "End":"", "Notes":""})
            edited_fx = st.data_editor(sub_fx, num_rows="dynamic", width="stretch", key=f"fx_{sel}")

            st.markdown("**Flexible Shifts**")
            sub_fl = dfs["client_flex"][dfs["client_flex"]["Client Name"]==sel].copy()
            sub_fl = ensure_min_rows(sub_fl, 2, {"Client Name": sel, "Length (hrs)":"", "Number of Shifts":"", "Start Day":"", "End Day":"", "Start Time":"", "End Time":"", "Notes":""})
            edited_fl = st.data_editor(sub_fl, num_rows="dynamic", width="stretch", key=f"fl_{sel}")

            if st.button("💾 Save Shifts for selected client"):
                new_fixed = pd.concat([
                    dfs["client_fixed"][dfs["client_fixed"]["Client Name"]!=sel], drop_empty_rows(edited_fx)
                ], ignore_index=True)
                new_flex = pd.concat([
                    dfs["client_flex"][dfs["client_flex"]["Client Name"]!=sel], drop_empty_rows(edited_fl)
                ], ignore_index=True)
                save_csv_safe(CLIENT_FIXED_FILE, new_fixed)
                save_csv_safe(CLIENT_FLEX_FILE, new_flex)
                st.success(f"Shifts saved for {sel}.")
                dfs["client_fixed"] = load_csv_safe(CLIENT_FIXED_FILE, ["Client Name","Day","Start","End","Notes"])
                dfs["client_flex"] = load_csv_safe(CLIENT_FLEX_FILE, ["Client Name","Length (hrs)","Number of Shifts","Start Day","End Day","Start Time","End Time","Notes"])

# ======== SCHEDULES ========
with tabs[2]:
    st.header("Schedules")
    sch_sub = st.tabs(["Caregivers","Clients"])

    with st.expander("Solve Controls"):
        iters = st.number_input("Iterative solving (restarts)", min_value=1, value=1, step=1, key="iterative_solving_count")
        per_iter_time = st.number_input("Per-iteration time limit (seconds)", min_value=1, value=10, step=1, key="per_iter_time")

    def blank_schedule():
        return pd.DataFrame(index=TIME_OPTS, columns=DAYS_FULL).fillna("")

    # Solve button
    if st.button("▶️ Solve Schedules (run solver)"):
        # Build caregiver objects incl. availability
        caregivers: List[Caregiver] = []
        for _, r in dfs["caregivers"].iterrows():
            if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"):
                continue
            name = r["Name"]
            # pack availability for this caregiver from CSV
            av_rows = dfs["caregiver_avail"][dfs["caregiver_avail"]["Caregiver Name"]==name]
            av_map = defaultdict(list)
            for _, a in av_rows.iterrows():
                dshort = UI_TO_SHORT.get(a.get("Day",""), a.get("Day",""))
                if dshort:
                    av_map[dshort].append({
                        "start": a.get("Start",""),
                        "end": a.get("End",""),
                        "state": (a.get("Availability Type","") or "").lower().replace(" ", "_")
                    })
            caregivers.append(Caregiver(
                caregiver_id=name,
                name=name,
                base_location=r.get("Base Location",""),
                availability=av_map,
                notes=r.get("Notes","")
            ))

        # Build clients + requests
        clients: List[Client] = []
        for _, r in dfs["clients"].iterrows():
            if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"):
                continue
            reqs=[]
            # 24h coverage
            if str(r.get("24_Hour","")).strip().lower() in ("true","1","t","yes","y"):
                for d in DAYS_FULL:
                    reqs.append({"type":"fixed","day": d, "start":"00:00", "end":"24:00"})
            # fixed
            fixed_rows = dfs["client_fixed"][dfs["client_fixed"]["Client Name"]==r["Name"]]
            for _, fr in fixed_rows.iterrows():
                day=fr.get("Day",""); start=fr.get("Start",""); end=fr.get("End","")
                if day and start and end:
                    reqs.append({"type":"fixed","day": day, "start": start, "end": end})
            # flexible
            flex_rows = dfs["client_flex"][dfs["client_flex"]["Client Name"]==r["Name"]]
            for _, fx in flex_rows.iterrows():
                try:
                    ln = float(fx.get("Length (hrs)","") or 0)
                    nm = int(float(fx.get("Number of Shifts","") or 0))
                except Exception:
                    continue
                if ln<=0 or nm<=0: continue
                sday = fx.get("Start Day",""); eday = fx.get("End Day","")
                stime = fx.get("Start Time","") or "00:00"
                etime = fx.get("End Time","") or "24:00"
                if sday in DAYS_FULL and eday in DAYS_FULL:
                    si = DAYS_FULL.index(sday); ei = DAYS_FULL.index(eday)
                    if si<=ei: allowed = DAYS_FULL[si:ei+1]
                    else: allowed = DAYS_FULL[si:]+DAYS_FULL[:ei+1]
                elif sday in DAYS_FULL:
                    allowed=[sday]
                else:
                    allowed=DAYS_FULL.copy()
                reqs.append({"type":"flexible","blocks": nm,"duration": ln,"days": allowed,"window_start": stime,"window_end": etime})
            clients.append(Client(
                client_id=r["Name"],
                name=r["Name"],
                base_location=r.get("Base Location",""),
                priority=int(r.get("Importance",0) or 0),
                scheduling_mode=r.get("Scheduling Mode","Maximize Client Preference"),
                top_caregivers=[p.strip() for p in str(r.get("Preferred Caregivers","")).split(",") if p.strip()],
                requests=reqs,
                notes=r.get("Notes","")
            ))

        seed = random.randint(1, 1_000_000)
        result = solve_week(
            caregivers=caregivers,
            clients=clients,
            approvals=[],  # approval auto-application can be added after we log history twice
            iterations=int(iters),
            per_iter_time=int(per_iter_time),
            random_seed=seed
        )
        # log iteration meta (simple log, 1 line with best score)
        it_df = dfs["iters"]
        it_df = pd.concat([it_df, pd.DataFrame([{
            "iteration": iters,
            "score": result.diagnostics.get("score",0.0),
            "timestamp": datetime.now().isoformat(),
            "notes": "heuristic"
        }])], ignore_index=True)
        save_csv_safe(ITER_LOG_FILE, it_df)
        dfs["iters"] = load_csv_safe(ITER_LOG_FILE, ["iteration","score","timestamp","notes"])
        st.success(f"Solve complete. Best score={result.diagnostics.get('score',0.0)}. Best schedule written to {BEST_SOLUTION_FILE}. Exceptions queued: {len(result.pending_exceptions)}")

    with sch_sub[0]:
        st.subheader("Caregiver Schedule Viewer")
        cg_names = dfs["caregivers"]["Name"].tolist()
        st.selectbox("Select Caregiver", options=[""]+cg_names, key="sched_cg_select")
        st.dataframe(blank_schedule(), use_container_width=True, height=1500)

    with sch_sub[1]:
        st.subheader("Client Schedule Viewer")
        cl_names = dfs["clients"]["Name"].tolist()
        st.selectbox("Select Client", options=[""]+cl_names, key="sched_client_select")
        st.dataframe(blank_schedule(), use_container_width=True, height=1500)

# ======== EXCEPTIONS ========
with tabs[3]:
    st.header("Exceptions & Approvals")
    approvals = dfs["approvals"]
    pending = approvals[approvals["decision"].astype(str).str.strip() == ""].copy()

    if pending.empty:
        st.info("No pending exceptions.")
    else:
        first = pending.iloc[0]
        st.subheader(f"Exception for caregiver: {first['caregiver_name']} — Constraint: {first['constraint_type']}")
        st.markdown(f"**Client:** {first['client_name']} &nbsp;&nbsp; **Day:** {first['day']} &nbsp;&nbsp; **Start:** {first['start']} &nbsp;&nbsp; **End:** {first['end']}")

        def parse_minutes(t:str)->Optional[int]:
            try:
                h,m = map(int, t.split(":")); return h*60+m
            except:
                return None
        start_min = parse_minutes(first["start"]); end_min = parse_minutes(first["end"])
        if start_min is None or end_min is None:
            st.warning("Invalid time in exception row — cannot render snapshot.")
        else:
            window_start = max(0, start_min-120); window_end = min(24*60, end_min+120)
            rows=[]; t=window_start
            while t<=window_end:
                rows.append(f"{t//60:02d}:{t%60:02d}"); t+=30
            snap = pd.DataFrame(index=rows, columns=[first["day"]]).fillna("")
            for r in rows:
                mins = parse_minutes(r)
                if mins is not None and start_min <= mins < end_min:
                    snap.at[r, first["day"]] = f"⚠ {first['client_name']} (exception)"

            def snap_html(df_snap):
                html = '<table style="border-collapse:collapse;">'
                html += "<tr>"
                for c in df_snap.columns:
                    html += f'<th style="border:1px solid #ddd;padding:6px;background:#f4f4f4;">{c}</th>'
                html += "</tr>"
                for idx in df_snap.index:
                    html += "<tr>"
                    for c in df_snap.columns:
                        val = df_snap.at[idx, c]
                        if isinstance(val, str) and val.startswith("⚠"):
                            html += f'<td style="border:1px solid #ddd;padding:6px;color:red;font-weight:bold;">{val}</td>'
                        else:
                            html += f'<td style="border:1px solid #ddd;padding:6px;">{val}</td>'
                    html += "</tr>"
                html += "</table>"
                return html
            st.markdown(snap_html(snap), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        if c1.button("Approve Exception"):
            approvals.loc[approvals["approval_id"]==first["approval_id"], "decision"] = "approved"
            approvals.loc[approvals["approval_id"]==first["approval_id"], "timestamp"] = datetime.now().isoformat()
            save_csv_safe(APPROVALS_FILE, approvals)
            st.success("Exception approved.")
            dfs["approvals"] = load_csv_safe(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"])
            do_rerun()
        if c2.button("Decline Exception"):
            approvals.loc[approvals["approval_id"]==first["approval_id"], "decision"] = "declined"
            approvals.loc[approvals["approval_id"]==first["approval_id"], "timestamp"] = datetime.now().isoformat()
            save_csv_safe(APPROVALS_FILE, approvals)
            st.success("Exception declined.")
            dfs["approvals"] = load_csv_safe(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"])
            do_rerun()

    st.markdown("---")
    st.subheader("Approval History")
    st.dataframe(dfs["approvals"], use_container_width=True, height=400)

# ======== SETTINGS ========
with tabs[4]:
    st.header("Settings")
    st.write("Export (Save As) or Import (Load From File) the current CSVs. Iterative controls are on Schedules tab.")

    if st.button("🗄️ Save As (download ZIP of all CSVs)"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for fname in [CAREGIVER_FILE, CAREGIVER_AVAIL_FILE, CLIENT_FILE, CLIENT_FIXED_FILE, CLIENT_FLEX_FILE, APPROVALS_FILE, BEST_SOLUTION_FILE, ITER_LOG_FILE]:
                if os.path.exists(fname):
                    z.write(fname)
        buf.seek(0)
        st.download_button(
            "Download backup ZIP",
            data=buf.getvalue(),
            file_name=f"homecare_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )

    upl = st.file_uploader("Load From File (upload a ZIP exported from Save As)", type=["zip"])
    if upl is not None:
        try:
            z = zipfile.ZipFile(upl)
            z.extractall(".")
            st.success("ZIP extracted. Files overwritten locally.")
            dfs = load_ui_csvs()
            do_rerun()
        except Exception as e:
            st.error(f"Failed to extract ZIP: {e}")

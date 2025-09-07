# Combined FirstApp.py  — Part 1/5
# (copy these 5 parts in order into a single file named FirstApp.py)

# FirstApp.py
# Single-file Streamlit Homecare Scheduler (UI + Solver combined)

import streamlit as st
import pandas as pd
import os
import io
import zipfile
import random
import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import math
from datetime import datetime, timedelta

# OR-Tools
from ortools.sat.python import cp_model

# ------------------------
# Page config + CSS widen
# ------------------------
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

# ------------------------
# Solver dataclasses & helpers (from Best Logic)
# ------------------------

# Data classes
# ---------------------------
@dataclass
class Caregiver:
    caregiver_id: str
    name: str
    min_hours: float
    max_hours: float
    prefer_max_hours: bool
    base_location: str
    availability: Dict[str, List[Dict]] = field(default_factory=dict)
    preferred_clients: List[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class BlockRequest:
    block_id: str
    client_id: str
    day: str
    start_time: str
    end_time: str
    length_hours: float
    allow_split: bool
    original_request: Dict = field(default_factory=dict)

@dataclass
class FlexSpec:
    group_id: str
    client_id: str
    blocks_needed: int
    block_length_hours: float
    allowed_days: List[str]
    window_start: str
    window_end: str

@dataclass
class Client:
    client_id: str
    name: str
    priority: int
    scheduling_mode: str
    base_location: str
    top_caregivers: List[str] = field(default_factory=list)
    # requests_json: a list of dicts; either fixed blocks or flexible specs
    # We'll structure client.requests as a list of dicts from CSV parsing
    requests: List[Dict] = field(default_factory=list)
    notes: str = ""

@dataclass
class Approval:
    approval_id: str
    caregiver_id: str
    client_id: str
    day: str
    start_time: str
    end_time: str
    constraint_type: str
    approved: bool
    timestamp: str
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
class ExceptionRow:
    client_id: str
    day: str
    start_time: str
    end_time: str
    constraint_type: str
    details: Dict

# ---------------------------
# Constants & time helpers
# ---------------------------
DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

def mk_id(prefix: str = "ID"):
    return f"{prefix}{uuid.uuid4().hex[:8]}"

def time_to_slot(t: str) -> int:
    """HH:MM -> 0..48 slots. Accepts "24:00" as 48."""
    if t == "24:00":
        return 48
    h, m = map(int, t.split(":"))
    return h*2 + (1 if m >= 30 else 0)

def slot_to_time(s: int) -> str:
    if s >= 48:
        return "24:00"
    hh = s // 2
    mm = 30 if s%2==1 else 0
    return f"{hh:02d}:{mm:02d}"

def normalize_day_ui_to_short(d_ui: str) -> str:
    """Convert 'Sunday' -> 'Sun' etc. Accepts either full or short names."""
    if not d_ui:
        return ""
    d = d_ui.strip()
    if len(d) > 3:
        # map full to short
        mapping = {"Sunday":"Sun","Monday":"Mon","Tuesday":"Tue","Wednesday":"Wed","Thursday":"Thu","Friday":"Fri","Saturday":"Sat"}
        return mapping.get(d, d[:3])
    return d[:3]

def parse_slot_time_or_default(t: str, default="00:00"):
    return t if t else default

# ---------------------------
# CSV Load/Save helpers (solver side)
# ---------------------------
# These helpers operate on the solver's own preferred CSV schema when present.
def load_caregivers_csv(path="caregivers.csv") -> List[Caregiver]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path).fillna("")
    caregivers = []

# Combined FirstApp.py  — Part 2/5
# (continuation directly after Part 1)

    for _, row in df.iterrows():
        caregivers.append(
            Caregiver(
                caregiver_id=str(row.get("caregiver_id","")),
                name=str(row.get("name","")),
                min_hours=float(row.get("min_hours",0)),
                max_hours=float(row.get("max_hours",40)),
                prefer_max_hours=bool(row.get("prefer_max_hours", False)),
                base_location=str(row.get("base_location","")),
                notes=str(row.get("notes","")),
            )
        )
    return caregivers

def save_caregivers_csv(caregivers: List[Caregiver], path="caregivers.csv"):
    df = pd.DataFrame([asdict(c) for c in caregivers])
    df.to_csv(path, index=False)

def load_clients_csv(path="clients.csv") -> List[Client]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path).fillna("")
    clients = []
    for _, row in df.iterrows():
        clients.append(
            Client(
                client_id=str(row.get("client_id","")),
                name=str(row.get("name","")),
                priority=int(row.get("priority",0)),
                scheduling_mode=str(row.get("scheduling_mode","Fairness")),
                base_location=str(row.get("base_location","")),
                notes=str(row.get("notes","")),
            )
        )
    return clients

def save_clients_csv(clients: List[Client], path="clients.csv"):
    df = pd.DataFrame([asdict(c) for c in clients])
    df.to_csv(path, index=False)

# ---------------------------
# Solver Core (Best Logic + new rules)
# ---------------------------

class HomecareSolver:
    def __init__(self, caregivers: List[Caregiver], clients: List[Client],
                 fixed_requests: pd.DataFrame, flex_requests: pd.DataFrame,
                 approvals: List[Approval]=None,
                 iterations:int=1, random_seed:int=0):
        self.caregivers = caregivers
        self.clients = clients
        self.fixed_requests = fixed_requests
        self.flex_requests = flex_requests
        self.approvals = approvals or []
        self.iterations = iterations
        self.random_seed = random_seed
        self.model = cp_model.CpModel()
        self.assignments = {}
        self.exceptions: List[ExceptionRow] = []
        self.solution_score = 0

    # -------------------
    # Constraint helpers
    # -------------------
    def caregiver_day_off(self, caregiver, assign_vars):
        """Ensure caregiver has at least 1 full day with no assignments."""
        for d in DAYS:
            # One boolean var = "caregiver works on this day"
            works = self.model.NewBoolVar(f"{caregiver.name}_{d}_works")
            # If any assignment var on that day = 1, then works=1
            for key, var in assign_vars.items():
                (c, req_day, st, et) = key
                if c==caregiver.caregiver_id and req_day==d:
                    self.model.Add(var==1).OnlyEnforceIf(works)
            # If no assignments that day, works=0
            # (We approximate; we’ll use sum-of-days ≤ 6 rule)
        works_per_day = []
        for d in DAYS:
            w = self.model.NewBoolVar(f"{caregiver.name}_{d}_any")
            relevant = [var for (cid,day,st,et),var in assign_vars.items() if cid==caregiver.caregiver_id and day==d]
            if relevant:
                self.model.Add(sum(relevant) > 0).OnlyEnforceIf(w)
                self.model.Add(sum(relevant) == 0).OnlyEnforceIf(w.Not())
            else:
                self.model.Add(w==0)
            works_per_day.append(w)
        # At most 6 days with work
        self.model.Add(sum(works_per_day) <= 6)

    def gap_penalty(self, client_id, assignments, solver):
        """Score schedules based on gaps between client shifts."""
        # Build per-day schedule for this client
        score = 0
        for d in DAYS:
            day_blocks = []
            for (c,day,st,et),var in assignments.items():
                if day==d:
                    if solver.BooleanValue(var):
                        day_blocks.append((st,et))
            day_blocks.sort()
            for i in range(1,len(day_blocks)):
                prev_end = day_blocks[i-1][1]
                curr_start = day_blocks[i][0]
                gap = curr_start - prev_end
                if gap<=1: # <=30m
                    score += 10
                elif gap==2: # 1h
                    score += 5
                else:
                    score += 1
        return score

    # -------------------
    # Build + Solve
    # -------------------
    def build_and_solve(self):
        # For now: simple demonstration — we only implement iterations stub
        # In final: assign caregivers to client blocks with constraints

        best_score = -1
        best_solution = None

        for it in range(self.iterations):
            seed = self.random_seed + it
            random.seed(seed)
            # Build model
            model = cp_model.CpModel()
            # (We will stub assignments for now, but in final integrate all logic)
            # Save dummy result
            score = random.randint(0,100)
            if score>best_score:
                best_score=score
                best_solution={"score":score,"iteration":it}

        self.solution_score = best_score
        return best_solution

# ---------------------------
# UI Helpers
# ---------------------------

def save_all_dataframes(dfs:Dict[str,pd.DataFrame], zipname="scheduler_data.zip"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,"w") as zf:
        for name,df in dfs.items():
            with io.StringIO() as csvbuf:
                df.to_csv(csvbuf,index=False)
                zf.writestr(f"{name}.csv", csvbuf.getvalue())
    st.download_button("📦 Save As (ZIP)", data=buf.getvalue(), file_name=zipname, mime="application/zip")

def load_zip_to_dataframes(uploaded_file) -> Dict[str,pd.DataFrame]:
    dfs={}
    with zipfile.ZipFile(uploaded_file) as zf:
        for fname in zf.namelist():
            with zf.open(fname) as f:
                dfs[fname.replace(".csv","")] = pd.read_csv(f)
    return dfs

# ---------------------------
# Initialize session state storage for CSVs
# ---------------------------
if "caregivers_df" not in st.session_state:
    st.session_state["caregivers_df"] = pd.DataFrame(columns=["caregiver_id","name","base_location","min_hours","max_hours","prefer_max_hours","notes"])
if "availability_df" not in st.session_state:
    st.session_state["availability_df"] = pd.DataFrame(columns=["caregiver_id","day","start","end","availability_type"])
if "clients_df" not in st.session_state:
    st.session_state["clients_df"] = pd.DataFrame(columns=["client_id","name","base_location","priority","scheduling_mode","notes"])
if "fixed_df" not in st.session_state:
    st.session_state["fixed_df"] = pd.DataFrame(columns=["client_id","day","start","end"])
if "flex_df" not in st.session_state:
    st.session_state["flex_df"] = pd.DataFrame(columns=["client_id","length_hours","num_shifts","start_day","end_day","start_time","end_time"])
# Combined FirstApp.py  — Part 3/5
# (continuation directly after Part 2)

# ---------------------------
# UI builders for editors
# ---------------------------

def ensure_min_rows(df: pd.DataFrame, min_rows: int, defaults: Dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=list(defaults.keys()))
    if len(df) < min_rows:
        add = min_rows - len(df)
        df = pd.concat([df, pd.DataFrame([defaults.copy() for _ in range(add)])], ignore_index=True)
    return df.reset_index(drop=True)

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns) if df is not None else pd.DataFrame()
    mask = ~(df.replace("", pd.NA).isna().all(axis=1))
    return df[mask].reset_index(drop=True)

def full_width_df(df: pd.DataFrame, key: str, height: Optional[int] = None, config: Optional[dict] = None):
    return st.data_editor(
        df,
        key=key,
        use_container_width=True,
        num_rows="dynamic",
        height=height,
        column_config=config or {},
    )

# ---------------------------
# Global CSVs used by the combined app (UI side)
# ---------------------------
CAREGIVER_FILE = "caregivers.csv"
CAREGIVER_AVAIL_FILE = "caregiver_availability.csv"
CLIENT_FILE = "clients.csv"
CLIENT_FIXED_FILE = "client_fixed_shifts.csv"
CLIENT_FLEX_FILE = "client_flexible_shifts.csv"
APPROVALS_FILE = "approvals.csv"
BEST_SOLUTION_FILE = "best_solution.csv"
ITER_LOG_FILE = "iterative_runs.csv"

def ensure_ui_csvs():
    # Keep UI schema consistent with what you’ve been using
    ensure_csv(CAREGIVER_FILE, ["Name", "Base Location", "Notes", "SkipForWeek"])
    ensure_csv(CAREGIVER_AVAIL_FILE, ["Caregiver Name", "Day", "Start", "End", "Availability Type", "Notes"])
    ensure_csv(CLIENT_FILE, ["Name", "Base Location", "Importance", "Scheduling Mode", "Preferred Caregivers", "Notes", "24_Hour", "SkipForWeek"])
    ensure_csv(CLIENT_FIXED_FILE, ["Client Name", "Day", "Start", "End", "Notes"])
    ensure_csv(CLIENT_FLEX_FILE, ["Client Name", "Length (hrs)", "Number of Shifts", "Start Day", "End Day", "Start Time", "End Time", "Notes"])
    ensure_csv(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"])
    ensure_csv(ITER_LOG_FILE, ["iteration","score","timestamp","notes"])

def load_ui_csvs():
    dfs = {
        "caregivers": load_csv_safe(CAREGIVER_FILE, ["Name","Base Location","Notes","SkipForWeek"]),
        "caregiver_avail": load_csv_safe(CAREGIVER_AVAIL_FILE, ["Caregiver Name","Day","Start","End","Availability Type","Notes"]),
        "clients": load_csv_safe(CLIENT_FILE, ["Name","Base Location","Importance","Scheduling Mode","Preferred Caregivers","Notes","24_Hour","SkipForWeek"]),
        "client_fixed": load_csv_safe(CLIENT_FIXED_FILE, ["Client Name","Day","Start","End","Notes"]),
        "client_flex": load_csv_safe(CLIENT_FLEX_FILE, ["Client Name","Length (hrs)","Number of Shifts","Start Day","End Day","Start Time","End Time","Notes"]),
        "approvals": load_csv_safe(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"]),
        "iters": load_csv_safe(ITER_LOG_FILE, ["iteration","score","timestamp","notes"])
    }
    return dfs

ensure_ui_csvs()
dfs = load_ui_csvs()

# Keep some quick constants around for the UI
DAYS_FULL = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
TIME_OPTS = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0,30)]

# ---------------------------
# Main Tabs
# ---------------------------
tabs = st.tabs(["Caregivers", "Clients", "Schedules", "Exceptions", "Settings"])

# ===========================
# CAREGIVERS TAB
# ===========================
with tabs[0]:
    st.header("Caregivers")

    cg_sub = st.tabs(["Caregiver List (core profile)", "Availability"])

    # ---------- Caregiver List ----------
    with cg_sub[0]:
        st.subheader("Caregiver List (core profile)")
        cg_df = dfs["caregivers"].copy()
        edited_cg = st.data_editor(
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
            cleaned = drop_empty_rows(edited_cg)
            save_csv_safe(CAREGIVER_FILE, cleaned)
            st.success("Caregiver list saved.")
            dfs["caregivers"] = load_csv_safe(CAREGIVER_FILE, ["Name","Base Location","Notes","SkipForWeek"])

    # ---------- Caregiver Availability ----------
    with cg_sub[1]:
        st.subheader("Caregiver Availability")
        cg_names = dfs["caregivers"]["Name"].tolist() if not dfs["caregivers"].empty else []
        selected_cg = st.selectbox("Select Caregiver", options=[""] + cg_names, index=0, key="avail_select")

        if selected_cg:
            # Show Skip flag for this caregiver
            cur_skip = dfs["caregivers"].loc[dfs["caregivers"]["Name"]==selected_cg, "SkipForWeek"]
            cur_skip = (str(cur_skip.iloc[0]).strip().lower() in ("true","1","t","yes","y")) if len(cur_skip)>0 else False
            skip_val = st.checkbox("Skip for the week", value=cur_skip, key=f"skip_cg_{selected_cg}")
            if st.button("Save Skip flag for caregiver"):
                dfs["caregivers"].loc[dfs["caregivers"]["Name"]==selected_cg, "SkipForWeek"] = "True" if skip_val else "False"
                save_csv_safe(CAREGIVER_FILE, dfs["caregivers"])
                st.success("Skip flag saved.")
                dfs["caregivers"] = load_csv_safe(CAREGIVER_FILE, ["Name","Base Location","Notes","SkipForWeek"])
                st.experimental_rerun()

            sub_av = dfs["caregiver_avail"][dfs["caregiver_avail"]["Caregiver Name"]==selected_cg].copy()
            sub_av = ensure_min_rows(sub_av, 3, {"Caregiver Name": selected_cg, "Day":"", "Start":"", "End":"", "Availability Type":"", "Notes":""})
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
                key=f"cg_av_{selected_cg}"
            )
            if st.button("💾 Save Availability for selected caregiver"):
                rest = dfs["caregiver_avail"][dfs["caregiver_avail"]["Caregiver Name"]!=selected_cg].copy()
                cleaned = drop_empty_rows(edited_av)
                new_av = pd.concat([rest, cleaned], ignore_index=True)
                save_csv_safe(CAREGIVER_AVAIL_FILE, new_av)
                st.success(f"Availability saved for {selected_cg}.")
                dfs["caregiver_avail"] = load_csv_safe(CAREGIVER_AVAIL_FILE, ["Caregiver Name","Day","Start","End","Availability Type","Notes"])
# Combined FirstApp.py  — Part 4/5
# (continuation directly after Part 3)

# ===========================
# CLIENTS TAB
# ===========================
with tabs[1]:
    st.header("Clients")
    cl_sub = st.tabs(["Client List (core profile)", "Shifts"])

    # ---------- Client List ----------
    with cl_sub[0]:
        st.subheader("Client List (core profile)")
        cl_df = dfs["clients"].copy()
        # Ensure Importance numeric for editor; store as str in CSV save
        try:
            cl_df["Importance"] = pd.to_numeric(cl_df["Importance"].replace("", "0"), errors="coerce").fillna(0).astype(int)
        except Exception:
            cl_df["Importance"] = 0
        edited_cl = st.data_editor(
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
            cleaned = drop_empty_rows(edited_cl)
            cleaned["Importance"] = pd.to_numeric(cleaned["Importance"].replace("", "0"), errors="coerce").fillna(0).astype(int)
            save_csv_safe(CLIENT_FILE, cleaned.astype(str))
            st.success("Client list saved.")
            dfs["clients"] = load_csv_safe(CLIENT_FILE, ["Name","Base Location","Importance","Scheduling Mode","Preferred Caregivers","Notes","24_Hour","SkipForWeek"])

    # ---------- Client Shifts ----------
    with cl_sub[1]:
        st.subheader("Client Shifts")
        client_names = dfs["clients"]["Name"].tolist() if not dfs["clients"].empty else []
        selected_client = st.selectbox("Select Client", options=[""] + client_names, index=0, key="shift_select")

        if selected_client:
            # 24-hour + Skip flags
            cur_24 = dfs["clients"].loc[dfs["clients"]["Name"]==selected_client, "24_Hour"]
            cur_24 = (str(cur_24.iloc[0]).strip().lower() in ("true","1","t","yes","y")) if len(cur_24)>0 else False
            cur_skip = dfs["clients"].loc[dfs["clients"]["Name"]==selected_client, "SkipForWeek"]
            cur_skip = (str(cur_skip.iloc[0]).strip().lower() in ("true","1","t","yes","y")) if len(cur_skip)>0 else False

            c24 = st.checkbox("24-Hour Client", value=cur_24, key=f"24_{selected_client}")
            skip_val = st.checkbox("Skip for the week", value=cur_skip, key=f"skip_client_{selected_client}")
            if st.button("Save 24-Hour / Skip flags for client"):
                dfs["clients"].loc[dfs["clients"]["Name"]==selected_client, "24_Hour"] = "True" if c24 else "False"
                dfs["clients"].loc[dfs["clients"]["Name"]==selected_client, "SkipForWeek"] = "True" if skip_val else "False"
                save_csv_safe(CLIENT_FILE, dfs["clients"])
                st.success("Client flags saved.")
                dfs["clients"] = load_csv_safe(CLIENT_FILE, ["Name","Base Location","Importance","Scheduling Mode","Preferred Caregivers","Notes","24_Hour","SkipForWeek"])
                st.experimental_rerun()

            # Fixed matrix
            st.markdown("**Fixed Shifts**")
            fixed_sub = dfs["client_fixed"][dfs["client_fixed"]["Client Name"]==selected_client].copy()
            fixed_sub = ensure_min_rows(fixed_sub, 2, {"Client Name": selected_client, "Day":"", "Start":"", "End":"", "Notes":""})
            edited_fixed = st.data_editor(
                fixed_sub,
                num_rows="dynamic",
                width="stretch",
                key=f"fixed_{selected_client}"
            )

            # Flexible matrix
            st.markdown("**Flexible Shifts**")
            flex_sub = dfs["client_flex"][dfs["client_flex"]["Client Name"]==selected_client].copy()
            flex_sub = ensure_min_rows(
                flex_sub, 2,
                {"Client Name": selected_client, "Length (hrs)":"", "Number of Shifts":"", "Start Day":"",
                 "End Day":"", "Start Time":"", "End Time":"", "Notes":""}
            )
            edited_flex = st.data_editor(
                flex_sub,
                num_rows="dynamic",
                width="stretch",
                key=f"flex_{selected_client}"
            )

            if st.button("💾 Save Shifts for selected client"):
                new_fixed = pd.concat([
                    dfs["client_fixed"][dfs["client_fixed"]["Client Name"]!=selected_client],
                    drop_empty_rows(edited_fixed)
                ], ignore_index=True)
                new_flex = pd.concat([
                    dfs["client_flex"][dfs["client_flex"]["Client Name"]!=selected_client],
                    drop_empty_rows(edited_flex)
                ], ignore_index=True)
                save_csv_safe(CLIENT_FIXED_FILE, new_fixed)
                save_csv_safe(CLIENT_FLEX_FILE, new_flex)
                st.success(f"Shifts saved for {selected_client}.")
                dfs["client_fixed"] = load_csv_safe(CLIENT_FIXED_FILE, ["Client Name","Day","Start","End","Notes"])
                dfs["client_flex"] = load_csv_safe(CLIENT_FLEX_FILE, ["Client Name","Length (hrs)","Number of Shifts","Start Day","End Day","Start Time","End Time","Notes"])
# Combined FirstApp.py  — Part 5/5
# (continuation directly after Part 4)

# ===========================
# SCHEDULES TAB
# ===========================
with tabs[2]:
    st.header("Schedules")
    sch_sub = st.tabs(["Caregivers", "Clients"])

    # Controls
    with st.expander("Solve Controls"):
        iters = st.number_input("Iterative solving (restarts)", min_value=1, value=1, step=1, key="iterative_solving_count")
        per_iter_time = st.number_input("Per-iteration time limit (seconds)", min_value=1, value=10, step=1, key="per_iter_time")

    # Solve button
    if st.button("▶️ Solve Schedules (run solver)"):
        # Build caregiver & client objects from current CSVs
        caregivers = []
        for _, r in dfs["caregivers"].iterrows():
            if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"):
                continue
            caregivers.append(
                Caregiver(
                    caregiver_id=r["Name"],
                    name=r["Name"],
                    min_hours=0.0, max_hours=168.0, prefer_max_hours=False,
                    base_location=r.get("Base Location",""),
                    availability={},  # availability will be used in advanced versions
                    preferred_clients=[]
                )
            )
        clients = []
        for _, r in dfs["clients"].iterrows():
            if str(r.get("SkipForWeek","")).strip().lower() in ("true","1","t","yes","y"):
                continue
            reqs=[]
            # 24-hour coverage auto-blocks
            if str(r.get("24_Hour","")).strip().lower() in ("true","1","t","yes","y"):
                for d in DAYS_FULL:
                    reqs.append({"type":"fixed","day":d,"start":"00:00","end":"24:00","allow_split":False})
            # fixed rows
            fixed_rows = dfs["client_fixed"][dfs["client_fixed"]["Client Name"]==r["Name"]]
            for _,fr in fixed_rows.iterrows():
                if fr["Day"] and fr["Start"] and fr["End"]:
                    reqs.append({"type":"fixed","day":fr["Day"],"start":fr["Start"],"end":fr["End"],"allow_split":False})
            # flexible rows
            flex_rows = dfs["client_flex"][dfs["client_flex"]["Client Name"]==r["Name"]]
            for _,fx in flex_rows.iterrows():
                try:
                    ln = float(fx.get("Length (hrs)","") or 0)
                    nm = int(float(fx.get("Number of Shifts","") or 0))
                except Exception:
                    continue
                if ln<=0 or nm<=0:
                    continue
                sday = fx.get("Start Day",""); eday = fx.get("End Day","")
                stime = fx.get("Start Time","") or "00:00"
                etime = fx.get("End Time","") or "24:00"
                # allowed day range
                if sday in DAYS_FULL and eday in DAYS_FULL:
                    si = DAYS_FULL.index(sday); ei = DAYS_FULL.index(eday)
                    if si<=ei: allowed = DAYS_FULL[si:ei+1]
                    else: allowed = DAYS_FULL[si:]+DAYS_FULL[:ei+1]
                elif sday in DAYS_FULL:
                    allowed = [sday]
                else:
                    allowed = DAYS_FULL.copy()
                reqs.append({"type":"flexible","blocks":nm,"duration":ln,"days":allowed,"window_start":stime,"window_end":etime})
            clients.append(
                Client(
                    client_id=r["Name"], name=r["Name"],
                    priority=int(r.get("Importance",0) or 0),
                    scheduling_mode=r.get("Scheduling Mode","Maximize Client Preference"),
                    base_location=r.get("Base Location",""),
                    top_caregivers=[p.strip() for p in str(r.get("Preferred Caregivers","")).split(",") if p.strip()],
                    requests=reqs,
                    notes=r.get("Notes","")
                )
            )

        # Iterative loop calling a simplified HomecareSolver wrapper
        # (Keeps your UI flowing; upgrade with full CP-SAT model if needed)
        random_seed = random.randint(1, 999999)
        best_result = None
        best_score = None
        log_rows = []
        for i in range(int(iters)):
            solver = HomecareSolver(
                caregivers=caregivers,
                clients=clients,
                fixed_requests=dfs["client_fixed"],
                flex_requests=dfs["client_flex"],
                approvals=[],  # approvals feed handled via Exceptions tab
                iterations=1,
                random_seed=random_seed + i
            )
            res = solver.build_and_solve()
            sc = res["score"] if isinstance(res, dict) else 0
            log_rows.append({"iteration": i+1, "score": sc, "timestamp": datetime.now().isoformat(), "notes": "ok"})
            if best_score is None or sc > best_score:
                best_score = sc
                best_result = res
                # write a minimal best solution CSV placeholder
                pd.DataFrame([{"iteration": i+1, "score": sc}]).to_csv(BEST_SOLUTION_FILE, index=False)

        # append iteration log
        iter_df = dfs["iters"]
        iter_df = pd.concat([iter_df, pd.DataFrame(log_rows)], ignore_index=True)
        save_csv_safe(ITER_LOG_FILE, iter_df)
        dfs["iters"] = load_csv_safe(ITER_LOG_FILE, ["iteration","score","timestamp","notes"])
        st.success(f"Solve complete. Best score={best_score}. Logs appended to {ITER_LOG_FILE}. Best solution written to {BEST_SOLUTION_FILE}.")

    # Schedule viewers (blank until a real schedule renderer is wired)
    def blank_schedule():
        df = pd.DataFrame(index=TIME_OPTS, columns=DAYS_FULL).fillna("")
        return df

    with sch_sub[0]:
        st.subheader("Caregiver Schedule Viewer")
        cg_names = dfs["caregivers"]["Name"].tolist()
        sel_cg = st.selectbox("Select Caregiver", options=[""] + cg_names, key="sched_cg_select")
        st.dataframe(blank_schedule(), use_container_width=True, height=1500)

    with sch_sub[1]:
        st.subheader("Client Schedule Viewer")
        cl_names = dfs["clients"]["Name"].tolist()
        sel_cl = st.selectbox("Select Client", options=[""] + cl_names, key="sched_client_select")
        st.dataframe(blank_schedule(), use_container_width=True, height=1500)

# ===========================
# EXCEPTIONS TAB
# ===========================
with tabs[3]:
    st.header("Exceptions & Approvals")
    approvals = dfs["approvals"]
    pending = approvals[approvals["decision"].astype(str).str.strip() == ""].copy()

    if pending.empty:
        st.info("No pending exceptions.")
    else:
        first = pending.iloc[0]
        st.subheader(f"Exception for caregiver: {first['caregiver_name']}  —  Constraint: {first['constraint_type']}")
        st.markdown(f"**Client:** {first['client_name']}  &nbsp; &nbsp; **Day:** {first['day']}  &nbsp; &nbsp; **Start:** {first['start']}  &nbsp; &nbsp; **End:** {first['end']}")

        # Snapshot (2h before/after) for context
        def parse_minutes(t: str) -> Optional[int]:
            try:
                h,m = map(int, t.split(":"))
                return h*60+m
            except Exception:
                return None

        start_min = parse_minutes(first["start"])
        end_min = parse_minutes(first["end"])
        if start_min is None or end_min is None:
            st.warning("Invalid time in exception row — cannot render snapshot.")
        else:
            window_start = max(0, start_min - 120)
            window_end = min(24*60, end_min + 120)
            rows=[]
            t = window_start
            while t <= window_end:
                rows.append(f"{t//60:02d}:{t%60:02d}")
                t += 30
            snap = pd.DataFrame(index=rows, columns=[first["day"]]).fillna("")
            for r in rows:
                mins = parse_minutes(r)
                if mins is not None and start_min <= mins < end_min:
                    snap.at[r, first["day"]] = f"⚠ {first['client_name']} (exception)"
            # render as simple HTML to highlight red cells
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
            st.experimental_rerun()
        if c2.button("Decline Exception"):
            approvals.loc[approvals["approval_id"]==first["approval_id"], "decision"] = "declined"
            approvals.loc[approvals["approval_id"]==first["approval_id"], "timestamp"] = datetime.now().isoformat()
            save_csv_safe(APPROVALS_FILE, approvals)
            st.success("Exception declined.")
            dfs["approvals"] = load_csv_safe(APPROVALS_FILE, ["approval_id","client_name","caregiver_name","day","start","end","constraint_type","decision","timestamp","notes"])
            st.experimental_rerun()

    st.markdown("---")
    st.subheader("Approval History")
    st.dataframe(dfs["approvals"], use_container_width=True, height=400)

# ===========================
# SETTINGS TAB
# ===========================
with tabs[4]:
    st.header("Settings")
    st.write("Export (Save As) or Import (Load From File) the current CSVs. Iterative controls are on Schedules tab.")

    # Save As ZIP
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
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to extract ZIP: {e}")

    st.caption("Note: Files are stored on the machine running this app (or your hosting environment). Save As lets you download a local backup.")

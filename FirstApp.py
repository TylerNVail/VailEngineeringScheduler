# FirstApp.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta

# ------------------------
# Page config + CSS widen
# ------------------------
st.set_page_config(page_title="Homecare Scheduler", layout="wide")
st.markdown(
    """
    <style>
        .block-container { max-width: 100% !important; padding-left: 1.5rem; padding-right: 1.5rem; }
        .exception-cell { color: red; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------
# File paths (defaults)
# ------------------------
CAREGIVER_FILE = "caregivers.csv"
CAREGIVER_AVAIL_FILE = "caregiver_availability.csv"
CLIENT_FILE = "clients.csv"
CLIENT_FIXED_FILE = "client_fixed_shifts.csv"
CLIENT_FLEX_FILE = "client_flexible_shifts.csv"
APPROVALS_FILE = "approvals.csv"

# ------------------------
# Ensure CSVs exist & load with safe dtypes
# ------------------------
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

# default column lists
caregiver_cols = ["Name", "Base Location", "Notes"]
caregiver_avail_cols = ["Caregiver Name", "Day", "Start", "End", "Availability Type", "Notes"]
client_cols = ["Name", "Base Location", "Importance", "Scheduling Mode", "Preferred Caregivers", "Notes"]
client_fixed_cols = ["Client Name", "Day", "Start", "End", "Notes"]
client_flex_cols = ["Client Name", "Length (hrs)", "Number of Shifts", "Start Day", "End Day", "Start Time", "End Time", "Notes"]
approvals_cols = ["approval_id", "client_name", "caregiver_name", "day", "start", "end", "constraint_type", "decision", "timestamp", "notes"]

# ensure files exist
for p, cols in [
    (CAREGIVER_FILE, caregiver_cols),
    (CAREGIVER_AVAIL_FILE, caregiver_avail_cols),
    (CLIENT_FILE, client_cols),
    (CLIENT_FIXED_FILE, client_fixed_cols),
    (CLIENT_FLEX_FILE, client_flex_cols),
    (APPROVALS_FILE, approvals_cols)
]:
    ensure_csv(p, cols)

# load
caregivers_df = load_csv_safe(CAREGIVER_FILE, caregiver_cols)
caregiver_avail_df = load_csv_safe(CAREGIVER_AVAIL_FILE, caregiver_avail_cols)
clients_df = load_csv_safe(CLIENT_FILE, client_cols)
client_fixed_df = load_csv_safe(CLIENT_FIXED_FILE, client_fixed_cols)
client_flex_df = load_csv_safe(CLIENT_FLEX_FILE, client_flex_cols)
approvals_df = load_csv_safe(APPROVALS_FILE, approvals_cols)

# ------------------------
# Helpers
# ------------------------
DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

def time_30m_options():
    t = datetime(2000,1,1,0,0)
    opts=[]
    for i in range(48):
        opts.append(t.strftime("%H:%M"))
        t = t + timedelta(minutes=30)
    return opts
TIME_OPTS = time_30m_options()

def ensure_min_rows(df, min_rows, defaults):
    if df is None or df.empty:
        df = pd.DataFrame(columns=list(defaults.keys()))
    if len(df) < min_rows:
        add = min_rows - len(df)
        newrows = pd.DataFrame([defaults.copy() for _ in range(add)])
        df = pd.concat([df, newrows], ignore_index=True)
    return df.reset_index(drop=True)

def drop_empty_rows(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns) if df is not None else pd.DataFrame()
    mask = ~(df.replace("", pd.NA).isna().all(axis=1))
    return df[mask].reset_index(drop=True)

def make_blank_schedule_df():
    times = TIME_OPTS
    df = pd.DataFrame(index=times, columns=DAYS)
    return df.fillna("")

def parse_time_to_minutes(t):
    # t like "13:30"
    try:
        h,m = map(int, t.split(":"))
        return h*60 + m
    except Exception:
        return None

# ------------------------
# Main tabs (kept layout you liked)
# ------------------------
tabs = st.tabs(["Caregivers", "Clients", "Schedules", "Exceptions", "Settings"])

# ------------------------
# CAREGIVERS tab
# ------------------------
with tabs[0]:
    st.header("Caregivers")
    cg_sub = st.tabs(["Caregiver List (core profile)", "Availability"])

    # Caregiver List
    with cg_sub[0]:
        st.subheader("Caregiver List (core profile)")
        # ensure Importance etc not here; it's caregiver core
        edited_caregivers = st.data_editor(
            caregivers_df,
            num_rows="dynamic",
            column_config={
                "Name": st.column_config.TextColumn("Name"),
                "Base Location": st.column_config.TextColumn("Base Location"),
                "Notes": st.column_config.TextColumn("Notes")
            },
            key="cg_list_editor",
            width="stretch",
        )
        if st.button("💾 Save Caregiver List"):
            cleaned = drop_empty_rows(edited_caregivers)
            save_csv_safe(CAREGIVER_FILE, cleaned)
            st.success("Caregiver list saved.")
            caregivers_df = load_csv_safe(CAREGIVER_FILE, caregiver_cols)

    # Availability (one caregiver at a time)
    with cg_sub[1]:
        st.subheader("Caregiver Availability")
        cg_names = caregivers_df["Name"].tolist() if not caregivers_df.empty else []
        selected = st.selectbox("Select Caregiver", options=[""] + cg_names, index=0, key="avail_select")

        if selected:
            sub_df = caregiver_avail_df[caregiver_avail_df["Caregiver Name"] == selected].copy()
            sub_df = ensure_min_rows(sub_df, 3, {"Caregiver Name": selected, "Day": "", "Start": "", "End": "", "Availability Type": "", "Notes": ""})

            edited_avail = st.data_editor(
                sub_df,
                num_rows="dynamic",
                column_config={
                    "Caregiver Name": st.column_config.TextColumn("Caregiver Name"),
                    "Day": st.column_config.TextColumn("Day"),
                    "Start": st.column_config.TextColumn("Start"),
                    "End": st.column_config.TextColumn("End"),
                    "Availability Type": st.column_config.TextColumn("Availability Type"),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                key=f"avail_editor_{selected}",
                width="stretch",
            )

            if st.button("💾 Save Availability for selected caregiver"):
                rest = caregiver_avail_df[caregiver_avail_df["Caregiver Name"] != selected].copy()
                edited_clean = drop_empty_rows(edited_avail)
                caregiver_avail_df = pd.concat([rest, edited_clean], ignore_index=True)
                save_csv_safe(CAREGIVER_AVAIL_FILE, caregiver_avail_df)
                st.success(f"Availability saved for {selected}.")
                caregiver_avail_df = load_csv_safe(CAREGIVER_AVAIL_FILE, caregiver_avail_cols)

# ------------------------
# CLIENTS tab
# ------------------------
with tabs[1]:
    st.header("Clients")
    cl_sub = st.tabs(["Client List (core profile)", "Shifts"])

    # Client List
    with cl_sub[0]:
        st.subheader("Client List (core profile)")
        # Ensure Importance numeric display
        clients_df["Importance"] = pd.to_numeric(clients_df["Importance"].replace("", "0"), errors="coerce").fillna(0).astype(int)
        edited_clients = st.data_editor(
            clients_df,
            num_rows="dynamic",
            column_config={
                "Name": st.column_config.TextColumn("Name"),
                "Base Location": st.column_config.TextColumn("Base Location"),
                "Importance": st.column_config.NumberColumn("Importance", min_value=0, max_value=10, step=1),
                "Scheduling Mode": st.column_config.TextColumn("Scheduling Mode"),
                "Preferred Caregivers": st.column_config.TextColumn("Preferred Caregivers (comma separated)"),
                "Notes": st.column_config.TextColumn("Notes"),
            },
            key="client_list_editor",
            width="stretch",
        )
        if st.button("💾 Save Client List"):
            cleaned = drop_empty_rows(edited_clients)
            cleaned["Importance"] = pd.to_numeric(cleaned["Importance"].replace("", "0"), errors="coerce").fillna(0).astype(int)
            save_csv_safe(CLIENT_FILE, cleaned)
            st.success("Client list saved.")
            clients_df = load_csv_safe(CLIENT_FILE, client_cols)

    # Client Shifts (one client at a time)
    with cl_sub[1]:
        st.subheader("Client Shifts")
        client_names = clients_df["Name"].tolist() if not clients_df.empty else []
        selected_client = st.selectbox("Select Client", options=[""] + client_names, index=0, key="shift_select")

        if selected_client:
            st.markdown("**Fixed Shifts**")
            sub_fixed = client_fixed_df[client_fixed_df["Client Name"] == selected_client].copy()
            sub_fixed = ensure_min_rows(sub_fixed, 2, {"Client Name": selected_client, "Day": "", "Start": "", "End": "", "Notes": ""})
            edited_fixed = st.data_editor(sub_fixed, key=f"fixed_editor_{selected_client}", num_rows="dynamic", width="stretch")

            st.markdown("**Flexible Shifts**")
            sub_flex = client_flex_df[client_flex_df["Client Name"] == selected_client].copy()
            sub_flex = ensure_min_rows(sub_flex, 2, {"Client Name": selected_client, "Length (hrs)": "", "Number of Shifts": "", "Start Day": "", "End Day": "", "Start Time": "", "End Time": "", "Notes": ""})
            edited_flex = st.data_editor(sub_flex, key=f"flex_editor_{selected_client}", num_rows="dynamic", width="stretch")

            if st.button("💾 Save Shifts for selected client"):
                client_fixed_df = pd.concat([client_fixed_df[client_fixed_df["Client Name"] != selected_client], drop_empty_rows(edited_fixed)], ignore_index=True)
                client_flex_df = pd.concat([client_flex_df[client_flex_df["Client Name"] != selected_client], drop_empty_rows(edited_flex)], ignore_index=True)
                save_csv_safe(CLIENT_FIXED_FILE, client_fixed_df)
                save_csv_safe(CLIENT_FLEX_FILE, client_flex_df)
                st.success(f"Shifts saved for {selected_client}.")
                client_fixed_df = load_csv_safe(CLIENT_FIXED_FILE, client_fixed_cols)
                client_flex_df = load_csv_safe(CLIENT_FLEX_FILE, client_flex_cols)

# ------------------------
# SCHEDULES tab
# ------------------------
with tabs[2]:
    st.header("Schedules")
    sch_sub = st.tabs(["Caregivers", "Clients"])

    # Solve button (placeholder hook)
    if st.button("▶️ Solve Schedules (run solver)"):
        # placeholder - will be replaced by solver integration
        st.info("Solver running... (placeholder). Once solver is integrated, this will generate assignments and exceptions.")
        # call to run_solver() would go here

    # helper blank matrix generator
    def blank_schedule():
        times = TIME_OPTS
        df = pd.DataFrame(index=times, columns=DAYS)
        return df.fillna("")

    with sch_sub[0]:
        st.subheader("Caregiver Schedule Viewer")
        cg_names = caregivers_df["Name"].tolist()
        sel = st.selectbox("Select Caregiver", options=[""] + cg_names, key="sched_cg_select")
        mat = blank_schedule()
        # make the dataframe taller so it displays more rows at once
        st.dataframe(mat, use_container_width=True, height=1500)

    with sch_sub[1]:
        st.subheader("Client Schedule Viewer")
        cl_names = clients_df["Name"].tolist()
        selc = st.selectbox("Select Client", options=[""] + cl_names, key="sched_client_select")
        mat2 = blank_schedule()
        st.dataframe(mat2, use_container_width=True, height=1500)

# ------------------------
# EXCEPTIONS tab
# ------------------------
with tabs[3]:
    st.header("Exceptions & Approvals")
    st.write("Pending exceptions will appear here for review. Select Approve or Decline to record a decision. (This currently updates the approvals CSV; later it will re-run the solver.)")

    # load approvals and find pending (decision empty)
    approvals_df = load_csv_safe(APPROVALS_FILE, approvals_cols)
    pending = approvals_df[approvals_df["decision"].astype(str).str.strip() == ""].copy()

    if pending.empty:
        st.info("No pending exceptions.")
    else:
        # show the first pending exception
        first = pending.iloc[0]
        st.subheader(f"Exception for caregiver: {first['caregiver_name']}  —  Constraint: {first['constraint_type']}")
        st.markdown(f"**Client:** {first['client_name']}  &nbsp; &nbsp; **Day:** {first['day']}  &nbsp; &nbsp; **Start:** {first['start']}  &nbsp; &nbsp; **End:** {first['end']}")

        # build snapshot window: 2 hours before start to 2 hours after end
        start_min = parse_time_to_minutes(first["start"])
        end_min = parse_time_to_minutes(first["end"])
        if start_min is None or end_min is None:
            st.warning("Invalid time in exception row — cannot render snapshot.")
        else:
            window_start = max(0, start_min - 120)
            window_end = min(24*60, end_min + 120)
            # build 30-min rows between window_start and window_end inclusive
            rows = []
            t = window_start
            while t < window_end:
                hh = t // 60
                mm = t % 60
                rows.append(f"{hh:02d}:{mm:02d}")
                t += 30
            # Single column for the exception day to keep snapshot focused
            snapshot = pd.DataFrame(index=rows, columns=[first["day"]])
            # Try to fill snapshot with any existing assignments (we don't have assignments yet), so leave blank
            # Mark the exception period in red text
            # Determine rows where exception falls
            ex_start = start_min
            ex_end = end_min
            for r in rows:
                mins = parse_time_to_minutes(r)
                if mins is not None and ex_start <= mins < ex_end:
                    snapshot.at[r, first["day"]] = f"⚠ {first['client_name']} (exception)"
                else:
                    snapshot.at[r, first["day"]] = ""

            # render snapshot, highlight exception rows by simple HTML in a table
            def render_snapshot_as_html(df_snap, exception_marker="⚠"):
                html = '<table style="border-collapse:collapse;">'
                # header
                html += "<tr>"
                for c in df_snap.columns:
                    html += f'<th style="border:1px solid #ddd;padding:6px;background:#f4f4f4;">{c}</th>'
                html += "</tr>"
                # rows
                for idx in df_snap.index:
                    html += "<tr>"
                    for c in df_snap.columns:
                        val = df_snap.at[idx, c]
                        if isinstance(val, str) and val.startswith(exception_marker):
                            html += f'<td style="border:1px solid #ddd;padding:6px;color:red;font-weight:bold;">{val}</td>'
                        else:
                            html += f'<td style="border:1px solid #ddd;padding:6px;">{val}</td>'
                    html += "</tr>"
                html += "</table>"
                return html

            st.markdown(render_snapshot_as_html(snapshot), unsafe_allow_html=True)

        # Approve / Decline buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve Exception"):
                approvals_df.loc[approvals_df["approval_id"] == first["approval_id"], "decision"] = "approved"
                approvals_df.loc[approvals_df["approval_id"] == first["approval_id"], "timestamp"] = pd.Timestamp.now().isoformat()
                save_csv_safe(APPROVALS_FILE, approvals_df)
                st.success("Exception approved.")
                st.experimental_rerun()
        with col2:
            if st.button("Decline Exception"):
                approvals_df.loc[approvals_df["approval_id"] == first["approval_id"], "decision"] = "declined"
                approvals_df.loc[approvals_df["approval_id"] == first["approval_id"], "timestamp"] = pd.Timestamp.now().isoformat()
                save_csv_safe(APPROVALS_FILE, approvals_df)
                st.success("Exception declined.")
                st.experimental_rerun()

    st.markdown("---")
    st.subheader("Approval History")
    approvals_history = load_csv_safe(APPROVALS_FILE, approvals_cols)
    st.dataframe(approvals_history, use_container_width=True, height=400)

# ------------------------
# SETTINGS tab
# ------------------------
with tabs[4]:
    st.header("Settings")
    st.write("CSV file paths are currently defaults. Changing these fields only updates display; wiring custom paths can be added later if needed.")
    st.text_input("Caregiver CSV path", value=CAREGIVER_FILE, key="cfg_cg_path")
    st.text_input("Caregiver Availability CSV path", value=CAREGIVER_AVAIL_FILE, key="cfg_cg_avail_path")
    st.text_input("Client CSV path", value=CLIENT_FILE, key="cfg_client_path")
    st.text_input("Client Fixed Shifts CSV path", value=CLIENT_FIXED_FILE, key="cfg_client_fixed")
    st.text_input("Client Flexible Shifts CSV path", value=CLIENT_FLEX_FILE, key="cfg_client_flex")
    st.text_input("Approvals CSV path", value=APPROVALS_FILE, key="cfg_approvals")
    st.info("Path editing here is informational for now; we can wire path persistence if you want.")

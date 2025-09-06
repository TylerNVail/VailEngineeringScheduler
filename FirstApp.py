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
    # ensure columns order and presence
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()

def save_csv_safe(path, df):
    # drop fully empty rows first
    df2 = df.dropna(how="all").reset_index(drop=True)
    # coerce all to strings for stable storage
    df2 = df2.astype(str)
    df2.to_csv(path, index=False)

# default columns
caregiver_cols = ["Name", "Base Location", "Notes"]
caregiver_avail_cols = ["Caregiver Name", "Day", "Start", "End", "Availability Type", "Notes"]
client_cols = ["Name", "Base Location", "Importance", "Scheduling Mode", "Preferred Caregivers", "Notes"]
client_fixed_cols = ["Client Name", "Day", "Start", "End", "Notes"]
client_flex_cols = ["Client Name", "Length (hrs)", "Number of Shifts", "Start Day", "End Day", "Start Time", "End Time", "Notes"]
approvals_cols = ["approval_id", "client_name", "caregiver_name", "day", "start", "end", "constraint_type", "decision", "timestamp", "notes"]

# make sure files exist
for p, cols in [
    (CAREGIVER_FILE, caregiver_cols),
    (CAREGIVER_AVAIL_FILE, caregiver_avail_cols),
    (CLIENT_FILE, client_cols),
    (CLIENT_FIXED_FILE, client_fixed_cols),
    (CLIENT_FLEX_FILE, client_flex_cols),
    (APPROVALS_FILE, approvals_cols)
]:
    ensure_csv(p, cols)

# load dataframes
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
    if df is None:
        df = pd.DataFrame(columns=list(defaults.keys()))
    if len(df) < min_rows:
        add = min_rows - len(df)
        newrows = pd.DataFrame([defaults.copy() for _ in range(add)])
        df = pd.concat([df, newrows], ignore_index=True)
    return df.reset_index(drop=True)

def drop_empty_rows(df):
    # consider a row empty if all cells are empty strings
    if df is None or df.empty:
        return df.copy()
    mask = ~(df.replace("", pd.NA).isna().all(axis=1))
    return df[mask].reset_index(drop=True)

# ------------------------
# Top-level tabs (keeps layout you liked)
# ------------------------
tabs = st.tabs(["Caregivers", "Clients", "Schedules", "Exceptions", "Settings"])

# ------------------------
# CAREGIVERS tab
# ------------------------
with tabs[0]:
    st.header("Caregivers")

    cg_sub = st.tabs(["Caregiver List", "Availability"])

    # Caregiver list editor
    with cg_sub[0]:
        st.subheader("Caregiver List (core profile)")
        # Keep Base Location and Notes as strings but provide dropdown helper below for convenience
        edited_cgs = st.data_editor(
            caregivers_df,
            num_rows="dynamic",
            column_config={
                "Name": st.column_config.TextColumn("Name"),
                "Base Location": st.column_config.TextColumn("Base Location"),  # text for stability
                "Notes": st.column_config.TextColumn("Notes")
            },
            key="cg_list_editor",
            width="stretch"
        )

        # convenience panel to set Base Location quickly for new/selected rows
        with st.expander("Quick controls (use to set dropdown-like values before saving)"):
            loc = st.selectbox("Set Base Location for new/selected rows", ["", "Paradise", "Chico", "Oroville"], key="cg_quick_loc")
            if st.button("Apply Base Location to all blank Base Location cells"):
                edited_cgs.loc[edited_cgs["Base Location"].astype(str).str.strip()=="", "Base Location"] = loc
                st.experimental_rerun()

        if st.button("Save Caregiver List"):
            # coerce types and save (remove empty rows)
            cleaned = drop_empty_rows(edited_cgs)
            save_csv_safe(CAREGIVER_FILE, cleaned)
            st.success("Caregiver list saved.")
            # reload into memory
            caregivers_df = load_csv_safe(CAREGIVER_FILE, caregiver_cols)

    # Availability (one caregiver at a time via filter dropdown)
    with cg_sub[1]:
        st.subheader("Caregiver Availability")
        cg_names = caregivers_df["Name"].tolist() if not caregivers_df.empty else []
        selected_cg = st.selectbox("Select Caregiver", options=[""] + cg_names, index=0, key="avail_select")

        # helper quick inputs acting as dropdowns for adding rows
        with st.expander("Quick add / dropdown controls"):
            day_sel = st.selectbox("Day", options=[""] + DAYS, key="avail_quick_day")
            start_sel = st.selectbox("Start time", options=[""] + TIME_OPTS, key="avail_quick_start")
            end_sel = st.selectbox("End time", options=[""] + TIME_OPTS, key="avail_quick_end")
            avtype_sel = st.selectbox("Availability Type", options=["", "Available", "Preferred Unavailable"], key="avail_quick_type")
            if st.button("Add availability row with selected values"):
                if selected_cg:
                    new = {
                        "Caregiver Name": selected_cg,
                        "Day": day_sel,
                        "Start": start_sel,
                        "End": end_sel,
                        "Availability Type": avtype_sel,
                        "Notes": ""
                    }
                    caregiver_avail_df = pd.concat([caregiver_avail_df, pd.DataFrame([new])], ignore_index=True)
                    save_csv_safe(CAREGIVER_AVAIL_FILE, caregiver_avail_df)
                    st.experimental_rerun()
                else:
                    st.warning("Select a caregiver first.")

        if selected_cg:
            sub_df = caregiver_avail_df[caregiver_avail_df["Caregiver Name"] == selected_cg].copy()
            sub_df = ensure_min_rows(sub_df, 3, {"Caregiver Name": selected_cg, "Day": "", "Start": "", "End": "", "Availability Type": "", "Notes": ""})

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
                key=f"avail_editor_{selected_cg}",
                width="stretch"
            )

            if st.button("Save Availability for selected caregiver"):
                # combine edited into master df, remove empties
                rest = caregiver_avail_df[caregiver_avail_df["Caregiver Name"] != selected_cg].copy()
                edited_clean = drop_empty_rows(edited_avail)
                caregiver_avail_df = pd.concat([rest, edited_clean], ignore_index=True)
                save_csv_safe(CAREGIVER_AVAIL_FILE, caregiver_avail_df)
                st.success(f"Availability saved for {selected_cg}.")
                # reload
                caregiver_avail_df = load_csv_safe(CAREGIVER_AVAIL_FILE, caregiver_avail_cols)

# ------------------------
# CLIENTS tab
# ------------------------
with tabs[1]:
    st.header("Clients")
    cl_sub = st.tabs(["Client List", "Shifts"])

    # Client List editor
    with cl_sub[0]:
        st.subheader("Client List (core profile)")
        # Importance is numeric, rest are text; Preferred Caregivers kept as comma-separated text for now
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
            width="stretch"
        )
        # helper to apply scheduling mode or base location via selectboxes (dropdown-like)
        with st.expander("Quick controls (apply to blank cells)"):
            cl_loc = st.selectbox("Base Location", options=["", "Paradise", "Chico", "Oroville"], key="client_quick_loc")
            cl_mode = st.selectbox("Scheduling Mode", options=["", "Maximize Client Preference", "Maximize Fairness"], key="client_quick_mode")
            if st.button("Apply quick values to blank cells"):
                if cl_loc:
                    edited_clients.loc[edited_clients["Base Location"].astype(str).str.strip() == "", "Base Location"] = cl_loc
                if cl_mode:
                    edited_clients.loc[edited_clients["Scheduling Mode"].astype(str).str.strip() == "", "Scheduling Mode"] = cl_mode
                st.experimental_rerun()

        if st.button("Save Client List"):
            cleaned = drop_empty_rows(edited_clients)
            # ensure Importance numeric
            cleaned["Importance"] = pd.to_numeric(cleaned["Importance"].replace("", "0"), errors="coerce").fillna(0).astype(int)
            save_csv_safe(CLIENT_FILE, cleaned)
            st.success("Client list saved.")
            clients_df = load_csv_safe(CLIENT_FILE, client_cols)

    # Client shifts (one client at a time filter)
    with cl_sub[1]:
        st.subheader("Client Shifts")
        client_names = clients_df["Name"].tolist() if not clients_df.empty else []
        selected_client = st.selectbox("Select Client", options=[""] + client_names, index=0, key="shift_select")

        # quick add controls for fixed & flexible
        with st.expander("Quick add controls"):
            # fixed
            f_day = st.selectbox("Fixed - Day", options=[""] + DAYS, key="fixed_quick_day")
            f_start = st.selectbox("Fixed - Start", options=[""] + TIME_OPTS, key="fixed_quick_start")
            f_end = st.selectbox("Fixed - End", options=[""] + TIME_OPTS, key="fixed_quick_end")
            if st.button("Add Fixed Shift (quick)"):
                if selected_client:
                    new = {"Client Name": selected_client, "Day": f_day, "Start": f_start, "End": f_end, "Notes": ""}
                    client_fixed_df = pd.concat([client_fixed_df, pd.DataFrame([new])], ignore_index=True)
                    save_csv_safe(CLIENT_FIXED_FILE, client_fixed_df)
                    st.experimental_rerun()
                else:
                    st.warning("Select a client first.")
            # flexible
            fl_len = st.number_input("Flexible - Length (hrs)", min_value=1, value=4, key="flex_quick_len")
            fl_num = st.number_input("Flexible - Number of Shifts", min_value=1, value=1, key="flex_quick_num")
            fl_start_day = st.selectbox("Flexible - Start Day", options=[""] + DAYS, key="flex_quick_sday")
            fl_end_day = st.selectbox("Flexible - End Day", options=[""] + DAYS, key="flex_quick_eday")
            fl_s = st.selectbox("Flexible - Start Time", options=[""] + TIME_OPTS, key="flex_quick_stime")
            fl_e = st.selectbox("Flexible - End Time", options=[""] + TIME_OPTS, key="flex_quick_etime")
            if st.button("Add Flexible Shift (quick)"):
                if selected_client:
                    new = {"Client Name": selected_client, "Length (hrs)": str(fl_len), "Number of Shifts": str(fl_num),
                           "Start Day": fl_start_day, "End Day": fl_end_day, "Start Time": fl_s, "End Time": fl_e, "Notes": ""}
                    client_flex_df = pd.concat([client_flex_df, pd.DataFrame([new])], ignore_index=True)
                    save_csv_safe(CLIENT_FLEX_FILE, client_flex_df)
                    st.experimental_rerun()
                else:
                    st.warning("Select a client first.")

        if selected_client:
            # Fixed shifts section
            st.markdown("**Fixed Shifts**")
            sub_fixed = client_fixed_df[client_fixed_df["Client Name"] == selected_client].copy()
            sub_fixed = ensure_min_rows(sub_fixed, 2, {"Client Name": selected_client, "Day": "", "Start": "", "End": "", "Notes": ""})
            edited_fixed = st.data_editor(
                sub_fixed,
                key=f"fixed_editor_{selected_client}",
                num_rows="dynamic",
                width="stretch"
            )

            # Flexible shifts section
            st.markdown("**Flexible Shifts**")
            sub_flex = client_flex_df[client_flex_df["Client Name"] == selected_client].copy()
            sub_flex = ensure_min_rows(sub_flex, 2, {"Client Name": selected_client, "Length (hrs)": "", "Number of Shifts": "", "Start Day": "", "End Day": "", "Start Time": "", "End Time": "", "Notes": ""})
            edited_flex = st.data_editor(
                sub_flex,
                key=f"flex_editor_{selected_client}",
                num_rows="dynamic",
                width="stretch"
            )

            if st.button("Save Shifts for selected client"):
                # combine back into master tables
                client_fixed_df = pd.concat([client_fixed_df[client_fixed_df["Client Name"] != selected_client], drop_empty_rows(edited_fixed)], ignore_index=True)
                client_flex_df = pd.concat([client_flex_df[client_flex_df["Client Name"] != selected_client], drop_empty_rows(edited_flex)], ignore_index=True)
                save_csv_safe(CLIENT_FIXED_FILE, client_fixed_df)
                save_csv_safe(CLIENT_FLEX_FILE, client_flex_df)
                st.success(f"Shifts saved for {selected_client}.")

# ------------------------
# SCHEDULES tab (non-editable schedule matrices)
# ------------------------
with tabs[2]:
    st.header("Schedules")
    sch_sub = st.tabs(["Caregivers", "Clients"])

    # Helper: create blank schedule matrix (48 rows x 7 cols)
    times = TIME_OPTS
    def blank_schedule_df():
        df = pd.DataFrame(index=times, columns=DAYS)
        df = df.fillna("")  # blank cells
        return df

    with sch_sub[0]:
        st.subheader("Caregiver Schedule Viewer")
        cg_names = caregivers_df["Name"].tolist()
        sel = st.selectbox("Select Caregiver", options=[""] + cg_names, key="sched_cg_select")
        mat = blank_schedule_df()
        # Non-editable display
        st.dataframe(mat, use_container_width=True)

    with sch_sub[1]:
        st.subheader("Client Schedule Viewer")
        cl_names = clients_df["Name"].tolist()
        selc = st.selectbox("Select Client", options=[""] + cl_names, key="sched_client_select")
        mat2 = blank_schedule_df()
        st.dataframe(mat2, use_container_width=True)

# ------------------------
# EXCEPTIONS tab (UI stub to collect approvals)
# ------------------------
with tabs[3]:
    st.header("Exceptions & Approvals")
    st.write("This page will list unresolved hard-constraint exceptions and let you Approve/Decline them. Right now it stores manual approvals for testing.")
    # Load approvals table
    approvals_df = load_csv_safe(APPROVALS_FILE, approvals_cols)
    # Show existing approvals history
    st.subheader("Approval History")
    st.dataframe(approvals_df, use_container_width=True)

    st.subheader("Add Manual Approval (for testing)")
    an_client = st.text_input("Client name")
    an_caregiver = st.text_input("Caregiver name")
    an_day = st.text_input("Day")
    an_start = st.text_input("Start")
    an_end = st.text_input("End")
    an_type = st.text_input("Constraint type")
    if st.button("Add Approval Row"):
        new = {
            "approval_id": f"A_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
            "client_name": an_client,
            "caregiver_name": an_caregiver,
            "day": an_day,
            "start": an_start,
            "end": an_end,
            "constraint_type": an_type,
            "decision": "approved",
            "timestamp": pd.Timestamp.now().isoformat(),
            "notes": ""
        }
        approvals_df = pd.concat([approvals_df, pd.DataFrame([new])], ignore_index=True)
        save_csv_safe(APPROVALS_FILE, approvals_df)
        st.success("Approval added (manual).")

# ------------------------
# SETTINGS tab
# ------------------------
with tabs[4]:
    st.header("Settings")
    st.write("CSV files are stored next to this app by default. If you want different paths or to move files, edit these paths and restart the app.")
    st.text_input("Caregiver CSV path", value=CAREGIVER_FILE, key="cfg_cg_path")
    st.text_input("Caregiver Availability CSV path", value=CAREGIVER_AVAIL_FILE, key="cfg_cg_avail_path")
    st.text_input("Client CSV path", value=CLIENT_FILE, key="cfg_client_path")
    st.text_input("Client Fixed Shifts CSV path", value=CLIENT_FIXED_FILE, key="cfg_client_fixed")
    st.text_input("Client Flexible Shifts CSV path", value=CLIENT_FLEX_FILE, key="cfg_client_flex")
    st.text_input("Approvals CSV path", value=APPROVALS_FILE, key="cfg_approvals")
    st.info("Path editing is informational for now. We'll wire actual path-save behavior if you want it.")

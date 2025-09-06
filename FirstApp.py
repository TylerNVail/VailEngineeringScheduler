import streamlit as st
import pandas as pd
import os
import uuid
from datetime import datetime, timedelta

# ==========================
# Utilities
# ==========================

def mk_id(prefix: str = "ID"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def time_options_30min():
    opts = []
    t = datetime(2000, 1, 1, 0, 0)
    for i in range(48):
        opts.append(t.strftime("%H:%M"))
        t = t + timedelta(minutes=30)
    return opts

DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
TIME_OPTS = time_options_30min()

# ==========================
# Default CSV file paths
# ==========================

def get_default_paths():
    return {
        "caregivers_list": "caregivers_list.csv",
        "caregivers_availability": "caregivers_availability.csv",
        "clients_list": "clients_list.csv",
        "clients_fixed": "clients_fixed.csv",
        "clients_flexible": "clients_flexible.csv",
        "approvals": "approvals.csv"
    }

# ==========================
# CSV helpers
# ==========================

def ensure_folder(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def load_or_create_csv(path, columns):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # ensure columns
            for c in columns:
                if c not in df.columns:
                    df[c] = ""
            return df[columns]
        except Exception:
            # corrupt or unreadable; recreate
            df = pd.DataFrame(columns=columns)
            df.to_csv(path, index=False)
            return df
    else:
        ensure_folder(path)
        df = pd.DataFrame(columns=columns)
        df.to_csv(path, index=False)
        return df


def save_csv(df, path):
    ensure_folder(path)
    df.to_csv(path, index=False)

# ==========================
# Session state init
# ==========================

if "file_paths" not in st.session_state:
    st.session_state.file_paths = get_default_paths()

# load current caregivers list to populate dropdowns
caregiver_list_cols = ["caregiver_id", "name", "base_location", "min_hours", "max_hours", "prefer_max_hours", "notes"]
caregivers_df = load_or_create_csv(st.session_state.file_paths["caregivers_list"], caregiver_list_cols)

# ==========================
# Page config and layout
# ==========================
st.set_page_config(page_title="Homecare Scheduler", layout="wide")
st.title("Homecare Scheduler")

main_tabs = st.tabs(["Profiles", "Schedules", "Settings"])

# ==========================
# PROFILES: Caregivers & Clients
# ==========================
with main_tabs[0]:
    st.header("Profiles")
    profile_tabs = st.tabs(["Caregivers", "Clients"])

    # ------------------ CAREGIVERS ------------------
    with profile_tabs[0]:
        st.subheader("Caregivers")
        cg_tabs = st.tabs(["Caregiver List", "Availability"])

        # ---- Caregiver List ----
        with cg_tabs[0]:
            st.markdown("**Caregiver List (core profile data)**")
            caregivers_df = load_or_create_csv(st.session_state.file_paths["caregivers_list"], caregiver_list_cols)

            # ensure caregiver_id exists for existing rows
            if caregivers_df.empty:
                pass
            else:
                # fill missing ids
                for i, r in caregivers_df.iterrows():
                    if not r.get("caregiver_id"):
                        caregivers_df.at[i, "caregiver_id"] = mk_id("CG")

            # column config: Base Location as selectbox; prefer_max_hours as checkbox
            col_cfg = {
                "base_location": st.column_config.SelectboxColumn("Base Location", options=["Paradise", "Chico", "Oroville"]),
                "min_hours": st.column_config.NumberColumn("Min Hours"),
                "max_hours": st.column_config.NumberColumn("Max Hours"),
                "prefer_max_hours": st.column_config.CheckboxColumn("Prefer max hours")
            }

            edited_cg = st.data_editor(
                caregivers_df,
                column_config=col_cfg,
                hide_index=True,
                num_rows="dynamic",
                width="stretch",
                key="caregiver_list_editor"
            )

            if st.button("Save Caregiver List", key="save_cg_list"):
                # ensure caregiver_id present
                for i, r in edited_cg.iterrows():
                    if not r.get("caregiver_id"):
                        edited_cg.at[i, "caregiver_id"] = mk_id("CG")
                save_csv(edited_cg, st.session_state.file_paths["caregivers_list"])
                st.success("Caregiver list saved.")

        # ---- Availability ----
        with cg_tabs[1]:
            st.markdown("**Caregiver Availability**")
            caregivers_df = load_or_create_csv(st.session_state.file_paths["caregivers_list"], caregiver_list_cols)
            # build selection list
            names = ["-- New Caregiver --"] + caregivers_df["name"].fillna("Unnamed").tolist()
            sel = st.selectbox("Select caregiver to edit availability", names, key="sel_cg_avail")

            # availability CSV schema
            avail_cols = ["availability_id", "caregiver_id", "day", "start", "end", "state"]
            avail_df = load_or_create_csv(st.session_state.file_paths["caregivers_availability"], avail_cols)

            # if new caregiver selected, create temp id and empty rows
            if sel == "-- New Caregiver --":
                st.info("Create a caregiver in 'Caregiver List' before saving availability.")
                # show blank editor for convenience (not saved until caregiver has id)
                temp_df = pd.DataFrame(columns=avail_cols)
                cfg = {
                    "day": st.column_config.SelectboxColumn("Day", options=DAYS),
                    "start": st.column_config.SelectboxColumn("Start", options=TIME_OPTS),
                    "end": st.column_config.SelectboxColumn("End", options=TIME_OPTS),
                    "state": st.column_config.SelectboxColumn("State", options=["available", "prefer_not"])
                }
                edited_temp = st.data_editor(temp_df, column_config=cfg, hide_index=True, num_rows="dynamic", width="stretch", key="avail_temp")
            else:
                # find caregiver id
                row = caregivers_df[caregivers_df["name"] == sel].iloc[0]
                cg_id = row["caregiver_id"]
                # filter avail rows
                person_avail = avail_df[avail_df["caregiver_id"] == cg_id].copy()
                if person_avail.empty:
                    person_avail = pd.DataFrame(columns=avail_cols)
                cfg = {
                    "day": st.column_config.SelectboxColumn("Day", options=DAYS),
                    "start": st.column_config.SelectboxColumn("Start", options=TIME_OPTS),
                    "end": st.column_config.SelectboxColumn("End", options=TIME_OPTS),
                    "state": st.column_config.SelectboxColumn("State", options=["available", "prefer_not"])
                }
                edited_avail = st.data_editor(person_avail, column_config=cfg, hide_index=True, num_rows="dynamic", width="stretch", key=f"avail_editor_{cg_id}")

                if st.button("Save Availability", key=f"save_avail_{cg_id}"):
                    # attach caregiver_id to any rows missing it and persist
                    edited_avail["caregiver_id"] = cg_id
                    # ensure availability_id
                    for i, r in edited_avail.iterrows():
                        if not r.get("availability_id"):
                            edited_avail.at[i, "availability_id"] = mk_id("AV")
                    # rebuild global avail_df by removing old rows for cg and appending new
                    global_avail = avail_df[avail_df["caregiver_id"] != cg_id].copy()
                    combined = pd.concat([global_avail, edited_avail], ignore_index=True)
                    save_csv(combined, st.session_state.file_paths["caregivers_availability"])
                    st.success("Availability saved for caregiver.")

    # ------------------ CLIENTS ------------------
    with profile_tabs[1]:
        st.subheader("Clients")
        client_tabs = st.tabs(["Client List", "Shifts"])

        # ---- Client List ----
        with client_tabs[0]:
            st.markdown("**Client List (core profile data)**")
            client_list_cols = ["client_id", "name", "base_location", "importance", "scheduling_mode", "preferred_caregivers", "notes", "is_24_hour"]
            clients_df = load_or_create_csv(st.session_state.file_paths["clients_list"], client_list_cols)

            # fill missing client_id
            for i, r in clients_df.iterrows():
                if not r.get("client_id"):
                    clients_df.at[i, "client_id"] = mk_id("CL")

            # prepare caregiver name options for preferred caregivers multi-select
            caregivers_df = load_or_create_csv(st.session_state.file_paths["caregivers_list"], caregiver_list_cols)
            caregiver_names = caregivers_df["name"].fillna("").tolist()

            col_cfg = {
                "base_location": st.column_config.SelectboxColumn("Base Location", options=["Paradise", "Chico", "Oroville"]),
                "importance": st.column_config.SelectboxColumn("Importance", options=[str(i) for i in range(11)]),
                "scheduling_mode": st.column_config.SelectboxColumn("Scheduling Mode", options=["Maximize Client Preference", "Maximize Fairness"]),
                "preferred_caregivers": st.column_config.MultiSelectColumn("Preferred Caregivers", options=caregiver_names),
                "is_24_hour": st.column_config.CheckboxColumn("24-Hour Client")
            }

            edited_clients = st.data_editor(clients_df, column_config=col_cfg, hide_index=True, num_rows="dynamic", width="stretch", key="client_list_editor")

            if st.button("Save Client List", key="save_client_list"):
                # ensure client_id
                for i, r in edited_clients.iterrows():
                    if not r.get("client_id"):
                        edited_clients.at[i, "client_id"] = mk_id("CL")
                save_csv(edited_clients, st.session_state.file_paths["clients_list"])
                st.success("Client list saved.")

        # ---- Shifts ----
        with client_tabs[1]:
            st.markdown("**Client Shifts**")
            clients_df = load_or_create_csv(st.session_state.file_paths["clients_list"], ["client_id", "name"])
            client_names = ["-- New Client --"] + clients_df["name"].fillna("Unnamed").tolist()
            sel_client = st.selectbox("Select client to edit shifts", client_names, key="sel_client_shifts")

            fixed_cols = ["shift_id", "client_id", "day", "start", "end"]
            fixed_df = load_or_create_csv(st.session_state.file_paths["clients_fixed"], fixed_cols)

            flex_cols = ["flex_id", "client_id", "length_hours", "num_shifts", "start_day", "end_day", "start_time", "end_time"]
            flex_df = load_or_create_csv(st.session_state.file_paths["clients_flexible"], flex_cols)

            is_24hr = False
            if sel_client == "-- New Client --":
                st.info("Create client entry in 'Client List' first to persist shifts.")
            else:
                row = clients_df[clients_df["name"] == sel_client].iloc[0]
                client_id = row["client_id"]
                # fixed shifts for client
                person_fixed = fixed_df[fixed_df["client_id"] == client_id].copy()
                if person_fixed.empty:
                    person_fixed = pd.DataFrame(columns=fixed_cols)
                cfg_fixed = {
                    "day": st.column_config.SelectboxColumn("Day", options=DAYS),
                    "start": st.column_config.SelectboxColumn("Start", options=TIME_OPTS),
                    "end": st.column_config.SelectboxColumn("End", options=TIME_OPTS)
                }
                edited_fixed = st.data_editor(person_fixed, column_config=cfg_fixed, hide_index=True, num_rows="dynamic", width="stretch", key=f"fixed_editor_{client_id}")

                # flexible shifts
                person_flex = flex_df[flex_df["client_id"] == client_id].copy()
                if person_flex.empty:
                    person_flex = pd.DataFrame(columns=flex_cols)
                cfg_flex = {
                    "length_hours": st.column_config.NumberColumn("Length (hrs)"),
                    "num_shifts": st.column_config.NumberColumn("Count"),
                    "start_day": st.column_config.SelectboxColumn("Start Day", options=DAYS),
                    "end_day": st.column_config.SelectboxColumn("End Day", options=DAYS),
                    "start_time": st.column_config.SelectboxColumn("Start Time", options=TIME_OPTS),
                    "end_time": st.column_config.SelectboxColumn("End Time", options=TIME_OPTS)
                }
                edited_flex = st.data_editor(person_flex, column_config=cfg_flex, hide_index=True, num_rows="dynamic", width="stretch", key=f"flex_editor_{client_id}")

                if st.button("Save Shifts for Client", key=f"save_shifts_{client_id}"):
                    # ensure ids and client_id present
                    for i, r in edited_fixed.iterrows():
                        if not r.get("shift_id"):
                            edited_fixed.at[i, "shift_id"] = mk_id("SH")
                        edited_fixed.at[i, "client_id"] = client_id
                    for i, r in edited_flex.iterrows():
                        if not r.get("flex_id"):
                            edited_flex.at[i, "flex_id"] = mk_id("FX")
                        edited_flex.at[i, "client_id"] = client_id
                    # rebuild global tables
                    remaining_fixed = fixed_df[fixed_df["client_id"] != client_id].copy()
                    remaining_flex = flex_df[flex_df["client_id"] != client_id].copy()
                    new_fixed = pd.concat([remaining_fixed, edited_fixed], ignore_index=True)
                    new_flex = pd.concat([remaining_flex, edited_flex], ignore_index=True)
                    save_csv(new_fixed, st.session_state.file_paths["clients_fixed"])
                    save_csv(new_flex, st.session_state.file_paths["clients_flexible"])
                    st.success("Client shifts saved.")

# ==========================
# SCHEDULES (placeholder)
# ==========================
with main_tabs[1]:
    st.header("Schedules")
    sched_tabs = st.tabs(["Caregivers", "Clients", "Exceptions"])

    with sched_tabs[0]:
        st.subheader("Caregiver Schedule View")
        st.info("After running solver, select a caregiver to view their weekly schedule here.")

    with sched_tabs[1]:
        st.subheader("Client Schedule View")
        st.info("After running solver, select a client to view their weekly schedule here.")

    with sched_tabs[2]:
        st.subheader("Exceptions")
        st.info("Unfilled shifts and pending exceptions will appear here after solver run.")

# ==========================
# SETTINGS
# ==========================
with main_tabs[2]:
    st.header("Settings")
    st.write("Configure CSV file paths (defaults shown).")

    paths = st.session_state.file_paths
    for key in list(paths.keys()):
        new = st.text_input(f"Path for {key}", value=paths[key], key=f"path_{key}")
        paths[key] = new
    if st.button("Save Settings"):
        st.session_state.file_paths = paths
        st.success("Settings saved. CSV paths updated.")

# ==========================
# End
# ==========================


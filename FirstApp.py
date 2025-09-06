import streamlit as st
import pandas as pd
import os

st.set_page_config(layout="wide")

# ---------- File paths ----------
CAREGIVER_FILE = "caregivers.csv"
CAREGIVER_AVAIL_FILE = "caregiver_availability.csv"
CLIENT_FILE = "clients.csv"
CLIENT_FIXED_FILE = "client_shifts_fixed.csv"
CLIENT_FLEX_FILE = "client_shifts_flexible.csv"

# ---------- Ensure CSVs exist ----------
for f, cols in [
    (CAREGIVER_FILE, ["Name", "Base Location", "Notes"]),
    (CAREGIVER_AVAIL_FILE, ["Caregiver Name", "Day", "Start", "End", "Availability Type"]),
    (CLIENT_FILE, ["Name", "Base Location", "Importance", "Scheduling Mode", "Preferred Caregivers", "Notes"]),
    (CLIENT_FIXED_FILE, ["Client Name", "Day", "Start", "End"]),
    (CLIENT_FLEX_FILE, ["Client Name", "Length", "Number", "Start Day", "End Day", "Start Time", "End Time"]),
]:
    if not os.path.exists(f):
        pd.DataFrame(columns=cols).to_csv(f, index=False)

# ---------- Load Data ----------
def load_data():
    return {
        "caregivers": pd.read_csv(CAREGIVER_FILE),
        "caregiver_avail": pd.read_csv(CAREGIVER_AVAIL_FILE),
        "clients": pd.read_csv(CLIENT_FILE),
        "client_fixed": pd.read_csv(CLIENT_FIXED_FILE),
        "client_flex": pd.read_csv(CLIENT_FLEX_FILE),
    }

def save_data(dfs):
    dfs["caregivers"].to_csv(CAREGIVER_FILE, index=False)
    dfs["caregiver_avail"].to_csv(CAREGIVER_AVAIL_FILE, index=False)
    dfs["clients"].to_csv(CLIENT_FILE, index=False)
    dfs["client_fixed"].to_csv(CLIENT_FIXED_FILE, index=False)
    dfs["client_flex"].to_csv(CLIENT_FLEX_FILE, index=False)

dfs = load_data()

# ---------- Helpers for pre-populated rows ----------
def ensure_min_rows(df, min_rows, defaults=None):
    """Ensure at least min_rows exist in df, filling with defaults if provided."""
    if len(df) < min_rows:
        add_rows = min_rows - len(df)
        empty_rows = pd.DataFrame([defaults or {}] * add_rows, columns=df.columns)
        df = pd.concat([df, empty_rows], ignore_index=True)
    return df

# ---------- Tabs ----------
main_tabs = st.tabs(["Caregivers", "Clients", "Schedules", "Settings"])

# ================= CAREGIVERS TAB =================
with main_tabs[0]:
    sub_tabs = st.tabs(["Caregiver List", "Availability"])

    # --- Caregiver List ---
    with sub_tabs[0]:
        st.header("Caregiver List")

        edited_caregivers = st.data_editor(
            dfs["caregivers"],
            num_rows="dynamic",
            use_container_width=True,
            key="caregiver_list_editor"
        )

        if st.button("💾 Save Caregiver List"):
            dfs["caregivers"] = edited_caregivers
            save_data(dfs)
            st.success("Caregiver list saved!")

    # --- Availability ---
    with sub_tabs[1]:
        st.header("Caregiver Availability")

        if st.button("🔄 Refresh Caregivers"):
            dfs = load_data()
            st.experimental_rerun()

        caregiver_avail_updated = []
        for name in dfs["caregivers"]["Name"].unique():
            st.subheader(f"Availability for {name}")
            sub_df = dfs["caregiver_avail"][dfs["caregiver_avail"]["Caregiver Name"] == name]

            sub_df = ensure_min_rows(
                sub_df,
                min_rows=3,
                defaults={"Caregiver Name": name, "Day": "", "Start": "", "End": "", "Availability Type": ""}
            )

            edited = st.data_editor(
                sub_df,
                num_rows="dynamic",
                key=f"caregiver_avail_{name}",
                use_container_width=True
            )
            caregiver_avail_updated.append(edited)

        if st.button("💾 Save All Caregiver Availability"):
            dfs["caregiver_avail"] = pd.concat(caregiver_avail_updated, ignore_index=True)
            save_data(dfs)
            st.success("Caregiver availability saved!")

# ================= CLIENTS TAB =================
with main_tabs[1]:
    sub_tabs = st.tabs(["Client List", "Shifts"])

    # --- Client List ---
    with sub_tabs[0]:
        st.header("Client List")

        edited_clients = st.data_editor(
            dfs["clients"],
            num_rows="dynamic",
            use_container_width=True,
            key="client_list_editor"
        )

        if st.button("💾 Save Client List"):
            dfs["clients"] = edited_clients
            save_data(dfs)
            st.success("Client list saved!")

    # --- Shifts ---
    with sub_tabs[1]:
        st.header("Client Shifts")

        if st.button("🔄 Refresh Clients"):
            dfs = load_data()
            st.experimental_rerun()

        client_fixed_updated = []
        client_flex_updated = []
        for name in dfs["clients"]["Name"].unique():
            st.subheader(f"Shifts for {name}")

            st.markdown("**Fixed Shifts**")
            fixed_df = dfs["client_fixed"][dfs["client_fixed"]["Client Name"] == name]
            fixed_df = ensure_min_rows(
                fixed_df,
                min_rows=2,
                defaults={"Client Name": name, "Day": "", "Start": "", "End": ""}
            )

            fixed_edited = st.data_editor(
                fixed_df,
                num_rows="dynamic",
                key=f"client_fixed_{name}",
                use_container_width=True
            )
            client_fixed_updated.append(fixed_edited)

            st.markdown("**Flexible Shifts**")
            flex_df = dfs["client_flex"][dfs["client_flex"]["Client Name"] == name]
            flex_df = ensure_min_rows(
                flex_df,
                min_rows=2,
                defaults={
                    "Client Name": name,
                    "Length": "",
                    "Number": "",
                    "Start Day": "",
                    "End Day": "",
                    "Start Time": "",
                    "End Time": "",
                }
            )

            flex_edited = st.data_editor(
                flex_df,
                num_rows="dynamic",
                key=f"client_flex_{name}",
                use_container_width=True
            )
            client_flex_updated.append(flex_edited)

        if st.button("💾 Save All Client Shifts"):
            dfs["client_fixed"] = pd.concat(client_fixed_updated, ignore_index=True)
            dfs["client_flex"] = pd.concat(client_flex_updated, ignore_index=True)
            save_data(dfs)
            st.success("Client shifts saved!")

# ================= SCHEDULES TAB =================
with main_tabs[2]:
    sub_tabs = st.tabs(["Caregivers", "Clients", "Exceptions"])

    with sub_tabs[0]:
        st.header("Caregiver Schedules")
        st.info("📅 This will show caregiver schedules after solver integration.")

    with sub_tabs[1]:
        st.header("Client Schedules")
        st.info("📅 This will show client schedules after solver integration.")

    with sub_tabs[2]:
        st.header("Scheduling Exceptions")
        st.info("⚠️ This will list unresolved hard-constraint exceptions.")

# ================= SETTINGS TAB =================
with main_tabs[3]:
    st.header("Settings")
    st.text("Future options: configure CSV file paths, solver parameters, etc.")

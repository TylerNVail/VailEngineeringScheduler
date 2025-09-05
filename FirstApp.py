import streamlit as st
import pandas as pd
from datetime import time

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Homecare Scheduler",
    layout="wide",
)

# ========== LOAD DATA ==========
def load_csv(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        return pd.DataFrame()

caregivers_df = load_csv("caregivers.csv")
clients_df = load_csv("clients.csv")
approvals_df = load_csv("approvals.csv")

# ========== SIDEBAR ==========
st.sidebar.title("Navigation")
tabs = ["Caregivers", "Clients", "Schedule", "Exceptions", "Settings"]
selected_tab = st.sidebar.radio("Go to:", tabs)

# ========== CAREGIVER TAB ==========
if selected_tab == "Caregivers":
    st.title("Caregiver Profiles")
    st.write("Add new caregivers or edit existing profiles.")

    with st.form("add_caregiver"):
        name = st.text_input("Caregiver Name")
        base_location = st.selectbox("Base Location", ["Paradise", "Chico", "Oroville"])
        min_hours = st.number_input("Minimum Hours", 0, 80, 20)
        max_hours = st.number_input("Maximum Hours", 0, 80, 40)
        as_many_as_possible = st.checkbox("As many hours as possible")
        submitted = st.form_submit_button("Save Caregiver")

        if submitted:
            new_row = {
                "name": name,
                "base_location": base_location,
                "min_hours": min_hours,
                "max_hours": max_hours,
                "as_many_as_possible": as_many_as_possible
            }
            caregivers_df = pd.concat([caregivers_df, pd.DataFrame([new_row])], ignore_index=True)
            caregivers_df.to_csv("caregivers.csv", index=False)
            st.success(f"Caregiver {name} saved!")

    if not caregivers_df.empty:
        st.subheader("Existing Caregivers")
        st.dataframe(caregivers_df)

# ========== CLIENT TAB ==========
elif selected_tab == "Clients":
    st.title("Client Profiles")
    st.write("Add new clients and their care needs.")

    with st.form("add_client"):
        name = st.text_input("Client Name")
        base_location = st.selectbox("Base Location", ["Paradise", "Chico", "Oroville"])
        importance = st.slider("Importance (0-10)", 0, 10, 5)
        scheduling_mode = st.selectbox("Scheduling Mode", ["Maximize Client Preference", "Maximize Fairness"])
        preferred_caregivers = st.text_area("Preferred Caregivers (comma-separated)")
        submitted = st.form_submit_button("Save Client")

        if submitted:
            new_row = {
                "name": name,
                "base_location": base_location,
                "importance": importance,
                "scheduling_mode": scheduling_mode,
                "preferred_caregivers": preferred_caregivers
            }
            clients_df = pd.concat([clients_df, pd.DataFrame([new_row])], ignore_index=True)
            clients_df.to_csv("clients.csv", index=False)
            st.success(f"Client {name} saved!")

    if not clients_df.empty:
        st.subheader("Existing Clients")
        st.dataframe(clients_df)

# ========== SCHEDULE TAB ==========
elif selected_tab == "Schedule":
    st.title("Weekly Schedule")
    st.write("Auto-generate or manually review schedules.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Schedule"):
            # Placeholder for solver integration
            st.info("Running solver... (to be integrated)")
    with col2:
        if st.button("Save Schedule"):
            st.success("Schedule saved to CSV.")

    st.write("📅 Current Week’s Schedule")
    # Placeholder schedule table
    schedule_df = pd.DataFrame({
        "Day": ["Mon", "Tue", "Wed"],
        "Client": ["Client A", "Client B", "Client C"],
        "Caregiver": ["John Doe", "Jane Smith", "Joe Brown"],
        "Start": ["09:00", "10:00", "14:00"],
        "End": ["13:00", "14:00", "18:00"]
    })
    st.dataframe(schedule_df)

# ========== EXCEPTIONS TAB ==========
elif selected_tab == "Exceptions":
    st.title("Exceptions Review")
    st.write("Approve or decline hard-constraint exceptions.")

    # Placeholder exceptions
    exceptions = [
        {"id": 1, "desc": "Joe Doe travels Oroville → Chico (Tue 14–20h)", "auto": False},
        {"id": 2, "desc": "Jane Smith split 8h block into 2 shifts (Wed 09–17h)", "auto": True},
    ]

    approvals = []
    for exc in exceptions:
        col1, col2 = st.columns([4,1])
        with col1:
            label = f"{'⚡' if exc['auto'] else ''} {exc['desc']}"
            st.write(label)
        with col2:
            decision = st.radio(
                f"Decision {exc['id']}",
                ["Pending", "Approve", "Decline"],
                key=f"exc_{exc['id']}"
            )
            approvals.append((exc["id"], decision))

    if st.button("Resolve Exceptions"):
        st.write("Resolving based on decisions...")
        st.json(approvals)

# ========== SETTINGS TAB ==========
elif selected_tab == "Settings":
    st.title("Settings")
    st.write("General app configuration.")

    if st.button("Reset All Data"):
        caregivers_df.to_csv("caregivers.csv", index=False)
        clients_df.to_csv("clients.csv", index=False)
        approvals_df.to_csv("approvals.csv", index=False)
        st.warning("All data reset!")

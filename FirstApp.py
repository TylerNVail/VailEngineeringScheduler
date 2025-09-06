import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Homecare Scheduler", layout="wide")

# ========== HELPERS ==========
def load_csv(filename):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        return pd.DataFrame()

caregivers_df = load_csv("caregivers.csv")
clients_df = load_csv("clients.csv")
approvals_df = load_csv("approvals.csv")

days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
time_slots = [f"{(datetime(2000,1,1,0,0) + timedelta(minutes=30*i)).strftime('%H:%M')}" for i in range(48)]

# ========== MAIN NAVIGATION ==========
main_tabs = st.tabs(["Profiles", "Schedules", "Settings"])

# ====== PROFILES TAB ======
with main_tabs[0]:
    st.header("Profiles")
    sub_tabs = st.tabs(["Caregivers", "Clients"])

    # --- Caregivers ---
    with sub_tabs[0]:
        st.subheader("Caregiver Profiles")
        caregiver_names = ["New"] + caregivers_df["name"].tolist() if not caregivers_df.empty else ["New"]
        selected = st.selectbox("Select Caregiver", caregiver_names)

        if selected == "New":
            name = st.text_input("Name")
            base_location = st.selectbox("Base Location", ["Paradise", "Chico", "Oroville"])
            min_hours = st.number_input("Min Hours", 0, 80, 20)
            max_hours = st.number_input("Max Hours", 0, 80, 40)
            as_many = st.checkbox("As many hours as possible")
            st.markdown("### Weekly Availability")
            caregiver_matrix = pd.DataFrame(
                [["Unavailable"]*7 for _ in time_slots], 
                columns=days, index=time_slots
            )
            st.dataframe(caregiver_matrix, use_container_width=True)

        else:
            row = caregivers_df[caregivers_df["name"] == selected].iloc[0]
            st.text_input("Name", value=row["name"])
            st.selectbox("Base Location", ["Paradise", "Chico", "Oroville"], index=["Paradise","Chico","Oroville"].index(row["base_location"]))
            st.number_input("Min Hours", 0, 80, int(row["min_hours"]))
            st.number_input("Max Hours", 0, 80, int(row["max_hours"]))
            st.checkbox("As many hours as possible", value=row["as_many_as_possible"])
            st.markdown("### Weekly Availability")
            # TODO: Load saved caregiver availability matrix
            st.info("Availability matrix loading placeholder")

    # --- Clients ---
    with sub_tabs[1]:
        st.subheader("Client Profiles")
        client_names = ["New"] + clients_df["name"].tolist() if not clients_df.empty else ["New"]
        selected = st.selectbox("Select Client", client_names)

        if selected == "New":
            name = st.text_input("Name")
            base_location = st.selectbox("Base Location", ["Paradise", "Chico", "Oroville"])
            importance = st.slider("Importance (0–10)", 0, 10, 5)
            mode = st.selectbox("Scheduling Mode", ["Maximize Client Preference", "Maximize Fairness"])
            preferred = st.text_area("Preferred Caregivers (comma-separated)")
            st.markdown("### Requested Coverage")
            client_matrix = pd.DataFrame(
                [["None"]*7 for _ in time_slots], 
                columns=days, index=time_slots
            )
            st.dataframe(client_matrix, use_container_width=True)

        else:
            row = clients_df[clients_df["name"] == selected].iloc[0]
            st.text_input("Name", value=row["name"])
            st.selectbox("Base Location", ["Paradise", "Chico", "Oroville"], index=["Paradise","Chico","Oroville"].index(row["base_location"]))
            st.slider("Importance (0–10)", 0, 10, int(row["importance"]))
            st.selectbox("Scheduling Mode", ["Maximize Client Preference", "Maximize Fairness"], index=0 if row["scheduling_mode"]=="Maximize Client Preference" else 1)
            st.text_area("Preferred Caregivers", value=row["preferred_caregivers"])
            st.markdown("### Requested Coverage")
            # TODO: Load saved client coverage matrix
            st.info("Client coverage matrix loading placeholder")

# ====== SCHEDULES TAB ======
with main_tabs[1]:
    st.header("Schedules")
    sub_tabs = st.tabs(["Caregivers", "Clients", "Exceptions"])

    # --- Caregivers Schedule ---
    with sub_tabs[0]:
        st.subheader("Caregiver Schedule")
        if not caregivers_df.empty:
            caregiver = st.selectbox("Select Caregiver", caregivers_df["name"].tolist())
            st.markdown(f"### Weekly Schedule for {caregiver}")
            # TODO: Replace with actual schedule
            schedule_matrix = pd.DataFrame(
                [[""]*7 for _ in time_slots], 
                columns=days, index=time_slots
            )
            st.dataframe(schedule_matrix, use_container_width=True)

    # --- Clients Schedule ---
    with sub_tabs[1]:
        st.subheader("Client Schedule")
        if not clients_df.empty:
            client = st.selectbox("Select Client", clients_df["name"].tolist())
            st.markdown(f"### Weekly Schedule for {client}")
            # TODO: Replace with actual schedule
            schedule_matrix = pd.DataFrame(
                [[""]*7 for _ in time_slots], 
                columns=days, index=time_slots
            )
            st.dataframe(schedule_matrix, use_container_width=True)

    # --- Exceptions ---
    with sub_tabs[2]:
        st.subheader("Exceptions Review")
        # TODO: Replace with actual unsolved shifts
        exceptions = pd.DataFrame([
            {"Client": "Client A", "Importance": 10, "Shift": "Tue 09–13", "Type": "Fixed"},
            {"Client": "Client B", "Importance": 7, "Shift": "Wed 14–18", "Type": "Flexible"},
        ])
        st.dataframe(exceptions, use_container_width=True)
        st.write("Approve or Decline Exceptions Below:")
        # Placeholder for decision controls
        st.button("Resolve Exceptions")

# ====== SETTINGS TAB ======
with main_tabs[2]:
    st.header("Settings")
    if st.button("Reset All Data"):
        caregivers_df.to_csv("caregivers.csv", index=False)
        clients_df.to_csv("clients.csv", index=False)
        approvals_df.to_csv("approvals.csv", index=False)
        st.warning("All data reset!")

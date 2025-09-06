import pandas as pd
import streamlit as st

# -----------------------
# --- Helper Functions ---
# -----------------------
def load_csv(path, default_columns):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=default_columns)
    for col in default_columns:
        if col not in df.columns:
            df[col] = ""
    return df.fillna("")

def save_csv(df, path):
    df.to_csv(path, index=False)

# -----------------------
# --- CSV Paths ---
# -----------------------
caregiver_csv = "caregivers.csv"
caregiver_availability_csv = "caregiver_availability.csv"
client_csv = "clients.csv"
client_fixed_shifts_csv = "client_fixed_shifts.csv"
client_flexible_shifts_csv = "client_flexible_shifts.csv"

# -----------------------
# --- Dropdown Options ---
# -----------------------
days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
availability_types = ["Available", "Preferred Unavailable"]
shift_types = ["Fixed", "Flexible"]
base_locations = ["Paradise", "Chico", "Oroville"]

# 30-minute increments
def generate_time_options():
    times = []
    for h in range(24):
        for m in [0, 30]:
            times.append(f"{h:02d}:{m:02d}")
    return times
time_options = generate_time_options()

# -----------------------
# --- Main Tabs ---
# -----------------------
tab_main = st.tabs(["Profiles", "Schedules", "Settings"])

# -----------------------
# --- Profiles Tab ---
# -----------------------
with tab_main[0]:
    profile_tabs = st.tabs(["Caregivers", "Clients"])

    # --- Caregivers ---
    with profile_tabs[0]:
        caregiver_subtabs = st.tabs(["Caregiver List", "Availability"])

        # Caregiver List
        with caregiver_subtabs[0]:
            st.subheader("Caregiver List")
            caregiver_columns = ["Name", "Base Location", "Notes"]
            caregivers_df = load_csv(caregiver_csv, caregiver_columns)

            edited_caregivers = st.data_editor(
                caregivers_df,
                column_config={
                    "Name": st.column_config.TextColumn("Name"),
                    "Base Location": st.column_config.ChoiceColumn("Base Location", options=base_locations),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                num_rows="dynamic",
                key="caregiver_list_editor",
                width="stretch"
            )
            if st.button("Save Caregivers"):
                save_csv(edited_caregivers, caregiver_csv)
                st.success("Caregivers saved!")

        # Caregiver Availability
        with caregiver_subtabs[1]:
            st.subheader("Caregiver Availability")
            availability_columns = ["Caregiver Name", "Day of Week", "Start", "End", "Availability Type", "Notes"]
            availability_df = load_csv(caregiver_availability_csv, availability_columns)

            edited_availability = st.data_editor(
                availability_df,
                column_config={
                    "Caregiver Name": st.column_config.ChoiceColumn("Caregiver Name", options=caregivers_df["Name"].tolist()),
                    "Day of Week": st.column_config.ChoiceColumn("Day of Week", options=days_of_week),
                    "Start": st.column_config.ChoiceColumn("Start Time", options=time_options),
                    "End": st.column_config.ChoiceColumn("End Time", options=time_options),
                    "Availability Type": st.column_config.ChoiceColumn("Availability Type", options=availability_types),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                num_rows="dynamic",
                key="caregiver_availability_editor",
                width="stretch"
            )
            if st.button("Save Availability"):
                save_csv(edited_availability, caregiver_availability_csv)
                st.success("Caregiver availability saved!")

    # --- Clients ---
    with profile_tabs[1]:
        client_subtabs = st.tabs(["Client List", "Fixed Shifts", "Flexible Shifts"])

        # Client List
        with client_subtabs[0]:
            st.subheader("Client List")
            client_columns = ["Name", "Base Location", "Importance", "Scheduling Mode", "Preferred Caregivers", "Notes"]
            clients_df = load_csv(client_csv, client_columns)
            clients_df["Importance"] = pd.to_numeric(clients_df["Importance"], errors="coerce").fillna(0).astype(int)

            edited_clients = st.data_editor(
                clients_df,
                column_config={
                    "Name": st.column_config.TextColumn("Name"),
                    "Base Location": st.column_config.ChoiceColumn("Base Location", options=base_locations),
                    "Importance": st.column_config.NumberColumn("Importance (0-10)", min_value=0, max_value=10, step=1),
                    "Scheduling Mode": st.column_config.TextColumn("Scheduling Mode"),
                    "Preferred Caregivers": st.column_config.TextColumn("Preferred Caregivers (comma separated)"),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                num_rows="dynamic",
                key="client_list_editor",
                width="stretch"
            )
            if st.button("Save Clients"):
                save_csv(edited_clients, client_csv)
                st.success("Clients saved!")

        # Fixed Shifts
        with client_subtabs[1]:
            st.subheader("Client Fixed Shifts")
            fixed_columns = ["Client Name", "Day of Week", "Start", "End", "Notes"]
            fixed_df = load_csv(client_fixed_shifts_csv, fixed_columns)

            edited_fixed = st.data_editor(
                fixed_df,
                column_config={
                    "Client Name": st.column_config.ChoiceColumn("Client Name", options=clients_df["Name"].tolist()),
                    "Day of Week": st.column_config.ChoiceColumn("Day of Week", options=days_of_week),
                    "Start": st.column_config.ChoiceColumn("Start Time", options=time_options),
                    "End": st.column_config.ChoiceColumn("End Time", options=time_options),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                num_rows="dynamic",
                key="client_fixed_editor",
                width="stretch"
            )
            if st.button("Save Fixed Shifts"):
                save_csv(edited_fixed, client_fixed_shifts_csv)
                st.success("Fixed shifts saved!")

        # Flexible Shifts
        with client_subtabs[2]:
            st.subheader("Client Flexible Shifts")
            flex_columns = ["Client Name", "Length (hrs)", "Number of Shifts", "Start Day", "End Day", "Start Time", "End Time", "Notes"]
            flex_df = load_csv(client_flexible_shifts_csv, flex_columns)

            edited_flex = st.data_editor(
                flex_df,
                column_config={
                    "Client Name": st.column_config.ChoiceColumn("Client Name", options=clients_df["Name"].tolist()),
                    "Length (hrs)": st.column_config.NumberColumn("Length of Shift (hrs)", min_value=1, step=1),
                    "Number of Shifts": st.column_config.NumberColumn("Number of Shifts", min_value=1, step=1),
                    "Start Day": st.column_config.ChoiceColumn("Start Day", options=days_of_week),
                    "End Day": st.column_config.ChoiceColumn("End Day", options=days_of_week),
                    "Start Time": st.column_config.ChoiceColumn("Start Time", options=time_options),
                    "End Time": st.column_config.ChoiceColumn("End Time", options=time_options),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                num_rows="dynamic",
                key="client_flexible_editor",
                width="stretch"
            )
            if st.button("Save Flexible Shifts"):
                save_csv(edited_flex, client_flexible_shifts_csv)
                st.success("Flexible shifts saved!")

# -----------------------
# --- Schedules Tab ---
# -----------------------
with tab_main[1]:
    schedule_subtabs = st.tabs(["Caregivers", "Clients", "Exceptions"])
    for subtab in schedule_subtabs:
        with subtab:
            st.info("Schedules will be displayed here after solver integration.")

# -----------------------
# --- Settings Tab ---
# -----------------------
with tab_main[2]:
    st.subheader("Settings")
    st.text("Currently using default CSV paths. Update paths here if needed.")
    caregiver_csv = st.text_input("Caregiver CSV Path", caregiver_csv)
    caregiver_availability_csv = st.text_input("Caregiver Availability CSV Path", caregiver_availability_csv)
    client_csv = st.text_input("Client CSV Path", client_csv)
    client_fixed_shifts_csv = st.text_input("Client Fixed Shifts CSV Path", client_fixed_shifts_csv)
    client_flexible_shifts_csv = st.text_input("Client Flexible Shifts CSV Path", client_flexible_shifts_csv)
    st.info("These paths will be used when saving/loading data.")

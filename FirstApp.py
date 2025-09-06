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

# Default CSV paths
caregiver_csv = "caregivers.csv"
caregiver_availability_csv = "caregiver_availability.csv"
client_csv = "clients.csv"
client_shifts_csv = "client_shifts.csv"

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
                    "Base Location": st.column_config.TextColumn("Base Location"),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                num_rows="dynamic",
                key="caregiver_list_editor"
            )
            if st.button("Save Caregivers"):
                save_csv(edited_caregivers, caregiver_csv)
                st.success("Caregivers saved!")

        # Caregiver Availability
        with caregiver_subtabs[1]:
            st.subheader("Caregiver Availability")
            availability_columns = ["Day of Week", "Start", "End", "Availability Type", "Notes"]
            availability_df = load_csv(caregiver_availability_csv, availability_columns)

            edited_availability = st.data_editor(
                availability_df,
                column_config={
                    "Day of Week": st.column_config.TextColumn("Day of Week"),
                    "Start": st.column_config.TextColumn("Start Time (24h)"),
                    "End": st.column_config.TextColumn("End Time (24h)"),
                    "Availability Type": st.column_config.TextColumn("Availability Type"),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                num_rows="dynamic",
                key="caregiver_availability_editor"
            )
            if st.button("Save Availability"):
                save_csv(edited_availability, caregiver_availability_csv)
                st.success("Caregiver availability saved!")

    # --- Clients ---
    with profile_tabs[1]:
        client_subtabs = st.tabs(["Client List", "Shifts"])

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
                    "Base Location": st.column_config.TextColumn("Base Location"),
                    "Importance": st.column_config.NumberColumn("Importance (0-10)", min_value=0, max_value=10, step=1),
                    "Scheduling Mode": st.column_config.TextColumn("Scheduling Mode"),
                    "Preferred Caregivers": st.column_config.TextColumn("Preferred Caregivers (comma separated)"),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                num_rows="dynamic",
                key="client_list_editor"
            )
            if st.button("Save Clients"):
                save_csv(edited_clients, client_csv)
                st.success("Clients saved!")

        # Client Shifts
        with client_subtabs[1]:
            st.subheader("Client Shifts")
            shift_columns = ["Day of Week", "Start", "End", "Shift Type", "Notes"]
            shifts_df = load_csv(client_shifts_csv, shift_columns)

            edited_shifts = st.data_editor(
                shifts_df,
                column_config={
                    "Day of Week": st.column_config.TextColumn("Day of Week"),
                    "Start": st.column_config.TextColumn("Start Time (24h)"),
                    "End": st.column_config.TextColumn("End Time (24h)"),
                    "Shift Type": st.column_config.TextColumn("Shift Type (Fixed/Flexible)"),
                    "Notes": st.column_config.TextColumn("Notes"),
                },
                num_rows="dynamic",
                key="client_shifts_editor"
            )
            if st.button("Save Shifts"):
                save_csv(edited_shifts, client_shifts_csv)
                st.success("Client shifts saved!")

# -----------------------
# --- Schedules Tab ---
# -----------------------
with tab_main[1]:
    schedule_subtabs = st.tabs(["Caregivers", "Clients", "Exceptions"])
    for subtab in schedule_subtabs:
        with subtab:
            st.info("Schedules will be displayed here after solver is integrated.")

# -----------------------
# --- Settings Tab ---
# -----------------------
with tab_main[2]:
    st.subheader("Settings")
    st.text("Currently using default CSV paths. Update paths here if needed.")
    caregiver_csv = st.text_input("Caregiver CSV Path", caregiver_csv)
    caregiver_availability_csv = st.text_input("Caregiver Availability CSV Path", caregiver_availability_csv)
    client_csv = st.text_input("Client CSV Path", client_csv)
    client_shifts_csv = st.text_input("Client Shifts CSV Path", client_shifts_csv)
    st.info("These paths will be used when saving/loading data.")

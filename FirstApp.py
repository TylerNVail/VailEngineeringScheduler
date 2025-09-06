import streamlit as st
import pandas as pd

# ================================
# Helpers for loading/saving CSVs
# ================================
def load_csv(path, columns):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=columns)

def save_csv(path, df):
    df.to_csv(path, index=False)

# ================================
# File paths (local CSVs)
# ================================
CAREGIVER_FILE = "caregivers.csv"
CLIENT_FILE = "clients.csv"
AVAILABILITY_FILE = "availability.csv"
FIXED_SHIFTS_FILE = "fixed_shifts.csv"
FLEX_SHIFTS_FILE = "flexible_shifts.csv"

# ================================
# Initialize DataFrames
# ================================
caregivers_df = load_csv(CAREGIVER_FILE, ["name", "base_location", "notes"])
clients_df = load_csv(CLIENT_FILE, ["name", "base_location", "importance", "scheduling_mode", "preferred_caregivers", "notes"])
availability_df = load_csv(AVAILABILITY_FILE, ["caregiver_name", "day_of_week", "start_time", "end_time", "availability_type"])
fixed_shifts_df = load_csv(FIXED_SHIFTS_FILE, ["client_name", "day_of_week", "start_time", "end_time"])
flex_shifts_df = load_csv(FLEX_SHIFTS_FILE, ["client_name", "length_hours", "num_shifts", "start_day", "end_day", "start_time", "end_time"])

# ================================
# Streamlit App Layout
# ================================
st.set_page_config(page_title="Scheduling App", layout="wide")

st.title("🏥 Homecare Scheduling Tool")

main_tabs = st.tabs(["Profiles", "Schedules", "Settings"])

# ================================
# Profiles Tab
# ================================
with main_tabs[0]:
    st.subheader("Profiles")
    profile_tabs = st.tabs(["Caregivers", "Clients"])

    # -------------------- Caregivers --------------------
    with profile_tabs[0]:
        cg_tabs = st.tabs(["Caregiver List", "Availability"])

        # Caregiver List
        with cg_tabs[0]:
            st.markdown("### Caregiver List")
            edited_caregivers = st.data_editor(
                caregivers_df,
                num_rows="dynamic",
                column_config={
                    "name": st.column_config.TextColumn("Name"),
                    "base_location": st.column_config.SelectboxColumn("Base Location", options=["Paradise", "Chico", "Oroville"]),
                    "notes": st.column_config.TextColumn("Notes")
                },
                key="caregiver_list_editor"
            )
            if st.button("💾 Save Caregivers"):
                save_csv(CAREGIVER_FILE, edited_caregivers)
                st.success("Caregivers saved!")

        # Caregiver Availability
        with cg_tabs[1]:
            st.markdown("### Caregiver Availability")
            caregiver_names = caregivers_df["name"].dropna().unique().tolist()
            selected_cg = st.selectbox("Select Caregiver", options=["-"] + caregiver_names, key="avail_select")

            if selected_cg != "-":
                filtered = availability_df[availability_df["caregiver_name"] == selected_cg]
                edited_avail = st.data_editor(
                    filtered,
                    num_rows="dynamic",
                    column_config={
                        "day_of_week": st.column_config.SelectboxColumn("Day of Week", options=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]),
                        "start_time": st.column_config.SelectboxColumn("Start of Availability", options=[f"{h:02d}:{m:02d}" for h in range(24) for m in (0,30)]),
                        "end_time": st.column_config.SelectboxColumn("End of Availability", options=[f"{h:02d}:{m:02d}" for h in range(24) for m in (0,30)]),
                        "availability_type": st.column_config.SelectboxColumn("Availability Type", options=["Available", "Preferred Unavailable"])
                    },
                    key=f"avail_editor_{selected_cg}"
                )
                if st.button("💾 Save Availability"):
                    # remove old caregiver rows, replace with new
                    availability_df.drop(availability_df[availability_df["caregiver_name"] == selected_cg].index, inplace=True)
                    edited_avail["caregiver_name"] = selected_cg
                    availability_df_updated = pd.concat([availability_df, edited_avail], ignore_index=True)
                    save_csv(AVAILABILITY_FILE, availability_df_updated)
                    st.success("Availability saved!")

    # -------------------- Clients --------------------
    with profile_tabs[1]:
        cl_tabs = st.tabs(["Client List", "Shifts"])

        # Client List
        with cl_tabs[0]:
            st.markdown("### Client List")
            edited_clients = st.data_editor(
                clients_df,
                num_rows="dynamic",
                column_config={
                    "name": st.column_config.TextColumn("Name"),
                    "base_location": st.column_config.SelectboxColumn("Base Location", options=["Paradise", "Chico", "Oroville"]),
                    "importance": st.column_config.NumberColumn("Importance (0–10)", min_value=0, max_value=10),
                    "scheduling_mode": st.column_config.SelectboxColumn("Scheduling Mode", options=["Maximize Fairness", "Maximize Client Preference"]),
                    "preferred_caregivers": st.column_config.TextColumn("Preferred Caregivers (comma separated)"),
                    "notes": st.column_config.TextColumn("Notes")
                },
                key="client_list_editor"
            )
            if st.button("💾 Save Clients"):
                save_csv(CLIENT_FILE, edited_clients)
                st.success("Clients saved!")

        # Client Shifts
        with cl_tabs[1]:
            st.markdown("### Client Shifts")
            client_names = clients_df["name"].dropna().unique().tolist()
            selected_client = st.selectbox("Select Client", options=["-"] + client_names, key="shift_select")

            if selected_client != "-":
                st.write("#### Fixed Shifts")
                fixed_for_client = fixed_shifts_df[fixed_shifts_df["client_name"] == selected_client]
                edited_fixed = st.data_editor(
                    fixed_for_client,
                    num_rows="dynamic",
                    column_config={
                        "day_of_week": st.column_config.SelectboxColumn("Day of Week", options=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]),
                        "start_time": st.column_config.SelectboxColumn("Start Time", options=[f"{h:02d}:{m:02d}" for h in range(24) for m in (0,30)]),
                        "end_time": st.column_config.SelectboxColumn("End Time", options=[f"{h:02d}:{m:02d}" for h in range(24) for m in (0,30)])
                    },
                    key=f"fixed_editor_{selected_client}"
                )
                if st.button("💾 Save Fixed Shifts"):
                    fixed_shifts_df.drop(fixed_shifts_df[fixed_shifts_df["client_name"] == selected_client].index, inplace=True)
                    edited_fixed["client_name"] = selected_client
                    fixed_updated = pd.concat([fixed_shifts_df, edited_fixed], ignore_index=True)
                    save_csv(FIXED_SHIFTS_FILE, fixed_updated)
                    st.success("Fixed shifts saved!")

                st.write("#### Flexible Shifts")
                flex_for_client = flex_shifts_df[flex_shifts_df["client_name"] == selected_client]
                edited_flex = st.data_editor(
                    flex_for_client,
                    num_rows="dynamic",
                    column_config={
                        "length_hours": st.column_config.NumberColumn("Length of Shift (hrs)", min_value=1),
                        "num_shifts": st.column_config.NumberColumn("Number of Shifts", min_value=1),
                        "start_day": st.column_config.SelectboxColumn("Start Day", options=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]),
                        "end_day": st.column_config.SelectboxColumn("End Day", options=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]),
                        "start_time": st.column_config.SelectboxColumn("Start Time", options=[f"{h:02d}:{m:02d}" for h in range(24) for m in (0,30)]),
                        "end_time": st.column_config.SelectboxColumn("End Time", options=[f"{h:02d}:{m:02d}" for h in range(24) for m in (0,30)])
                    },
                    key=f"flex_editor_{selected_client}"
                )
                if st.button("💾 Save Flexible Shifts"):
                    flex_shifts_df.drop(flex_shifts_df[flex_shifts_df["client_name"] == selected_client].index, inplace=True)
                    edited_flex["client_name"] = selected_client
                    flex_updated = pd.concat([flex_shifts_df, edited_flex], ignore_index=True)
                    save_csv(FLEX_SHIFTS_FILE, flex_updated)
                    st.success("Flexible shifts saved!")

# ================================
# Placeholder Tabs
# ================================
with main_tabs[1]:
    st.subheader("Schedules")
    st.info("Schedules view coming soon…")

with main_tabs[2]:
    st.subheader("Settings")
    st.info("Settings options coming soon…")

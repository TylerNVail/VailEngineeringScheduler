import streamlit as st
import pandas as pd
import os

# ==========================
# Default CSV file paths
# ==========================
def get_default_paths():
    return {
        "caregivers": "caregivers.csv",
        "clients_fixed": "clients_fixed.csv",
        "clients_flexible": "clients_flexible.csv",
        "approvals": "approvals.csv"
    }

# ==========================
# Utility: load or create CSV
# ==========================
def load_or_create_csv(path, columns):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=columns)
        df.to_csv(path, index=False)
        return df

# ==========================
# App State Management
# ==========================
if "file_paths" not in st.session_state:
    st.session_state.file_paths = get_default_paths()

# ==========================
# Tabs Layout
# ==========================
tabs = st.tabs(["Profiles", "Schedules", "Settings"])

# ==========================
# PROFILES TAB
# ==========================
with tabs[0]:
    sub_tabs = st.tabs(["Caregivers", "Clients"])

    # --- Caregivers ---
    with sub_tabs[0]:
        st.header("Caregiver Profiles")
        caregiver_file = st.session_state.file_paths["caregivers"]

        caregiver_cols = ["Name", "Base Location", "Day of Week", "Start of Availability", "End of Availability", "Availability Type"]
        caregivers_df = load_or_create_csv(caregiver_file, caregiver_cols)

        edited_caregivers = st.data_editor(
            caregivers_df,
            num_rows="dynamic",
            use_container_width=True,
            key="caregiver_editor"
        )

        if st.button("Save Caregivers"):
            edited_caregivers.to_csv(caregiver_file, index=False)
            st.success("Caregivers saved!")

    # --- Clients ---
    with sub_tabs[1]:
        st.header("Client Profiles")

        fixed_file = st.session_state.file_paths["clients_fixed"]
        flexible_file = st.session_state.file_paths["clients_flexible"]

        # Checkbox for 24-hour client
        is_24hr = st.checkbox("24-Hour Client", key="client_24hr")

        st.subheader("Fixed Shifts")
        fixed_cols = ["Name", "Day of Week", "Start of Shift", "End of Shift"]
        fixed_df = load_or_create_csv(fixed_file, fixed_cols)

        edited_fixed = st.data_editor(
            fixed_df,
            num_rows="dynamic",
            use_container_width=True,
            key="fixed_editor"
        )

        st.subheader("Flexible Shifts")
        flex_cols = ["Name", "Length of Shift (hrs)", "Number of Shifts", "Start Day", "End Day", "Start Time", "End Time"]
        flex_df = load_or_create_csv(flexible_file, flex_cols)

        edited_flex = st.data_editor(
            flex_df,
            num_rows="dynamic",
            use_container_width=True,
            key="flex_editor"
        )

        if st.button("Save Clients"):
            edited_fixed.to_csv(fixed_file, index=False)
            edited_flex.to_csv(flexible_file, index=False)
            st.success("Clients saved!")

# ==========================
# SCHEDULES TAB
# ==========================
with tabs[1]:
    sub_tabs = st.tabs(["Caregivers", "Clients", "Exceptions"])

    with sub_tabs[0]:
        st.header("Caregiver Schedules")
        st.info("This will display caregiver schedules once solver logic is added.")

    with sub_tabs[1]:
        st.header("Client Schedules")
        st.info("This will display client schedules once solver logic is added.")

    with sub_tabs[2]:
        st.header("Exceptions")
        st.info("Unassigned shifts and hard-constraint approvals will be shown here.")

# ==========================
# SETTINGS TAB
# ==========================
with tabs[2]:
    st.header("Settings")
    st.write("Configure file paths for CSV storage.")

    for key, default_path in st.session_state.file_paths.items():
        new_path = st.text_input(f"Path for {key}", value=default_path, key=f"path_{key}")
        st.session_state.file_paths[key] = new_path

    if st.button("Save Settings"):
        st.success("File paths updated! They will be used going forward.")

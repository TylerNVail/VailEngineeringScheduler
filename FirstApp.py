import pandas as pd
import streamlit as st

# --- Helpers ---
def load_csv(path, default_columns):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=default_columns)
    # Fill defaults
    for col in default_columns:
        if col not in df.columns:
            df[col] = ""
    return df.fillna("")

def save_csv(df, path):
    df.to_csv(path, index=False)

# --- Caregiver List ---
st.subheader("Caregiver List")

caregiver_columns = ["Name", "Base Location", "Notes"]
caregivers_df = load_csv("caregivers.csv", caregiver_columns)

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
    save_csv(edited_caregivers, "caregivers.csv")
    st.success("Caregivers saved!")

# --- Client List ---
st.subheader("Client List")

client_columns = ["Name", "Base Location", "Importance", "Scheduling Mode", "Preferred Caregivers", "Notes"]
clients_df = load_csv("clients.csv", client_columns)

# Ensure Importance is always numeric
clients_df["Importance"] = pd.to_numeric(clients_df["Importance"], errors="coerce").fillna(0).astype(int)

edited_clients = st.data_editor(
    clients_df,
    column_config={
        "Name": st.column_config.TextColumn("Name"),
        "Base Location": st.column_config.TextColumn("Base Location"),
        "Importance": st.column_config.NumberColumn("Importance (0–10)", min_value=0, max_value=10, step=1),
        "Scheduling Mode": st.column_config.TextColumn("Scheduling Mode"),
        "Preferred Caregivers": st.column_config.TextColumn("Preferred Caregivers (comma separated)"),
        "Notes": st.column_config.TextColumn("Notes"),
    },
    num_rows="dynamic",
    key="client_list_editor"
)

if st.button("Save Clients"):
    save_csv(edited_clients, "clients.csv")
    st.success("Clients saved!")

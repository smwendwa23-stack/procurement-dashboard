import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest

# Load data
data = pd.read_csv("procurement_data.csv")

# Title
st.title("Kenya AI Governance Mini Dashboard 🤖")
st.subheader("Procurement Monitoring with AI Insights")

# Show raw data

# ---------------------------

# AI anomaly detection
model = IsolationForest(contamination=0.2, random_state=42)

data["Anomaly"] = model.fit_predict(data[["Amount"]])

data["Status"] = data["Anomaly"].apply(lambda x: "Suspicious" if x == -1 else "Normal")

def highlight_suspicious(row):
    return ['background-color: red' if row["Status"] == "Suspicious" else '' for _ in row]

st.dataframe(data.style.apply(highlight_suspicious, axis=1))

threshold = 100000

data["High_Value"] = data["Amount"] > threshold
# ---------------------------

# Use Amount column for anomaly detection
model = IsolationForest(contamination=0.2, random_state=42)

# Fit model
data["Anomaly"] = model.fit_predict(data[["Amount"]])

# Mark anomalies (-1 = anomaly, 1 = normal)
data["Status"] = data["Anomaly"].apply(lambda x: "Suspicious" if x == -1 else "Normal")

# Show flagged transactions
st.write("### AI-Detected Suspicious Transactions")
st.dataframe(data)

# ---------------------------
# Charts
# ---------------------------

st.write("### Total Spending by Ministry")
ministry_spending = data.groupby("Ministry")["Amount"].sum()
st.bar_chart(ministry_spending)

st.write("### Spending by Supplier")
supplier_spending = data.groupby("Supplier")["Amount"].sum()
st.bar_chart(supplier_spending)

# ---------------------------
# Insights Section
# ---------------------------

st.write("### AI Insights")

suspicious_count = len(data[data["Status"] == "Suspicious"])

st.write(f"Number of suspicious transactions detected: {suspicious_count}")

high_value_count = len(data[data["High_Value"] == True])

st.write(f"High-value transactions (>{threshold}): {high_value_count}")

if high_value_count > 0:
    st.warning("⚠️ High-value transactions detected. Review recommended.")

if suspicious_count > 0:
    st.error("⚠️ ALERT: Suspicious procurement activities detected!")
else:
    st.success("✅ All transactions appear normal.")

if suspicious_count > 0:
    st.warning("⚠️ Some transactions may require further review.")
else:
    st.success("✅ No anomalies detected.")
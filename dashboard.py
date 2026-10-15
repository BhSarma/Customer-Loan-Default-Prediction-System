import streamlit as st
import pandas as pd

st.set_page_config(page_title="Loan Default Dashboard", layout="wide")

st.title("Customer Loan Default Analytics Dashboard")
st.write("Explore customer loan data and default risk insights.")
# Load loan dataset
df = pd.read_csv("artifacts/loan_guard_ai_dataset.csv")

# Dashboard metrics
total_customers = len(df)
total_defaults = (df["Loan_Default"] == "Yes").sum()
default_rate = (total_defaults / total_customers) * 100

st.subheader("Loan Portfolio Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Loan Defaults", f"{total_defaults:,}")
col3.metric("Default Rate", f"{default_rate:.2f}%")
st.subheader("Loan Default Distribution")

default_counts = df["Loan_Default"].value_counts()

st.pyplot(default_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    figsize=(2, 2),
    ylabel=""
).figure)
st.subheader("Default Rate by Employment Type")

default_rate = (
    df.groupby("Employment_Type")["Loan_Default"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="Default Rate (%)")
)

st.bar_chart(
    default_rate.set_index("Employment_Type")["Default Rate (%)"]
)
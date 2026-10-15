import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
df = pd.read_csv("artifacts/loan_guard_ai_dataset.csv")

model = pickle.load(open("artifacts/loan_rf_model.pkl", "rb"))
meta = pickle.load(open("artifacts/feature_meta.pkl", "rb"))

st.title("Loan Default Prediction App")
st.write("Enter customer details below:")


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
values = {}
for col, options in meta["cat_options"].items():
    values[col] = st.selectbox(col.replace("_", " "), options)
for col, default in meta["num_defaults"].items():
    values[col] = st.number_input(col.replace("_", " "), value=default)

if st.button("Predict Loan Default"):
    row = {c: 0 for c in meta["columns"]}
    for col, val in values.items():
        if col in meta["cat_options"]:
            key = f"{col}_{val}"
            if key in row:
                row[key] = 1
        else:
            row[col] = val
    df = pd.DataFrame([row])[meta["columns"]]

    pred = str(model.predict(df)[0])
    classes = [str(c) for c in model.classes_]
    pos = [i for i, c in enumerate(classes) if c in ("Yes", "1", "True")][0]
    prob = model.predict_proba(df)[0][pos]

    if pred in ("Yes", "1", "True"):
        st.error(
            f"🔴 High Risk: Likely to default "
            f"(Probability {prob:.0%})"
        )
    else:
        st.success(
            f"🟢 Low Risk: Unlikely to default "
            f"(Probability {prob:.0%})"
        )
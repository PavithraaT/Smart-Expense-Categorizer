import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Expense Categorizer", page_icon="💸", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Title
st.title("💸 Smart Expense Categorizer")
st.write("AI-powered expense classification with analytics 📊")
st.markdown("---")

# Input
st.markdown("### 🧾 Enter transactions")
st.write("Format: description - amount (comma separated)")

user_input = st.text_area("Example: swiggy order - 200, uber ride - 150")

results = []

if st.button("🚀 Analyze"):

    if user_input.strip() != "":
        transactions = re.split(r'[,\n;]', user_input)

        for txn in transactions:
            txn = txn.strip()

            if "-" in txn:
                desc, amt = txn.split("-")
                desc = desc.strip()
                amt = float(amt.strip())

                cleaned = clean_text(desc)
                vec = vectorizer.transform([cleaned])
                category = model.predict(vec)[0]

                results.append({
                    "Description": desc,
                    "Amount": amt,
                    "Category": category
                })

        if results:
            df = pd.DataFrame(results)

            # 📊 Show table
            st.markdown("### 📋 Transactions")
            st.dataframe(df)

            # 💰 Total spend
            total = df["Amount"].sum()
            st.success(f"💰 Total Spend: ₹{total:.2f}")

            # 📊 Category summary
            st.markdown("### 📊 Category-wise Spending")
            cat_summary = df.groupby("Category")["Amount"].sum()
            st.bar_chart(cat_summary)

            # 🥧 Pie chart
            st.markdown("### 🥧 Expense Distribution")
            fig, ax = plt.subplots()
            cat_summary.plot.pie(autopct='%1.1f%%', ax=ax, figsize=(5,5))
            st.pyplot(fig)

            # 🏆 Top spending category
            top_category = cat_summary.idxmax()
            top_amount = cat_summary.max()
            st.info(f"🏆 Top Spending Category: **{top_category}** with ₹{top_amount:.2f}")

    else:
        st.warning("⚠️ Please enter transactions")
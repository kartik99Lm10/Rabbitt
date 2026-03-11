import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq
import os

st.set_page_config(page_title="Talking Rabbitt", page_icon="🐰", layout="wide")

st.title("🐰 Talking Rabbitt")
st.caption("Ask questions about your business data")

api_key = st.secrets["GROQ_API_KEY"]

if not api_key:
    st.error("GROQ_API_KEY not found. Please set it in environment variables.")
    st.stop()

client = Groq(api_key=api_key)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df = df.fillna("")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.divider()

    question = st.chat_input("Ask a question about your data")

    if question:

        st.chat_message("user").write(question)

        with st.chat_message("assistant"):

            with st.spinner("Analyzing your data..."):

                preview = df.head(5).to_string()

                prompt = f"""
You are a professional data analyst.

Dataset columns:
{list(df.columns)}

Sample rows:
{preview}

User question:
{question}

Rules:
- Use pandas dataframe called df
- Only use columns that exist
- Do NOT assume meanings of columns
- If dataset lacks required information return:

result = "The dataset does not contain the required information."

- Always store final output in variable called result
- If visualization is useful create a matplotlib chart
- Never print explanations
- Return ONLY Python code
"""

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )

                code = response.choices[0].message.content
                code = code.replace("```python", "").replace("```", "").strip()

                try:

                    plt.clf()

                    local_vars = {
                        "df": df.copy(),
                        "plt": plt,
                        "pd": pd
                    }

                    exec(code, {}, local_vars)

                    # Show result
                    if "result" in local_vars:

                        if isinstance(local_vars["result"], str):
                            st.info(local_vars["result"])
                        else:
                            st.write(local_vars["result"])

                    # Show chart only if one exists
                    fig = plt.gcf()

                    if fig.axes:
                        st.pyplot(fig)

                except Exception:

                    st.warning(
                        "Sorry, I couldn't analyze that question with the current dataset."
                    )

else:

    st.info("Upload a CSV file to begin analysis.")

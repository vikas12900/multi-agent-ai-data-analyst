import streamlit as st
import pandas as pd

from utils.profiler import profile_dataset
from agents.dataset_summarizer import summarize_dataset
from agents.cleaning_agent import clean_dataset
from agents.ml_readiness_agent import assess_ml_readiness

st.set_page_config(
    page_title="AI Data Analyst",
    layout="wide"
)

# Session State
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "cleaning_report" not in st.session_state:
    st.session_state.cleaning_report = None

if "cleaning_plan" not in st.session_state:
    st.session_state.cleaning_plan = None

if "ml_report" not in st.session_state:
    st.session_state.ml_report = None

st.title("AI Data Analyst")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Generate profile
    profile = profile_dataset(df)

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset Overview
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", profile["rows"])

    with col2:
        st.metric("Columns", profile["columns"])

    with col3:
        st.metric("Duplicates", profile["duplicate_rows"])

    # Columns
    st.subheader("Columns")
    st.write(profile["column_names"])

    # Types
    st.subheader("Column Types")
    st.json(profile["dtypes"])

    # Missing Values
    st.subheader("Missing Values")
    st.json(profile["missing_values"])

    # Executive Summary
    with st.spinner("Generating summary..."):
        summary = summarize_dataset(profile)

    st.subheader("Executive Summary")
    st.write(summary)

    st.divider()

    # CLEAN DATASET
    st.subheader("Dataset Cleaning")

    if st.button("Clean Dataset"):

        with st.spinner("Cleaning dataset..."):

            (
                st.session_state.cleaned_df,
                st.session_state.cleaning_report,
                st.session_state.cleaning_plan
            ) = clean_dataset(df)

    # Show Cleaning Results
    if st.session_state.cleaned_df is not None:

        st.subheader("Cleaning Plan")
        st.json(st.session_state.cleaning_plan)

        st.subheader("Cleaning Report")

        for item in st.session_state.cleaning_report:
            st.write(f"✓ {item}")

        st.subheader("Cleaned Dataset Preview")

        st.dataframe(
            st.session_state.cleaned_df.head()
        )

        csv = st.session_state.cleaned_df.to_csv(
            index=False
        )

        st.download_button(
            label="Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    st.divider()

    # ML READINESS
    st.subheader("ML Readiness Assessment")

    if st.button("Assess ML Readiness"):

        data_to_assess = (
            st.session_state.cleaned_df
            if st.session_state.cleaned_df is not None
            else df
        )

        assessment_profile = profile_dataset(
            data_to_assess
        )

        with st.spinner(
            "Assessing dataset..."
        ):

            st.session_state.ml_report = (
                assess_ml_readiness(
                    assessment_profile
                )
            )

    if st.session_state.ml_report:

        st.subheader(
            "ML Readiness Report"
        )

        st.write(
            st.session_state.ml_report
        )
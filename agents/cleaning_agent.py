import os
import json
import pandas as pd

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


def create_dataset_profile(df: pd.DataFrame):

    profile = {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": {
            col: int(count)
            for col, count in df.isnull().sum().items()
            if count > 0
        },
        "duplicates": int(df.duplicated().sum())
    }

    return profile


def generate_cleaning_plan(profile):

    prompt = ChatPromptTemplate.from_template(
        """
You are a senior data cleaning expert.

Dataset Profile:

{profile}

Available actions:

1. remove_duplicates
2. fill_missing_median
3. fill_missing_mode
4. standardize_column_names
5. drop_column

Rules:

- Numeric columns -> median
- Categorical columns -> mode
- Remove duplicates if duplicates exist
- Drop a column only if it is mostly missing

Return ONLY valid JSON.

Example:

{{
    "steps":[
        {{
            "action":"remove_duplicates"
        }},
        {{
            "action":"fill_missing_median",
            "column":"Age"
        }}
    ]
}}
"""
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "profile": json.dumps(
                profile,
                indent=2
            )
        }
    )

    return response.content


def apply_cleaning_plan(df, plan):

    report = []

    for step in plan["steps"]:

        action = step["action"]

        if action == "remove_duplicates":

            before = len(df)

            df = df.drop_duplicates()

            removed = before - len(df)

            report.append(
                f"Removed {removed} duplicate rows"
            )

        elif action == "fill_missing_median":

            col = step["column"]

            missing = df[col].isnull().sum()

            median = df[col].median()

            df[col] = df[col].fillna(median)

            report.append(
                f"Filled {missing} missing values in {col} using median"
            )

        elif action == "fill_missing_mode":

            col = step["column"]

            missing = df[col].isnull().sum()

            mode = df[col].mode()[0]

            df[col] = df[col].fillna(mode)

            report.append(
                f"Filled {missing} missing values in {col} using mode"
            )

        elif action == "standardize_column_names":

            old_columns = list(df.columns)

            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
            )

            report.append(
                "Standardized column names"
            )

        elif action == "drop_column":

            col = step["column"]

            if col in df.columns:

                df = df.drop(columns=[col])

                report.append(
                    f"Dropped column {col}"
                )

    return df, report


def clean_dataset(df):

    profile = create_dataset_profile(df)

    cleaning_plan = generate_cleaning_plan(profile)

    # Remove markdown if model returns it
    cleaning_plan = cleaning_plan.replace(
        "```json",
        ""
    ).replace(
        "```",
        ""
    ).strip()

    plan = json.loads(cleaning_plan)

    cleaned_df, report = apply_cleaning_plan(
        df,
        plan
    )

    return cleaned_df, report, plan
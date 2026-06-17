import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
    """
You are a senior data analyst.

Analyze the following dataset profile and generate a concise executive summary.

Dataset Profile:

Rows: {rows}

Columns: {columns}

Column Names:
{column_names}

Column Types:
{dtypes}

Missing Values:
{missing_values}

Duplicate Rows:
{duplicate_rows}

Your response should include:

1. Dataset Overview
2. Data Quality Issues
3. Potential Risks
4. Recommended Next Steps

Keep the response concise and business-oriented.
"""
)

chain = prompt | llm


def summarize_dataset(profile: dict):

    response = chain.invoke(
        {
            "rows": profile["rows"],
            "columns": profile["columns"],
            "column_names": profile["column_names"],
            "dtypes": profile["dtypes"],
            "missing_values": profile["missing_values"],
            "duplicate_rows": profile["duplicate_rows"]
        }
    )

    return response.content
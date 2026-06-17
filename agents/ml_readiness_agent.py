import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(model = "llama-3.3-70b-versatile",
               api_key = os.getenv("GROQ_API_KEY"))

prompt = ChatPromptTemplate.from_template(
    """
You are a senior Machine Learning Engineer.

Analyze the following dataset profile and assess its readiness for machine learning.

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

Provide the following:

1. ML Readiness Score (out of 10)
2. Potential Target Variables
3. Suitable Machine Learning Tasks
4. Required Preprocessing Steps
5. Potential Data Issues
6. Recommended Next Steps

Keep the report concise and practical.
"""
)

chain = prompt | llm

def assess_ml_readiness(profile):

    response = chain.invoke({
                    "rows": profile["rows"],
            "columns": profile["columns"],
            "column_names": profile["column_names"],
            "dtypes": profile["dtypes"],
            "missing_values": profile["missing_values"],
            "duplicate_rows": profile["duplicate_rows"]
    })

    return response.content



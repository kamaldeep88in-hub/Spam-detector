import os
import streamlit as st
from typing import List, Literal
from pydantic import BaseModel, Field, ValidationError

# Import LangChain + Google Gemini wrapper
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser
except ImportError:
    st.error(
        "LangChain or Google Generative AI modules not found. "
        "Make sure you installed langchain>=0.3.7 and langchain-google-genai>=2.0.0."
    )
    raise

# Define Pydantic model for classification result
class SpamClassification(BaseModel):
    label: Literal["Spam", "Not Spam", "Uncertain"]
    reasons: List[str]
    risk_score: int = Field(..., ge=0, le=100)
    red_flags: List[str]
    suggested_action: str

@st.cache_resource
def get_chain():
    parser = PydanticOutputParser(pydantic_object=SpamClassification)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"You are ScamGuard, an expert spam detector. Return ONLY a JSON object following this schema:\n{format_instructions}"
        ),
        ("human", "Classify this message:\n\"{message}\"")
    ])

    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found! Please add it to Streamlit Secrets.")
        st.stop()

    llm = ChatGoogleGenerativeAI(
        model_name="gemini-2.5-chat",
        api_key=api_key,
        temperature=0.7
    )

    return prompt | llm | parser, format_instructions

def classify_message(message: str) -> SpamClassification:
    chain, format_instructions = get_chain()
    return chain.invoke({
        "format_instructions": format_instructions,
        "message": message
    })

# Streamlit UI
st.set_page_config(page_title="Spam Detector", page_icon="🚨")
st.title("🚨 Spam Detector (Gemini + LangChain)")

st.markdown(
    "Paste an email or message and detect if it's **Spam / Not Spam / Uncertain** "
    "using **Google Gemini** and **LangChain**."
)

user_input = st.text_area("✉️ Message to classify:", height=150)

if st.button("🔍 Classify"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Analyzing message..."):
            try:
                result = classify_message(user_input)
                st.success("Classification complete!")

                st.subheader("📊 Classification Result")
                st.write("**Label:**", result.label)
                st.write("**Risk Score:**", result.risk_score)
                st.write("**Reasons:**", result.reasons)
                st.write("**Red Flags:**", result.red_flags)
                st.write("**Suggested Action:**", result.suggested_action)

                with st.expander("🧾 Raw JSON Result"):
                    st.json(result.model_dump())

            except ValidationError as ve:
                st.error(f"Validation failed: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

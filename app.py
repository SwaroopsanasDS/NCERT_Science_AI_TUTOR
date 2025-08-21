# app.py
import streamlit as st
from rag_pipeline import rag_qa
import os

st.set_page_config(
    page_title="Curiosity AI – Science Tutor",
    page_icon="🧪",
    layout="wide"
)

with st.sidebar:
    st.markdown("## 🧠 How to Use")
    st.write("Type a science question from Class 8 Science and get an answer with references!")
    st.write("👉 Example: *What is photosynthesis?*")
    st.markdown("---")
    st.write("Built with ❤️ using **RAG + Hugging Face**")

st.markdown("<h1 style='text-align:center;color:#4CAF50'>🌟 Curiosity AI – Your Science Buddy 🌟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Ask Class 8 Science questions and get answers with references!</p>", unsafe_allow_html=True)

query = st.text_input("🔍 Ask a question:", "")

if st.button("✨ Ask AI"):
    if not query.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking... 🤔"):
            try:
                answer, sources = rag_qa(query)
                st.markdown("### ✅ Answer")
                st.success(answer.strip())

                if sources:
                    with st.expander("📚 Show Sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** {src}")
                else:
                    st.info("No sources found for this answer.")
            except Exception as e:
                st.error(f"❌ Error running RAG QA: {str(e)}")
                st.info("Please make sure the FAISS index is properly built and available.")

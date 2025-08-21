# app.py
import streamlit as st
from rag_pipeline import rag_qa  # Import our RAG function

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Curiosity AI – Science Tutor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🧠 How to Use")
    st.write("Type a science question from your Class 8 syllabus and get answers with references!")
    st.write("👉 Try: *What is photosynthesis?*")
    st.markdown("---")
    st.write("Built with ❤️ using **RAG + Hugging Face**")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>🌟 Curiosity AI – Your Science Buddy 🌟</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Ask your Class 8 Science questions and get answers with references!</p>",
    unsafe_allow_html=True
)

# -----------------------------
# User Input
# -----------------------------
query = st.text_input("🔍 Ask a question:", "")

# -----------------------------
# Run RAG QA
# -----------------------------
if st.button("✨ Ask AI"):
    if query.strip() == "":
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking... 🤔"):
            try:
                # Call your rag_qa function
                answer, sources = rag_qa(query)

                # Display Answer
                st.markdown("### ✅ Answer")
                st.success(answer.strip())

                # Display Sources
                if sources:
                    with st.expander("📚 Show Sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** {src}")
                else:
                    st.info("No sources found for this answer.")

            except Exception as e:
                st.error(f"❌ Error running RAG QA: {e}")

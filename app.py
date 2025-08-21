import streamlit as st
from rag_pipeline import rag_qa

st.set_page_config(
    page_title="Curiosity AI – Science Tutor",
    page_icon="🧪",
    layout="wide"
)

st.sidebar.markdown("## 🧠 How to Use")
st.sidebar.write("Type a science question from Class 8 syllabus and get answers with references!")
st.sidebar.write("👉 Try: *What is photosynthesis?*")
st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ using **RAG + Hugging Face**")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🌟 Curiosity AI – Science Buddy 🌟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask Class 8 Science questions and get answers with references!</p>", unsafe_allow_html=True)

query = st.text_input("🔍 Ask a question:", "")

if st.button("✨ Ask AI"):
    if query.strip() == "":
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
                st.error(f"❌ Error running RAG QA: {e}")

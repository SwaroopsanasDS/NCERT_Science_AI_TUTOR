# app.py
import streamlit as st
import os
import time

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
    st.write("Built with ❤️ using **RAG + Transformers**")
    
    # Display status
    if os.path.exists("data/faiss_index/index.faiss"):
        st.success("✅ FAISS index found")
    else:
        st.error("❌ FAISS index missing")
    
    if os.getenv("HF_TOKEN"):
        st.success("✅ HF Token available")
    else:
        st.warning("⚠️ HF Token not set")

st.markdown("<h1 style='text-align:center;color:#4CAF50'>🌟 Curiosity AI – Your Science Buddy 🌟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Ask Class 8 Science questions and get answers with references!</p>", unsafe_allow_html=True)

# Try to import with error handling
try:
    from rag_pipeline import rag_qa
    rag_available = True
except Exception as e:
    st.error(f"❌ RAG system not available: {e}")
    rag_available = False

query = st.text_input("🔍 Ask a question:", "")

if st.button("✨ Ask AI"):
    if not query.strip():
        st.warning("Please enter a question!")
    elif not rag_available:
        st.error("RAG system is not available. Please check the logs for details.")
    else:
        with st.spinner("Thinking... 🤔"):
            try:
                start_time = time.time()
                answer, sources = rag_qa(query)
                end_time = time.time()
                
                st.markdown("### ✅ Answer")
                st.success(answer.strip())
                st.caption(f"Generated in {end_time - start_time:.2f} seconds")

                if sources:
                    with st.expander("📚 Show Sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** {src}")
                else:
                    st.info("No sources found for this answer.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("This might be due to model loading issues or missing dependencies.")

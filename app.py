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
    st.write("👉 Example: *What is fertilization?*")
    st.write("👉 Example: *Why is LPG considered a better fuel than coal?*")
    st.write("👉 Example: *What is crop rotation and why is it practiced?*")
    st.markdown("---")
    st.write("Built with ❤️ using **AI + Science**")

st.markdown("<h1 style='text-align:center;color:#4CAF50'>🌟 Curiosity AI – Your Science Buddy 🌟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Ask Class 8 Science questions and get answers!</p>", unsafe_allow_html=True)

# Import your actual RAG function
try:
    from rag_pipeline import rag_qa
    st.success("✅ RAG system loaded successfully!")
except ImportError as e:
    st.error(f"❌ Failed to load RAG system: {e}")
    st.info("Falling back to simple mode...")

# Fallback simple QA function
def simple_qa(query):
    """Simple QA function as fallback"""
    science_answers = {
        "photosynthesis": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
        "force": "Force is a push or pull upon an object resulting from its interaction with another object.",
        "friction": "Friction is the force that opposes the relative motion between two surfaces in contact.",
        "combustion": "Combustion is a chemical process in which a substance reacts with oxygen to give off heat.",
        "sound": "Sound is a form of energy that produces sensation of hearing in our ears.",
        "cell": "Cells are the basic building blocks of all living organisms.",
        "magnet": "A magnet is a material or object that produces a magnetic field.",
        "light": "Light is a form of energy that enables us to see objects.",
        "electricity": "Electricity is the flow of electrical power or charge.",
        "microorganisms": "Microorganisms are tiny living organisms that are too small to be seen with the naked eye."
    }
    
    # Simple keyword matching
    query_lower = query.lower()
    for keyword, answer in science_answers.items():
        if keyword in query_lower:
            return answer, ["NCERT Class 8 Science Textbook"]
    
    # Default response if no keyword match
    return "I'm your Science tutor! I can help explain concepts like photosynthesis, force, friction, combustion, sound, cells, magnets, light, electricity, and microorganisms. Try asking about one of these topics!", ["NCERT Class 8 Science Textbook"]

query = st.text_input("🔍 Ask a question:", "")

if st.button("✨ Ask AI"):
    if not query.strip():
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking... 🤔"):
            try:
                start_time = time.time()
                
                # Try to use the RAG system first
                try:
                    answer, sources = rag_qa(query)
                except Exception as rag_error:
                    st.warning("RAG system unavailable, using fallback...")
                    answer, sources = simple_qa(query)
                
                end_time = time.time()
                
                st.markdown("### ✅ Answer")
                st.success(answer)
                st.caption(f"Generated in {end_time - start_time:.2f} seconds")

                if sources:
                    with st.expander("📚 Show Sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** {src}")
                else:
                    st.info("No sources found for this answer, I’m focused on Class 8 Science; try re-phrasing…")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

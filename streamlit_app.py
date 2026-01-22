import streamlit as st
import requests

# ===============================
# CONFIG
# ===============================
API_URL = "https://olist-backend-api-420147884504.asia-southeast1.run.app/chat"

st.set_page_config(
    page_title="Olist AI Assistant",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Olist AI Assistant")
st.caption("Multi-Agent AI (SQL • RAG • LLM)")

# ===============================
# SESSION STATE
# ===============================
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# INPUT
# ===============================
query = st.text_input(
    "Tanyakan tentang produk Olist:",
    placeholder="contoh: rekomendasi produk kategori furniture dengan review bagus"
)

# ===============================
# SUBMIT
# ===============================
if st.button("Kirim"):
    if not query:
        st.warning("Silakan masukkan pertanyaan.")
    else:
        with st.spinner("Memproses..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=60
                )

                if response.status_code != 200:
                    st.error(f"HTTP {response.status_code}: {response.text}")
                else:
                    data = response.json()
                    st.session_state.history.append({
                        "query": query,
                        "answer": data.get("answer", ""),
                        "source": data.get("source", "")
                    })

            except requests.exceptions.RequestException as e:
                st.error(f"Koneksi ke backend gagal: {e}")

# ===============================
# CHAT HISTORY
# ===============================
for chat in reversed(st.session_state.history):
    st.markdown("### ❓ Pertanyaan")
    st.write(chat["query"])

    st.markdown("### 💡 Jawaban")
    st.write(chat["answer"])

    st.markdown(
        f"**🧠 Source Agent:** `{chat['source']}`"
    )

    st.divider()

"""
app.py — Streamlit Frontend (Kullanıcı Arayüzü)
=================================================
Sohbet tabanlı hukuki asistan arayüzü.
Soruları Cloud Run'daki FastAPI backend'e HTTP ile gönderir.

"""

import streamlit as st
import requests
import os
import time

# ==============================================================================
# 1. BACKEND BAĞLANTISI
# ==============================================================================
# Cloud Run URL'si (.env veya Streamlit Cloud Secrets'tan okunur)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ==============================================================================
# 2. SAYFA AYARLARI (Page Config)
# ==============================================================================
st.set_page_config(
    page_title="Multi-Law Legal Agent",
    page_icon="⚖️",
    layout="centered"
)
st.title("⚖️ Komşuluk & Apartman Hukuku Asistanı")
st.caption("Uzmanlık Alanı: Site Yönetimi, Komşuluk İlişkileri ve Apartman Sorunları (KMK Odaklı)")

# ==============================================================================
# 2.5. YAN MENÜ (Sidebar) - Proje Bilgisi & Teknik Detaylar
# ==============================================================================
with st.sidebar:
    st.header("📌 Proje Hakkında")
    st.markdown(
        """
        <small>Bu asistan, <b>Kat Mülkiyeti Kanunu (KMK)</b> başta olmak üzere, 
        apartman ve site yönetimiyle ilgili hukuki soruları yanıtlamak için geliştirilmiştir.
        
        <b>Kapsam:</b>
        <ul>
        <li>Site Yönetimi</li>
        <li>Aidat & Gider Paylaşımı</li>
        <li>Komşuluk Hakları</li>
        </ul>
        </small>
        """, 
        unsafe_allow_html=True
    )
    
    st.divider()
    
    st.header("Teknik Detaylar")
    st.caption("Bu ayarlar sabittir, sadece bilgi amaçlı gösterilmektedir.")
    
    st.markdown("### 🧠 Model Yapısı")
    st.markdown("**LLM:** `gpt-4o`")
    st.markdown("**Embedding:** `text-embedding-3-small`")
    st.markdown("**Vektör DB:** `ChromaDB (Cloud)`")

# ==============================================================================
# 3. BACKEND SAĞLIK KONTROLÜ
# ==============================================================================
# Sayfa yüklendiğinde backend'in erişilebilir olup olmadığını kontrol et
if "backend_ready" not in st.session_state:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=10)
        data = resp.json()
        st.session_state.backend_ready = data.get("rag_ready", False)
        if st.session_state.backend_ready:
            st.success("Sistem Hazır! Sorunuzu sorabilirsiniz.")
            time.sleep(1)
            st.rerun()
        else:
            st.warning("Backend erişilebilir ama RAG sistemi henüz hazır değil.")
    except Exception as e:
        st.error(f"Backend bağlantısı kurulamadı: {e}")
        st.session_state.backend_ready = False
        st.stop()

# ==============================================================================
# 4. SOHBET GEÇMİŞİ (Chat History)
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhabalar. Apartman yönetimi, kiracı hakları veya komşuluk ilişkileri hakkında sorularınızı yanıtlayabilirim."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================================================================
# 5. KULLANICI ETKİLEŞİMİ (User Input)
# ==============================================================================
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # 1. Kullanıcı mesajını ekrana ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Backend API'ye sor
    with st.chat_message("assistant"):
        with st.spinner("Kanun maddeleri taranıyor..."):
            try:
                # --- API ÇAĞRISI ---
                resp = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": prompt},
                    timeout=60
                )
                resp.raise_for_status()
                data = resp.json()
                
                cevap = data["answer"]
                kaynaklar = data.get("sources", [])
                
                # Cevabı Göster
                st.markdown(cevap)
                
                # Kaynakları Göster
                if kaynaklar:
                    with st.expander("📚 Başvurulan Kanun Maddeleri ve Kaynaklar"):
                        for i, src in enumerate(kaynaklar):
                            st.markdown(f"**Kaynak {i+1}: {src['doc_name']}**")
                            clean = src["content"] if len(src["content"]) < 600 else src["content"][:600] + "..."
                            st.markdown(f"> {clean}")
                            st.divider()
                
                # Cevabı hafızaya kaydet
                st.session_state.messages.append({"role": "assistant", "content": cevap})
            
            except requests.exceptions.ConnectionError:
                st.error("Backend sunucusuna bağlanılamadı. Lütfen API servisinin çalıştığından emin olun.")
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")

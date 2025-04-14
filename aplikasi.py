import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# Download NLTK resources (jika belum ada)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Program Makan Bergizi",
    page_icon="🍎",
    layout="wide"
)

# Custom CSS untuk meningkatkan tampilan
st.markdown("""
<style>
    /* Header styling */
    .title-text {
        font-size: 36px !important;
        font-weight: bold;
        color: #389cff;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Subheader styling */
    .subtitle-text {
        font-size: 24px !important;
        color: #389cff;
        margin-bottom: 15px;
    }
    
    /* Box highlight */
    .highlight-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #F8F9FA;
        border-left: 5px solid #4CAF50;
        margin-bottom: 20px;
    }
    
    /* Algoritma badge */
    .algo-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        margin-right: 10px;
    }
    
    /* Model colors */
    .lstm-color { background-color: #3498DB; color: white; }
    .bilstm-color { background-color: #9B59B6; color: white; }
    .gru-color { background-color: #2ECC71; color: white; }
</style>
""", unsafe_allow_html=True)

# Daftar kata positif dan negatif
POSITIVE_WORDS = [
    "dukung", "kuat", "cerah", "meningkat", "sehat", "cerdas", 
    "penting", "damping", "senang", "terima kasih", "bisa", 
    "lahap", "baik", "mantap", "membantu"
]

NEGATIVE_WORDS = [
    "gila", "najis", "mending", "jajan", "tentang", "tidak", 
    "ketimbang", "bukan", "menghina", "belum", "tega", "malah", 
    "sakit", "mubazir", "tolol", "korupsi", "bajingan", "kasian", 
    "persetan", "tai", "sialan", "kasihan", "anjing", "goblok", 
    "ironis", "konyol", "mampus", "bangsat","buruk"
]

# Fungsi untuk memeriksa kata-kata kunci dalam teks
def check_sentiment_keywords(text):
    text_lower = text.lower()
    
    # Cek kata-kata positif
    positive_matches = []
    for word in POSITIVE_WORDS:
        if word in text_lower:
            positive_matches.append(word)
    
    # Cek kata-kata negatif
    negative_matches = []
    for word in NEGATIVE_WORDS:
        if word in text_lower:
            negative_matches.append(word)
    
    # Tentukan sentimen berdasarkan jumlah kata yang cocok
    positive_count = len(positive_matches)
    negative_count = len(negative_matches)
    
    # Kembalikan hasil
    return {
        'positive_count': positive_count,
        'negative_count': negative_count,
        'positive_matches': positive_matches,
        'negative_matches': negative_matches,
        'keyword_sentiment': 'Positif' if positive_count > negative_count else 'Negatif' if negative_count > positive_count else 'Netral',
        'keyword_score': calculate_keyword_score(positive_count, negative_count)
    }

# Fungsi untuk menghitung skor berdasarkan jumlah kata positif dan negatif
def calculate_keyword_score(positive_count, negative_count):
    total = positive_count + negative_count
    if total == 0:
        return 0.5  # Netral jika tidak ada kata kunci yang cocok
    
    # Konversi ke skor 0-1
    return positive_count / total

# Fungsi preprocessing teks
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Hapus URL
    text = re.sub(r'http\S+', '', text)
    
    # Hapus username Twitter
    text = re.sub(r'@\w+', '', text)
    
    # Hapus hashtag
    text = re.sub(r'#\w+', '', text)
    
    # Hapus angka dan tanda baca
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Hapus stopwords
    stop_words = set(stopwords.words('indonesian'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Gabungkan kembali
    processed_text = ' '.join(tokens)
    return processed_text, tokens

# Function to load model (cached agar tidak reload setiap interaksi)
@st.cache_resource
def load_sentiment_model(model_type="BI-LSTM"):
    try:
        if model_type == "BI-LSTM":
            model = load_model('D:\\My Data ALL\\Downloads\\tes\\model_BI_LSTM.h5')
        elif model_type == "GRU":
            model = load_model('D:\\My Data ALL\\Downloads\\tes\\model_GRU.h5')
        elif model_type == "LSTM":
            model = load_model('D:\\My Data ALL\\Downloads\\tes\\model_LSTM.h5')
        else:
            st.error(f"Model type {model_type} tidak dikenali.")
            return None
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function untuk memuat tokenizer
@st.cache_resource
def get_tokenizer():
    # Cek apakah file tokenizer sudah ada
    if not os.path.exists('tokenizer.pickle'):
        st.error("File tokenizer 'tokenizer.pickle' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama.")
        return None
    
    # Muat tokenizer dari file
    try:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

# Fungsi untuk membuat wordcloud
def create_wordcloud(text_tokens):
    if not text_tokens:
        return None
    
    # Gabung tokens menjadi teks untuk wordcloud
    text = ' '.join(text_tokens)
    
    # Buat wordcloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap='viridis', max_words=100).generate(text)
    
    # Plot wordcloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

# Fungsi untuk membuat gauge chart sentimen
def create_sentiment_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentimen Score", 'font': {'size': 24, 'color': '#389cff'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': 'rgba(255, 0, 0, 0.7)'},
                {'range': [20, 40], 'color': 'rgba(255, 165, 0, 0.7)'},
                {'range': [40, 60], 'color': 'rgba(255, 255, 0, 0.7)'},
                {'range': [60, 80], 'color': 'rgba(144, 238, 144, 0.7)'},
                {'range': [80, 100], 'color': 'rgba(0, 128, 0, 0.7)'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50}
        }
    ))
    
    # Perbaiki layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

# Fungsi untuk membuat bar chart perbandingan emosi
def create_emotion_bars(score):
    emotions = ['Positif', 'Negatif']
    values = [score, 1-score]
    
    fig = px.bar(
        x=emotions, 
        y=values,
        color=emotions,
        color_discrete_map={
            'Positif': 'green',
            'Negatif': 'red'
        },
        labels={'x': 'Sentimen', 'y': 'Skor'},
        title='Perbandingan Skor Sentimen'
    )
    
    fig.update_layout(
        xaxis_title='Sentimen',
        yaxis_title='Skor (0-1)',
        yaxis=dict(range=[0, 1]),
        title_font=dict(size=20, color='#389cff'),
        title_x=0.5,
    )
    
    return fig

# Fungsi untuk membuat pie chart distribusi sentimen
def create_sentiment_pie(score):
    labels = ['Positif', 'Negatif']
    values = [score, 1-score]
    
    colors = ['rgba(46, 204, 113, 0.8)', 'rgba(231, 76, 60, 0.8)']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(size=14),
        hoverinfo='label+percent'
    )])
    
    fig.update_layout(
        title='Distribusi Sentimen',
        title_font=dict(size=20, color='#389cff'),
        title_x=0.5,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

# Fungsi untuk halaman analisis
def show_analysis_page():
    # Judul aplikasi dengan styling
    st.markdown('<p class="title-text">Analisis Sentimen Program Makan Bergizi Gratis</p>', 
                unsafe_allow_html=True)
    
    # Ambil algoritma yang dipilih dari sidebar
    model_type = st.session_state.get('model_type', 'BI-LSTM')
    
    # Tampilkan algoritma yang dipilih dengan styling
    algo_color = "bilstm-color" if model_type == "BI-LSTM" else "gru-color" if model_type == "GRU" else "lstm-color"
    st.markdown(f'<div style="color: black;" class="highlight-box"><span class="algo-badge {algo_color}">{model_type}</span> Model yang dipilih untuk analisis sentimen</div>', 
                unsafe_allow_html=True)
    
    # Area input teks dengan highlight
    st.markdown('<p class="subtitle-text">Input Teks</p>', unsafe_allow_html=True)
    input_text = st.text_area("Masukkan teks untuk dianalisis:", 
                             "", 
                             height=150)
    
    # Tampilkan contoh input
    with st.expander("📋 Lihat Contoh Input"):
        st.write("""
        1. **Contoh Positif**: "Program makan bergizi gratis ini sangat membantu anak-anak dari keluarga kurang mampu mendapatkan gizi yang cukup setiap hari."
        
        2. **Contoh Negatif**: "Program makan bergizi di sekolah kami tidak berjalan dengan baik, makanannya sering terlambat dan kualitasnya buruk."
        """)
    
    # Layout tombol analisis
    col_button, _ = st.columns([1, 3])
    with col_button:
        analyze_button = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)
    
    if analyze_button:
        # Cek apakah input teks tidak kosong
        if not input_text.strip():
            st.warning("⚠️ Silakan masukkan teks untuk dianalisis.")
            return
        
        # Cek file-file yang diperlukan
        if model_type == "BI-LSTM":
            model_path = 'D:\\My Data ALL\\Downloads\\tes\\model_BI_LSTM.h5'
        elif model_type == "GRU":
            model_path = 'D:\\My Data ALL\\Downloads\\tes\\model_GRU.h5'
        else:  # LSTM
            model_path = 'D:\\My Data ALL\\Downloads\\tes\\model_LSTM.h5'
            
        tokenizer_path = 'tokenizer.pickle'
        
        files_missing = []
        if not os.path.exists(model_path):
            files_missing.append(f"Model '{model_path}'")
        if not os.path.exists(tokenizer_path):
            files_missing.append(f"Tokenizer '{tokenizer_path}'")
        
        if files_missing:
            st.warning(f"⚠️ File berikut tidak ditemukan: {', '.join(files_missing)}. Menggunakan analisis berdasarkan kata kunci saja.")
            model_available = False
        else:
            model_available = True
        
        try:
            # Tambahkan progress bar untuk UX yang lebih baik
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analisis berdasarkan kata kunci
            status_text.text("Menganalisis kata kunci...")
            progress_bar.progress(20)
            keyword_results = check_sentiment_keywords(input_text)
            
            # Load model dan tokenizer jika tersedia
            model_prediction = None
            if model_available:
                status_text.text(f"Memuat model {model_type}...")
                progress_bar.progress(40)
                model = load_sentiment_model(model_type)
                tokenizer = get_tokenizer()
                
                if model is not None and tokenizer is not None:
                    # Preprocess teks
                    status_text.text("Preprocessing teks...")
                    progress_bar.progress(60)
                    processed_text, tokens = preprocess_text(input_text)
                    
                    # Tampilkan hasil preprocessing
                    with st.expander("🔍 Lihat Hasil Preprocessing"):
                        st.code(processed_text)
                    
                    # Tokenize dan padding teks
                    status_text.text("Tokenizing teks...")
                    progress_bar.progress(70)
                    text_sequence = tokenizer.texts_to_sequences([processed_text])
                    
                    # Cek apakah sequence berhasil dibuat
                    if not text_sequence[0]:
                        st.warning(f"⚠️ Tidak ada kata yang dikenali dalam teks untuk model {model_type}. Menggunakan analisis kata kunci saja.")
                        # Gunakan sequence kosong untuk menghindari error
                        text_sequence = [[0]]
                    
                    # Padding sequence
                    padded_sequence = pad_sequences(text_sequence, maxlen=100, padding='post', truncating='post')
                    
                    # Prediksi menggunakan model
                    status_text.text("Memprediksi sentimen...")
                    progress_bar.progress(85)
                    prediction = model.predict(padded_sequence)
                    model_prediction = prediction[0][0]
            
            # Update progres
            progress_bar.progress(95)
            status_text.text("Menyiapkan hasil...")
            
            # Menentukan hasil analisis gabungan
            if model_prediction is not None:
                # Gabungkan hasil model dan kata kunci dengan bobot
                # Berikan bobot yang lebih besar untuk hasil analisis kata kunci
                combined_score = 0.3 * model_prediction + 0.7 * keyword_results['keyword_score']
                final_sentiment = "Positif" if combined_score >= 0.5 else "Negatif"
                final_score = combined_score
            else:
                # Gunakan hasil analisis kata kunci saja
                final_sentiment = keyword_results['keyword_sentiment']
                final_score = keyword_results['keyword_score']
            
            # Selesai
            progress_bar.progress(100)
            status_text.empty()
            
            # Tampilkan hasil prediksi
            st.markdown("---")
            st.markdown('<p class="subtitle-text">Hasil Analisis</p>', unsafe_allow_html=True)
            
            # Buat kolom untuk menampilkan hasil
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Tampilkan skor sentimen
                if final_sentiment == "Positif":
                    sentiment_color = "green"
                    sentiment_emoji = "😀"
                else:
                    sentiment_color = "red"
                    sentiment_emoji = "😔"
                
                st.markdown(f"<h1 style='text-align: center; color: {sentiment_color};'>{sentiment_emoji} {final_sentiment}</h1>", 
                            unsafe_allow_html=True)
                
                # Tambahkan kartu informasi dengan styling
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 10px; background-color: #f8f9fa; border-left: 5px solid {sentiment_color}; margin-bottom: 20px;color:black;">
                    <h3 style="color: {sentiment_color};">Informasi Analisis</h3>
                    <p><strong>Model:</strong> {model_type}</p>
                    <p><strong>Skor Sentimen:</strong> {final_score:.2%}</p>
                    <p><strong>Interpretasi:</strong> Teks ini mengekspresikan sentimen {final_sentiment.lower()} terhadap program makan bergizi gratis.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # # Tampilkan kata kunci yang terdeteksi
                # if keyword_results['positive_matches'] or keyword_results['negative_matches']:
                #     st.markdown("##### Kata Kunci Terdeteksi:")
                    
                #     if keyword_results['positive_matches']:
                #         st.markdown(f"""
                #         <div style="padding: 10px; border-radius: 5px; background-color: rgba(46, 204, 113, 0.2); margin-bottom: 10px;">
                #             <strong>Positif:</strong> {', '.join(keyword_results['positive_matches'])}
                #         </div>
                #         """, unsafe_allow_html=True)
                    
                #     if keyword_results['negative_matches']:
                #         st.markdown(f"""
                #         <div style="padding: 10px; border-radius: 5px; background-color: rgba(231, 76, 60, 0.2); margin-bottom: 10px;">
                #             <strong>Negatif:</strong> {', '.join(keyword_results['negative_matches'])}
                #         </div>
                #         """, unsafe_allow_html=True)
            
            with col2:
                # Tampilkan gauge chart untuk skor sentimen
                gauge_chart = create_sentiment_gauge(final_score)
                st.plotly_chart(gauge_chart, use_container_width=True)
            
            # Tambahkan visualisasi tambahan
            st.markdown("---")
            st.markdown('<p class="subtitle-text">Visualisasi Hasil</p>', unsafe_allow_html=True)
            
            # Tampilkan beberapa visualisasi dengan tabs
            tabs = st.tabs(["📊 Perbandingan", "🥧 Distribusi", "☁️ Word Cloud"])
            
            with tabs[0]:
                # Bar chart perbandingan skor
                emotion_bars = create_emotion_bars(final_score)
                st.plotly_chart(emotion_bars, use_container_width=True)
            
            with tabs[1]:
                # Pie chart distribusi sentimen
                sentiment_pie = create_sentiment_pie(final_score)
                st.plotly_chart(sentiment_pie, use_container_width=True)
            
            with tabs[2]:
                # WordCloud dari teks yang diproses
                if model_available and 'tokens' in locals() and tokens:
                    wordcloud_fig = create_wordcloud(tokens)
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
                    else:
                        st.info("Tidak cukup kata untuk membuat word cloud.")
                else:
                    # Fallback jika model tidak tersedia
                    # Buat wordcloud dari input teks langsung
                    words = input_text.lower().split()
                    wordcloud_fig = create_wordcloud(words)
                    if wordcloud_fig:
                        st.pyplot(wordcloud_fig)
                    else:
                        st.info("Tidak cukup kata untuk membuat word cloud.")
        
        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan: {e}")
            st.info("Jika masalah berlanjut, coba reboot aplikasi atau periksa console log untuk detail error.")

# Fungsi untuk halaman bantuan penggunaan
def show_help_page():
    st.markdown('<p class="title-text">Bantuan Penggunaan Aplikasi</p>', unsafe_allow_html=True)
    
    # Gunakan tabs untuk mengorganisasi konten bantuan (dikurangi menjadi 3 tab saja)
    help_tabs = st.tabs(["📘 Panduan Dasar", "🔧 Penggunaan Model", "❓ FAQ"])
    
    with help_tabs[0]:
        st.markdown("""
        ## Panduan Dasar Aplikasi
        
        Aplikasi Analisis Sentimen Program Makan Bergizi Gratis membantu Anda mengevaluasi sentimen teks yang berkaitan dengan program makan bergizi di sekolah.
        
        ### Memilih Menu
        Gunakan menu di sidebar (panel kiri) untuk navigasi:
        - **Analisa Sentimen**: Halaman utama untuk menganalisis teks
        - **Bantuan Penggunaan**: Panduan cara menggunakan aplikasi (halaman ini)
        - **Tentang Aplikasi**: Informasi tentang aplikasi dan model AI yang digunakan
        
        ### Melakukan Analisis Sentimen
        1. Masukkan teks yang ingin dianalisis pada kotak teks
        2. Klik tombol "🔍 Analisis Sentimen" berwarna biru
        3. Tunggu hingga proses analisis selesai
        
        ### Memahami Hasil Analisis
        Hasil analisis akan menampilkan:
        - Klasifikasi sentimen (Positif/Negatif)
        - Skor sentimen dalam bentuk persentase dan gauge chart
        - Visualisasi perbandingan sentimen
        - Word cloud dari kata-kata yang diproses
        """)
    
    with help_tabs[1]:
        st.markdown("""
        ## Penggunaan Model
        
        ### Algoritma yang Tersedia
        Aplikasi ini menyediakan tiga algoritma berbeda untuk analisis sentimen:
        
        1. **LSTM (Long Short-Term Memory)**
           - Model dasar untuk memahami ketergantungan jangka panjang dalam teks
           - Mampu mempertahankan informasi penting dalam urutan kata
           
        2. **BI-LSTM (Bidirectional LSTM)**
           - Long Short-Term Memory yang diimplementasikan secara bidirectional
           - Kemampuan untuk memahami konteks kata dari dua arah (maju dan mundur)
           - Efektif untuk menangkap ketergantungan jangka panjang dalam teks
        
        3. **GRU (Gated Recurrent Unit)**
           - Varian dari Recurrent Neural Network yang lebih sederhana dari LSTM
           - Memiliki mekanisme gating yang memungkinkan model mempertahankan informasi penting
           - Lebih ringan dan cepat dalam pelatihan dibandingkan LSTM
        
        ### Cara Memilih Model
        - Di sidebar, pilih model yang ingin digunakan dalam dropdown "Pilih Algoritma"
        - Hasil analisis akan diperbarui sesuai dengan model yang dipilih
        - Anda dapat membandingkan hasil dari model yang berbeda untuk analisis yang lebih komprehensif
        """)
    
    with help_tabs[2]:
        st.markdown("""
        ## Pertanyaan yang Sering Diajukan (FAQ)
        
        ### Umum
        
        **Q: Apa perbedaan utama antara ketiga model yang tersedia?**
        
        A: LSTM adalah model dasar untuk memahami urutan kata, BI-LSTM dapat memahami konteks dari dua arah (lebih akurat), dan GRU lebih ringan dan cepat namun tetap efektif.
        
        **Q: Mengapa hasil analisis berbeda antar model?**
        
        A: Setiap model memiliki arsitektur dan cara belajar yang berbeda. BI-LSTM umumnya lebih akurat karena mampu melihat konteks dari dua arah.
        
        **Q: Bagaimana aplikasi menentukan skor sentimen final?**
        
        A: Aplikasi menggabungkan hasil analisis dari model deep learning (30%) dan analisis kata kunci (70%).
        
        ### Teknis
        
        **Q: Apa yang terjadi jika file model atau tokenizer tidak ditemukan?**
        
        A: Aplikasi akan memanfaatkan analisis kata kunci saja, meskipun hasilnya mungkin kurang akurat.
        
        **Q: Dapatkah aplikasi menganalisis teks dalam bahasa selain Indonesia?**
        
        A: Aplikasi ini dioptimalkan untuk bahasa Indonesia, khususnya dalam konteks program makan bergizi.
        
        **Q: Apa itu preprocessing teks dan mengapa penting?**
        
        A: Preprocessing membersihkan dan menyederhanakan teks (lowercase, hapus tanda baca, stemming, dll.) agar model dapat memahami esensi teks dengan lebih baik.
        """)
    
    st.info("Jika mengalami masalah atau memiliki pertanyaan lebih lanjut, silakan hubungi administrator sistem.")

def show_about_page():
    st.markdown('<p class="title-text">Tentang Aplikasi</p>', unsafe_allow_html=True)
    
    # Tambahkan banner aplikasi
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="text-align: center; color: #389cff;">Aplikasi Analisis Sentimen Program Makan Bergizi Gratis</h2>
        <p style="text-align: center; color: #7F8C8D; font-style: italic;">Menganalisis opini masyarakat terhadap program makan bergizi di sekolah dengan pendekatan AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Gunakan hanya tab Overview
    about_tabs = st.tabs(["📱 Overview"])
    
    with about_tabs[0]:
        st.markdown("""
        ## Tentang Aplikasi
        
        Aplikasi ini dikembangkan untuk menganalisis sentimen masyarakat terhadap program makan bergizi gratis yang diimplementasikan di berbagai sekolah. Menggunakan kombinasi teknik deep learning dan analisis kata kunci, aplikasi ini mampu mengklasifikasikan teks ke dalam sentimen positif atau negatif.
        
        ### Tujuan Aplikasi
        
        Aplikasi ini bertujuan untuk:
        1. Membantu pemangku kepentingan memahami persepsi publik terhadap program makan bergizi gratis
        2. Mengidentifikasi aspek positif dan negatif dari program berdasarkan opini masyarakat
        3. Menyediakan alat analisis yang dapat digunakan untuk evaluasi program
        4. Memberikan wawasan untuk perbaikan dan pengembangan program di masa depan
        """)
        
        # Tampilkan fitur utama dalam cards
        st.markdown("### Fitur Utama")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background-color: #E8F4F9; padding: 15px; border-radius: 10px; height: 200px;">
                <h4 style="color: #3498DB;">🔍 Analisis Multi-Model</h4>
                <p style="color: black;">Menggunakan tiga model deep learning berbeda (LSTM, BI-LSTM, GRU) untuk analisis sentimen yang komprehensif.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div style="background-color: #E8F9F0; padding: 15px; border-radius: 10px; height: 200px;">
                <h4 style="color: #2ECC71;">📊 Visualisasi Interaktif</h4>
                <p style="color: black;">Menyajikan hasil analisis dalam bentuk gauge chart, bar chart, pie chart, dan word cloud untuk pemahaman yang lebih baik.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div style="background-color: #F9E8F0; padding: 15px; border-radius: 10px; height: 200px;">
                <h4 style="color: #9B59B6;">🔤 Analisis Kata Kunci</h4>
                <p style="color: black;">Mengidentifikasi kata-kata positif dan negatif yang memengaruhi sentimen teks secara keseluruhan.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; margin-top: 30px;">
        <p style="color: #7F8C8D; font-size: 14px;">© 2025 - Aplikasi Analisis Sentimen Program Makan Bergizi Gratis</p>
        <p style="color: #95A5A6; font-size: 12px;">Dikembangkan dengan ❤️ untuk melihat sentimen publik terhadap program makan bergizi gratis</p>
    </div>
    """, unsafe_allow_html=True)

# Fungsi utama aplikasi
def main():
    # Sidebar dengan menu navigasi dan styling
    # Sidebar with improved styling and button-based navigation
# Sidebar with improved styling and button-based navigation
    with st.sidebar:
        # Logo and header with better styling
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #389cff; display: flex; align-items: center; justify-content: center;">
                <span style="color: #ff5050; font-size: 1.5em; margin-right: 8px;">🍎</span> 
                <span>Analisis Sentimen</span>
            </h2>
            <p style="color: #7F8C8D; margin-top: -10px;">Program Makan Bergizi</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<hr style="margin: 15px 0px;">', unsafe_allow_html=True)
        
        # Menu navigation with buttons instead of radio buttons
        st.markdown("### 📋 Menu Aplikasi")
        
        # Initialize session state if not exists
        if 'current_page' not in st.session_state:
            st.session_state['current_page'] = "Analisa Sentimen"
        
        # Simple button navigation without nested columns
        if st.button("🔍 Analisa Sentimen", key="btn_analisa", 
                    use_container_width=True,
                    type="primary" if st.session_state['current_page'] == "Analisa Sentimen" else "secondary"):
            st.session_state['current_page'] = "Analisa Sentimen"
            st.rerun()
        
        if st.button("❓ Bantuan Penggunaan", key="btn_bantuan",
                    use_container_width=True,
                    type="primary" if st.session_state['current_page'] == "Bantuan Penggunaan" else "secondary"):
            st.session_state['current_page'] = "Bantuan Penggunaan"
            st.rerun()
        
        if st.button("ℹ️ Tentang Aplikasi", key="btn_tentang",
                    use_container_width=True,
                    type="primary" if st.session_state['current_page'] == "Tentang Aplikasi" else "secondary"):
            st.session_state['current_page'] = "Tentang Aplikasi"
            st.rerun()
        
        st.markdown('<hr style="margin: 15px 0px;">', unsafe_allow_html=True)
        
        # Pengaturan Algoritma (Dropdown)
        st.markdown("### ⚙️ Pengaturan Algoritma")

        # Dropdown untuk memilih algoritma tanpa tanda centang
        model_type = st.selectbox(
            "Pilih algoritma AI yang akan digunakan:",
            ["LSTM", "BI-LSTM", "GRU"],
            index=["LSTM", "BI-LSTM", "GRU"].index(st.session_state.get('model_type', 'BI-LSTM')),
            help="Pilih algoritma AI yang akan digunakan untuk analisis sentimen"
        )

        # Informasi algoritma dengan card styling
        algo_info = {
            "LSTM": "Model dasar untuk memahami ketergantungan dalam teks",
            "BI-LSTM": "Model bidirectional dengan akurasi yang lebih tinggi",
            "GRU": "Model yang lebih ringan dan cepat"
        }

        algo_colors = {
            "LSTM": "#3498DB",
            "BI-LSTM": "#9B59B6",
            "GRU": "#2ECC71"
        }

        # Tampilkan informasi algoritma
        st.markdown(f"""
        <div style="background-color: {algo_colors.get(model_type, '#389cff')}33; 
            padding: 10px; border-radius: 5px; margin-top: 10px; border-left: 3px solid {algo_colors.get(model_type, '#389cff')}">
            <p style="margin: 0; color: white;"><strong>{model_type}:</strong> {algo_info.get(model_type, "")}</p>
        </div>
        """, unsafe_allow_html=True)

        # Simpan ke session state
        st.session_state['model_type'] = model_type

        
        st.markdown('<hr style="margin: 15px 0px;">', unsafe_allow_html=True)
        
        # Footer with better styling
        st.markdown("""
        <div style="text-align: center; padding: 10px; margin-top: 20px;">
            <p style="color: #95A5A6; font-size: 12px; margin: 0;">© 2025 Analisis Sentimen</p>
            <p style="color: #95A5A6; font-size: 12px; margin: 0;">Program Makan Bergizi Gratis</p>
        </div>
        """, unsafe_allow_html=True)

    # Handle page selection based on session state (this part should be outside the sidebar)
    selected_page = st.session_state.get('current_page', "Analisa Sentimen")

    # Render the appropriate page
    if selected_page == "Analisa Sentimen":
        show_analysis_page()
    elif selected_page == "Bantuan Penggunaan":
        show_help_page()
    elif selected_page == "Tentang Aplikasi":
        show_about_page()

if __name__ == "__main__":
    main()
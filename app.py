import streamlit as st
import os
import requests # Backend iletişimleri ve görsel indirme için
import json
import base64
from PIL import Image
from io import BytesIO

# AI API'leri için doğrudan import'lar
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv # .env dosyasını yüklemek için (yerel için)

# --- API Anahtarlarını Yapılandırma ---
# Streamlit Cloud'da 'Secrets' kullanarak veya yerel ortam değişkenleri (.env ile)
# Önemli: Bu anahtarları doğrudan GitHub'a YÜKLEMEYİN!
try:
    # Yerel çalıştırmalar için .env dosyasını yükle
    load_dotenv() 

    # Ortam değişkenlerinden oku (hem yerel .env hem de sistem ortam değişkenleri)
    GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    
    # Eğer Streamlit Cloud'da çalışıyorsak ve ortam değişkenleri ayarlı değilse st.secrets'ı dene
    if not GEMINI_API_KEY and "GOOGLE_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GOOGLE_API_KEY"]
    if not OPENAI_API_KEY and "OPENAI_API_KEY" in st.secrets:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


    if not GEMINI_API_KEY:
        st.error("Gemini API anahtarı bulunamadı. Lütfen 'GOOGLE_API_KEY' ortam değişkenini veya Streamlit Secrets'ı ayarlayın.")
        st.stop() # Anahtar yoksa uygulamayı durdur

    if not OPENAI_API_KEY:
        st.error("OpenAI API anahtarı bulunamadı. Lütfen 'OPENAI_API_KEY' ortam değişkenini veya Streamlit Secrets'ı ayarlayın.")
        st.stop() # Anahtar yoksa uygulamayı durdur
        
    # AI kütüphanelerini yapılandır
    genai.configure(api_key=GEMINI_API_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
except Exception as e:
    st.error(f"API anahtarları yapılandırılamadı: {e}. Lütfen anahtarlarınızı kontrol edin.")
    st.stop() # Hata durumunda uygulamayı durdur

# --- Şirket Bilgileri (AI'ya sürekli anımsatılacak) ---
COMPANY_INFO_CONTEXT = """
Şirket Adı: SOYKOK PREMIUM HOME LTD (Premium Home)
Ana Faaliyet Alanları: Metal evler, prefabrik yapılar, Tiny House üretimi ve inşaatı, nanoteknoloji zemin ısıtma sistemleri.
Misyon: Yenilikçi, sürdürülebilir, modern ve uygun fiyatlı yaşam/çalışma alanları sunmak. Anahtar teslim çözümler.
Web Sitesi: https://www.premiumpluscy.eu
Katalog Sitesi: https://linktr.ee/premiumplushome
Instagram Hesabı: https://www.instagram.com/premiumplushome
Facebook Sayfası: https://www.facebook.com/PremiumPlusHomeCyprus (Varsayımsal URL, lütfen güncelleyin!)
LinkedIn Sayfası: https://www.linkedin.com/company/premium-home-ltd (Varsayımsal URL, lütfen güncelleyin!)
Hedef Kitle: Metal ev ve prefabrik yapılarla ilgilenen, Tiny House kültürünü benimsemek isteyen, Avrupa bölgelerinde bulunan kişiler ve profesyoneller.
"""

# --- Backend API URL'si ---
# Render.com'daki canlı backend URL'nizi buraya yapıştırın.
# Yerel test için: "http://localhost:5000"
BACKEND_API_URL = "https://premium-home-social-api.onrender.com" # BURAYA KENDİ RENDER URL'NİZİ YAPIŞTIRIN!

# --- AI Metin Üretme Fonksiyonu (Gemini Flash) ---
@st.cache_data # Bu dekoratör fonksiyon çıktısını önbelleğe alır
def generate_text_gemini_flash(prompt_text, target_language="Türkçe"):
    model = genai.GenerativeModel('gemini-2.0-flash')
    # Şirket bilgilerini prompt'a ekle
    full_prompt = (
        f"{COMPANY_INFO_CONTEXT}\n\n"
        f"Yukarıdaki şirket bilgilerini ve faaliyet alanlarını göz önünde bulundurarak, şu içerik isteğini tamamla: "
        f"'{prompt_text}'. Lütfen çıktıyı {target_language} dilinde oluştur."
    )
    try:
        response = model.generate_content(full_prompt)
        if response and response.text:
            return response.text
        else:
            return "Yanıt alınamadı veya boş. Lütfen prompt'u kontrol edin."
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg or "TooManyRequests" in error_msg:
            return f"Hata: Kota aşımı! Lütfen daha sonra tekrar deneyin veya kota durumunuzu kontrol edin. Detay: {e}"
        elif "api key not valid" in error_msg.lower() or "authentication error" in error_msg.lower():
            return "Hata: Gemini API anahtarı geçersiz veya yetkilendirme hatası. Lütfen anahtarınızı kontrol edin."
        else:
            return f"Hata: API Hatası: {e}"

# --- AI Görsel Yorumlama Fonksiyonu (Gemini Vision) ---
@st.cache_data
def interpret_image_gemini_vision(pil_image_object, prompt_text="Bu resimde ne görüyorsun?"):
    model = genai.GenerativeModel('gemini-1.5-flash')
    # Görsel yorumlama prompt'ına şirket bağlamını eklemeye gerek yok genellikle,
    # doğrudan görseli yorumlamalıdır. Ancak istenirse eklenebilir.
    try:
        contents = [prompt_text, pil_image_object]
        response = model.generate_content(contents)
        if response and response.text:
            return response.text
        else:
            return "Görsel yorumu alınamadı veya boş."
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg or "TooManyRequests" in error_msg:
            return f"Hata: Görsel yorumlama kota aşımı! Lütfen daha sonra tekrar deneyin. Detay: {e}"
        elif "api key not valid" in error_msg.lower() or "authentication error" in error_msg.lower():
            return "Hata: Gemini API anahtarı geçersiz veya yetkilendirme hatası. Lütfen anahtarınızı kontrol edin."
        else:
            return f"Hata: Görsel yorumlama hatası: {e}"

# --- AI Görsel Oluşturma Fonksiyonu (DALL-E 3) ---
def generate_image_dalle(image_prompt_text):
    global openai_client # OpenAI istemcisi global değişken olarak tanımlı
    if not openai_client:
        return "Hata: OpenAI istemcisi başlatılamadı."
    # Görsel prompt'ına şirket bağlamını ekle (eğer prompt_text boşsa ve AI metininden geliyorsa zaten olacaktır)
    # Direkt kullanıcıdan gelen prompt'lar için de şirket bağlamı ekleyebiliriz:
    full_image_prompt = (
        f"{COMPANY_INFO_CONTEXT}\n\n"
        f"Yukarıdaki şirket bilgilerini ve faaliyet alanlarını göz önünde bulundurarak, şu görseli oluştur: "
        f"'{image_prompt_text}'."
    )
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=full_image_prompt, # Güncellenmiş prompt
            n=1,
            size="1024x1024"
        )
        if response and response.data and response.data[0].url:
            image_url = response.data[0].url
            img_data = requests.get(image_url).content
            return base64.b64encode(img_data).decode('utf-8')
        else:
            return "Görsel oluşturulamadı veya URL bulunamadı."
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg or "TooManyRequests" in error_msg or "billing_not_active" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
            return f"Hata: Görsel oluşturma kota/ödeme hatası! Lütfen OpenAI hesabınızdaki DALL-E faturalandırmasını kontrol edin. Detay: {e}"
        elif "authentication error" in error_msg.lower():
            return "Hata: OpenAI API anahtarı geçersiz. Lütfen anahtarınızı kontrol edin."
        else:
            return f"Hata: Görsel oluşturma hatası: {e}"

# --- AI ile Metin Formatlama Fonksiyonu ---
@st.cache_data
def format_text_for_social_media(text, platform, target_language="Türkçe"):
    model = genai.GenerativeModel('gemini-2.0-flash')
    # Şirket bilgilerini ve sosyal medya hesaplarını prompt'a ekle
    # Doğrudan sosyal medya için hazır hale getirme isteğini vurgula
    company_social_info_context = f"""
    Şirket Adı: Premium Home
    Web Sitesi: https://www.premiumpluscy.eu
    Katalog Sitesi: https://linktr.ee/premiumplushome
    Instagram: https://www.instagram.com/premiumplushome
    Facebook: https://www.facebook.com/PremiumPlusHomeCyprus
    LinkedIn: https://www.linkedin.com/company/premium-home-ltd
    """

    format_prompt_base = f"{COMPANY_INFO_CONTEXT}\n{company_social_info_context}\n\n"

    if platform == "Instagram":
        format_prompt = (
            f"{format_prompt_base}"
            f"Yukarıdaki şirket bilgilerini ve sosyal medya hesaplarını göz önünde bulundurarak, aşağıdaki metni görsel odaklı ve direkt paylaşılmaya hazır bir Instagram gönderisine dönüştür. "
            f"Kısa paragraflar, emoji ve trend hashtagler kullan. Harekete geçirici (CTA) ifadeler ekle. "
            f"Metni orijinal anlamını koruyarak, Instagram'ın karakter sınırlamalarına uygun ama bilgilendirici olacak şekilde {target_language} dilinde düzenle. "
            f"Web sitesi ve katalog linklerini uygun yerlerde belirt. Metin: \n\n{text}"
        )
    elif platform == "Facebook":
        format_prompt = (
            f"{format_prompt_base}"
            f"Yukarıdaki şirket bilgilerini ve sosyal medya hesaplarını göz önünde bulundurarak, aşağıdaki metni Facebook topluluğu için samimi, bilgilendirici ve paylaşılmaya hazır bir gönderiye dönüştür. "
            f"Paylaşımı teşvik eden sorular, topluluk odaklı ifadeler ve uygun hashtagler kullan. "
            f"Metni video veya görsel içeriğe eşlik edebilecek, sohbeti başlatacak şekilde {target_language} dilinde yaz. Web sitesi ve katalog linklerini uygun yerlerde belirt. Metin: \n\n{text}"
        )
    elif platform == "LinkedIn":
        format_prompt = (
            f"{format_prompt_base}"
            f"Yukarıdaki şirket bilgilerini ve sosyal medya hesaplarını göz önünde bulundurarak, aşağıdaki metni LinkedIn profesyonel ağı için bilgilendirici, otoriter ve direkt paylaşılmaya hazır bir gönderiye dönüştür. "
            f"Sektörel içgörüler, profesyonel terimler ve konuyla ilgili hashtagler kullan. "
            f"Değer katan bilgiler sun ve tartışmayı teşvik et. Web sitesi ve katalog linklerini uygun yerlerde belirt. Çıktıyı {target_language} dilinde ver. Metin: \n\n{text}"
        )
    elif platform == "Genel Blog Yazısı":
        format_prompt = (
            f"{format_prompt_base}"
            f"Yukarıdaki şirket bilgilerini göz önünde bulundurarak, aşağıdaki metni bir blog yazısı formatına dönüştür. Blogun ana başlığını, alt başlıklarını ve paragraflarını açıkça belirt. "
            f"Okunabilirliği artırmak için giriş, gelişme (alt başlıklar kullanarak) ve sonuç bölümleri oluştur. "
            f"Anahtar kelimelerle zenginleştirilmiş, bilgilendirici ve SEO dostu bir yapı kur. "
            f"Web sitesi ve katalog linklerini uygun yerlerde belirt. Çıktıyı {target_language} dilinde ver. Metin: \n\n{text}"
        )
    elif platform == "E-posta Bülteni":
        format_prompt = (
            f"{format_prompt_base}"
            f"Yukarıdaki şirket bilgilerini göz önünde bulundurarak, aşağıdaki metni kısa, öz ve okuyucuyu harekete geçiren bir e-posta bülteni içeriğine dönüştür. "
            f"Net bir konu başlığı (subject line) öner, kısa giriş, ana faydaları vurgulayan maddeler veya kısa paragraflar ve net bir harekete geçirici mesaj (CTA) içer. "
            f"Web sitesi ve katalog linklerini uygun yerlerde belirt. Çıktıyı {target_language} dilinde ver. Metin: \n\n{text}"
        )
    else: # Varsayılan veya bilinmeyen platformlar için
        format_prompt = (
            f"{format_prompt_base}"
            f"Yukarıdaki şirket bilgilerini göz önünde bulundurarak, aşağıdaki metni genel bir sosyal medya platformu için uygun, ilgi çekici ve etkileşim artırıcı bir gönderi formatında yeniden yaz. "
            f"Gerektiğinde emoji ve uygun hashtagler ekle. Metni orijinal anlamını koruyarak düzenle. "
            f"Çıktıyı {target_language} dilinde ver. "
            f"Metin: \n\n{text}"
        )

    try:
        response = model.generate_content(format_prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "api key not valid" in error_msg.lower() or "authentication error" in error_msg.lower():
            return "Hata: Gemini API anahtarı geçersiz veya yetkilendirme hatası. Lütfen anahtarınızı kontrol edin."
        return f"Hata: Metin formatlama hatası (AI): {e}"

# --- YouTube Video Fikri Oluşturma Fonksiyonu (Gemini Flash) ---
@st.cache_data
def generate_youtube_idea_gemini(prompt_text, target_language="Türkçe"):
    model = genai.GenerativeModel('gemini-2.0-flash')
    # Şirket bilgilerini prompt'a ekle
    full_prompt = (
        f"{COMPANY_INFO_CONTEXT}\n\n"
        f"Yukarıdaki şirket bilgilerini ve faaliyet alanlarını göz önünde bulundurarak, "
        f"'{prompt_text}' konusunda bir YouTube videosu fikri oluştur. "
        f"Başlık önerileri, anahtar noktalar (video içeriği), kısa bir senaryo taslağı (giriş, gelişme, sonuç) ve potansiyel görsel/çekim fikirleri içermeli. "
        f"Çıktıyı {target_language} dilinde ver."
    )
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "api key not valid" in error_msg.lower() or "authentication error" in error_msg.lower():
            return "Hata: Gemini API anahtarı geçersiz veya yetkilendirme hatası. Lütfen anahtarınızı kontrol edin."
        return f"Hata: YouTube video fikri oluşturma hatası (AI): {e}"

# --- Frontend Yardımcı Fonksiyonları (Backend ile İletişim Kurar) ---
def call_backend_api(endpoint, method="GET", payload=None):
    """Genel backend API çağrı fonksiyonu."""
    url = f"{BACKEND_API_URL}{endpoint}"
    try:
        if method == "POST":
            response = requests.post(url, json=payload)
        else: # GET
            response = requests.get(url)
        response.raise_for_status() # HTTP hata kodları için istisna fırlatır
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend API'ye bağlanırken hata oluştu: {e}")
        return {"error": f"API Bağlantı Hatası: {e}"}
    except json.JSONDecodeError as e:
        st.error(f"Backend'den geçersiz JSON yanıtı alındı: {e}. Yanıt: {response.text}")
        return {"error": f"JSON Çözümleme Hatası: {e}"}

# Backend'den video oluşturma isteği gönderme
def generate_video_from_backend(video_prompt_text, target_language="Türkçe"):
    endpoint = "/api/generate_video"
    payload = {"video_prompt_text": video_prompt_text, "target_language": target_language}
    response = call_backend_api(endpoint, method="POST", payload=payload)
    # Backend'den gelen hata mesajlarını kontrol et
    if "error" in response:
        return f"Hata: Video oluşturma isteği başarısız oldu. Detay: {response['error']}"
    return response.get("message", "Video oluşturma isteği gönderilemedi.") + " " + \
           response.get("status_url", "Durum URL'si yok.") + " " + \
           response.get("estimated_time", "") + " Video ID: " + str(response.get("video_id", "Yok"))

def get_social_stats_from_backend():
    endpoint = "/api/social_stats" # Backend'deki mevcut endpoint
    return call_backend_api(endpoint, method="GET")

# --- Streamlit Uygulama Arayüzü ---
st.set_page_config(layout="wide")
st.title("Premium Home AI Sosyal Medya Asistanı 🚀")

st.markdown("""
    Bu asistan, Premium Home için sosyal medya içerikleri oluşturmanıza, görselleri yorumlamanıza ve yeni fikirler üretmenize yardımcı olur.
    Metal evler, prefabrik yapılar ve Tiny House kültürü odaklı içerikler üretir.
    ---
""")

# --- Sosyal Medya Yetkilendirme Bölümü ---
st.header("Sosyal Medya Hesaplarını Yetkilendir")
st.markdown("""
    İstatistikleri çekebilmek ve diğer sosyal medya özelliklerini kullanabilmek için hesaplarınızı bağlamalısınız.
    Bu işlem sizi backend servisimize yönlendirecektir.
""")

col_auth1, col_auth2 = st.columns(2)
with col_auth1:
    if st.button("Facebook/Instagram'ı Yetkilendir", type="primary", key="auth_facebook_button"):
        st.markdown(f"[Facebook/Instagram Yetkilendirme Başlat]({BACKEND_API_URL}/auth/facebook)", unsafe_allow_html=True)
        st.info("Yukarıdaki linke tıklayın ve Facebook yetkilendirmesini tamamlayın. Ardından bu uygulamaya geri dönün.")

with col_auth2:
    if st.button("Google/YouTube'u Yetkilendir", type="primary", key="auth_google_button"):
        st.markdown(f"[Google/YouTube Yetkilendirme Başlat]({BACKEND_API_URL}/auth/google)", unsafe_allow_html=True)
        st.info("Yukarıdaki linke tıklayın ve Google yetkilendirmesini tamamlayın. Ardından bu uygulamaya geri dönün.")

st.markdown("---")

# --- Metin Oluşturucu Bölümü ---
st.header("Metin Oluştur")
prompt_text = st.text_area(
    'İçerik İsteği:',
    placeholder='Örn: Kıbrıs\'taki Tiny House projelerinin avantajlarını anlatan bir sosyal medya metni yaz.',
    height=150,
    key='prompt_input'
)

col1, col2 = st.columns(2)
with col1:
    language_options = ['Türkçe', 'English', 'Ελληνικά']
    selected_language = st.selectbox('Çıktı Dili:', language_options, key='lang_selector')
with col2:
    if st.button('Metin Oluştur', type="primary", key='generate_text_button'):
        with st.spinner(f"'{selected_language}' dilinde içerik oluşturuluyor..."):
            generated_content = generate_text_gemini_flash(prompt_text, selected_language)
        st.session_state.last_generated_text = generated_content
        st.session_state.last_selected_language = selected_language
        st.markdown("### Oluşturulan Metin:")
        st.code(generated_content, language='markdown')

# --- Sosyal Medya Metnini Formatla ve Paylaş Bölümü ---
st.header("Sosyal Medya Metnini Formatla ve Paylaş")
st.markdown("<p style='font-size:13px; color:#555;'>*Yukarıdaki 'Metin Oluştur' bölümünde üretilen son metni kullanır.</p>", unsafe_allow_html=True)

if 'last_generated_text' in st.session_state and st.session_state.last_generated_text:
    col3, col4 = st.columns(2)
    with col3:
        platform_options = ['Instagram', 'Facebook', 'LinkedIn', 'Genel Blog Yazısı', 'E-posta Bülteni']
        selected_platform = st.selectbox('Formatla:', platform_options, key='platform_selector')
    with col4:
        if st.button('Formatla ve Paylaş (AI)', type="secondary", key='format_share_button'):
            with st.spinner(f"Metin '{selected_platform}' için formatlanıyor..."):
                formatted_text = format_text_for_social_media(st.session_state.last_generated_text, selected_platform, st.session_state.last_selected_language)
            st.markdown("### Oluşturulan Metin:")
            st.code(formatted_text, language='markdown')

            encoded_formatted_text_share = requests.utils.quote(formatted_text)
            website_url = "https://www.premiumpluscy.eu"
            linkedin_share_url = f"https://www.linkedin.com/feed/?shareActive=true&text={encoded_formatted_text_share}"
            facebook_share_url = f"https://www.facebook.com/sharer/sharer.php?quote={encoded_formatted_text_share}"
            instagram_placeholder_url = "https://www.instagram.com/"

            st.markdown(f"""
            <div class="social-media-buttons-container">
                <a href="{website_url}" target='_blank' class='social-button website'>Web Sitesine Git</a>
                <a href="{linkedin_share_url}" target='_blank' class='social-button linkedin'>LinkedIn'de Paylaş</a>
                <a href="{instagram_placeholder_url}" target='_blank' class='social-button instagram'>Instagram'da Paylaş</a>
                <a href="{facebook_share_url}" target='_blank' class='social-button facebook'>Facebook'ta Paylaş</a>
                <p style="font-size:12px; color:#666; margin-top:10px;"><i>Not: Bu butonlar manuel paylaşıma yönlendirir, API entegrasyonu backend'de yapılır.</i></p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Önce 'Metin Oluştur' bölümünden bir metin oluşturun.")

# --- Görsel Yükle ve Yorumla Bölümü ---
st.header("Görsel Yükle ve Yorumla")
uploaded_file = st.file_uploader("Yorumlamak için bir görsel yükleyin", type=['png', 'jpg', 'jpeg'], key="image_uploader")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Görsel', use_column_width=True)
    
    if st.button('Görseli Yorumla', type="secondary", key='interpret_image_button'):
        with st.spinner("Görsel yorumlanıyor..."):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            interpretation = interpret_image_gemini_vision(Image.open(BytesIO(base64.b64decode(img_b64))))
        st.markdown("### Görsel Yorumu:")
        st.code(interpretation, language='markdown')

# --- Yapay Zeka ile Görsel Oluştur Bölümü ---
st.header("Yapay Zeka ile Görsel Oluştur (DALL-E 3)")
image_prompt = st.text_area(
    'Görsel Açıklaması:',
    placeholder='Örn: Kıbrıs\'ta modern bir Tiny House\'un gün batımındaki panoramik görüntüsü.',
    height=100,
    key='image_prompt_input'
)

if st.button('Görsel Oluştur', type="primary", key='generate_image_button'):
    if not image_prompt.strip():
        if 'last_generated_text' in st.session_state and st.session_state.last_generated_text:
            image_prompt = f"{st.session_state.last_generated_text} Sosyal medya gönderisi için akılda kalıcı, profesyonel ve modern bir görsel olsun."
            st.warning("Görsel açıklaması boştu, son oluşturulan metin kullanıldı. İstem kutusunu düzenleyip tekrar 'Görsel Oluştur' butonuna tıklayınız.")
        else:
            st.error("Lütfen görsel için bir açıklama girin veya metin oluşturun.")
            st.stop()
    
    with st.spinner(f"Görsel oluşturuluyor: '{image_prompt[:50]}...'"):
        generated_image_b64 = generate_image_dalle(image_prompt)

    if generated_image_b64 and not "Hata:" in generated_image_b64:
        st.markdown("### Oluşturulan Görsel:")
        st.image(base64.b64decode(generated_image_b64), caption='Oluşturulan Görsel', use_column_width=True)
        st.download_button(
            label="Görseli İndir",
            data=base64.b64decode(generated_image_b64),
            file_name="ai_generated_image.png",
            mime="image/png"
        )
    else:
        st.error(f"Görsel oluşturma başarısız oldu: {generated_image_b64}")

# --- YouTube Video Fikri Oluştur Bölümü ---
st.header("YouTube Video Fikri Oluştur")
youtube_prompt = st.text_area(
    'Video Fikri İsteği:',
    placeholder='Örn: Tiny House inşaat sürecini anlatan bir YouTube videosu fikri.',
    height=150,
    key='youtube_prompt_input'
)

col5, col6 = st.columns(2)
with col5:
    if st.button('YouTube Fikri Oluştur', type="primary", key='generate_youtube_idea_button'):
        if not youtube_prompt.strip():
            if 'last_generated_text' in st.session_state and st.session_state.last_generated_text:
                youtube_prompt = st.session_state.last_generated_text
                st.warning("YouTube fikri açıklaması boştu, son oluşturulan metin önerildi. İstem kutusunu düzenleyip tekrar 'YouTube Fikri Oluştur' butonuna tıklayınız.")
            else:
                st.error("Lütfen YouTube video fikri için bir açıklama girin veya metin oluşturun.")
                st.stop()
        
        with st.spinner(f"YouTube video fikri oluşturuluyor: '{youtube_prompt[:50]}...'"):
            youtube_idea = generate_youtube_idea_gemini(youtube_prompt, "Türkçe")
        st.session_state.last_youtube_idea = youtube_idea
        st.markdown("### Oluşturulan YouTube Video Fikri:")
        st.code(youtube_idea, language='markdown')
    
with col6:
    if st.button('YouTube Fikrini Video İçin Kullan', type="secondary", key='use_for_video_creation_button'):
        if 'last_youtube_idea' in st.session_state and st.session_state.last_youtube_idea:
            st.session_state.video_creation_prompt_input_value = st.session_state.last_youtube_idea
            st.success("YouTube Fikri video istemine kopyalandı. Şimdi aşağıdan 'Video Oluştur' butonuna tıklayabilirsiniz.")
        else:
            st.error("Önce bir YouTube Fikri oluşturmanız gerekiyor.")

# --- AI ile Kısa Video Oluşturma (Backend'e yönlendirildi) Bölümü ---
st.header("AI ile Kısa Video Oluştur")
st.markdown("<p style='font-size:13px; color:#555;'>*Yukarıdaki 'YouTube Video Fikri Oluştur' bölümünde üretilen son fikri kullanır.</p>", unsafe_allow_html=True)

video_creation_prompt_input = st.text_area(
    'Video Oluşturma İstem:',
    value=st.session_state.get('video_creation_prompt_input_value', ''),
    placeholder='Video oluşturma istemi giriniz (Örn: Bir Tiny House\'un 15 saniyelik tanıtım videosu).',
    height=150,
    key='video_creation_prompt_input'
)

if st.button('Video Oluştur (API Gerekli)', type="secondary", key='generate_short_video_button'):
    if not video_creation_prompt_input.strip():
        st.error("Lütfen video oluşturmak için bir istem girin veya YouTube fikri oluşturun.")
        st.stop()
    
    with st.spinner(f"Video oluşturma isteği: '{video_creation_prompt_input[:50]}...'"):
        generated_video_info = generate_video_from_backend(video_creation_prompt_input, "Türkçe")
    st.markdown("### Oluşturulan Video Bilgisi:")
    st.code(generated_video_info, language='markdown')

# --- Sosyal Medya İstatistikleri Bölümü ---
st.header("Sosyal Medya İstatistikleri")
st.markdown("<p style='font-size:13px; color:#555;'>*Hesaplarınızı yetkilendirdikten sonra buradan istatistikleri çekebilirsiniz.</p>", unsafe_allow_html=True)

if st.button('İstatistikleri Çek', type="primary", key='fetch_stats_button'):
    with st.spinner("İstatistikler çekiliyor..."):
        stats_data = get_social_stats_from_backend() # Backend çağrısı
    
    st.markdown("### Toplam Sosyal Medya İstatistikleri:")
    if stats_data and not stats_data.get("error"):
        # Facebook/Instagram Stats
        fb_ig_stats = stats_data.get("facebook_instagram_stats", {})
        if fb_ig_stats.get("status") == "Facebook yetkilendirmesi yapılmadı.":
            st.warning("Facebook/Instagram yetkilendirmesi yapılmadığı için istatistikler çekilemedi.")
        elif fb_ig_stats.get("error"):
            st.error(f"Facebook/Instagram istatistik çekme hatası: {fb_ig_stats['error']}")
        else:
            st.subheader("Facebook/Instagram İstatistikleri:")
            fb_page = fb_ig_stats.get("facebook_page", {})
            if fb_page:
                st.write(f"- **Sayfa Adı:** {fb_page.get('page_name', 'Bilinmiyor')}")
                st.write(f"- **Sayfa Beğenileri:** {fb_page.get('page_likes', 'Yok')}")
                st.write(f"- **Sayfa Takipçileri:** {fb_page.get('page_followers', 'Yok')}")
            
            ig_profile = fb_ig_stats.get("instagram_profile", {})
            if ig_profile:
                st.write(f"- **Instagram Kullanıcı Adı:** {ig_profile.get('username', 'Bilinmiyor')}")
                st.write(f"- **Instagram Takipçileri:** {ig_profile.get('followers_count', 'Yok')}")
                st.write(f"- **Instagram Medya Sayısı:** {ig_profile.get('media_count', 'Yok')}")
            else:
                st.warning("Instagram İşletme Hesabı bulunamadı veya bağlı değil. Lütfen Facebook Sayfanızın bir Instagram İşletme Hesabına bağlı olduğundan emin olun.")


        # YouTube Stats
        yt_stats = stats_data.get("youtube_stats", {})
        if yt_stats.get("status") == "Google yetkilendirmesi yapılmadı.":
            st.warning("Google/YouTube yetkilendirmesi yapılmadığı için istatistikler çekilemedi.")
        elif yt_stats.get("error"):
            st.error(f"YouTube istatistik çekme hatası: {yt_stats['error']}")
        else:
            st.subheader("YouTube İstatistikleri:")
            yt_channel = yt_stats.get("channel", {})
            if yt_channel:
                st.write(f"- **Kanal Adı:** {yt_channel.get('channel_name', 'Bilinmiyor')}")
                st.write(f"- **Abone Sayısı:** {yt_channel.get('subscriber_count', 'Yok')}")
                st.write(f"- **Görüntülenme Sayısı:** {yt_channel.get('view_count', 'Yok')}")
                st.write(f"- **Video Sayısı:** {yt_channel.get('video_count', 'Yok')}")
            else:
                st.warning("Yetkilendirilmiş Google hesabına bağlı bir YouTube kanalı bulunamadı.")
    else:
        st.error(f"İstatistikler çekilirken genel bir hata oluştu: {stats_data.get('error', 'Bilinmeyen hata.')}")

st.markdown("---")
st.markdown("Developed with ❤️ by Premium Home AI Assistant")

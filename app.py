import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO
import base64
from openai import OpenAI # DALL-E için
from dotenv import load_dotenv # .env dosyasını yüklemek için

# .env dosyasını yükle (yerel çalıştırmalar için)
load_dotenv()

# --- API Anahtarlarını Yapılandırma ---
# Streamlit Cloud'da 'Secrets' kullanarak veya yerel ortam değişkenleri
# Önemli: Bu anahtarları doğrudan GitHub'a YÜKLEMEYİN!
try:
    # Ortam değişkenlerinden oku (hem yerel .env hem de sistem ortam değişkenleri)
    GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    
    # Eğer Streamlit Cloud'da çalışıyorsak ve ortam değişkenleri ayarlı değilse st.secrets'ı dene
    # st.secrets'a erişmek için uygulama Streamlit Cloud'da olmalı
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
        
    genai.configure(api_key=GEMINI_API_KEY)
    # st.success("API anahtarları yapılandırıldı (Ortam Değişkenleri/Streamlit Secrets).") # Bu mesaj arayüzde görünür
    
    # OpenAI istemcisini başlat
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
except Exception as e:
    st.error(f"API anahtarları yapılandırılamadı: {e}. Lütfen anahtarlarınızı kontrol edin.")
    st.stop() # Hata durumunda uygulamayı durdur


# --- AI Metin Üretme Fonksiyonu (Gemini Flash) ---
@st.cache_data # Bu dekoratör fonksiyon çıktısını önbelleğe alır
def generate_text_gemini_flash(prompt_text, target_language="Türkçe"):
    model = genai.GenerativeModel('gemini-2.0-flash')
    localized_prompt = f"{prompt_text} Lütfen çıktıyı {target_language} dilinde oluştur."
    try:
        response = model.generate_content(localized_prompt)
        # Yanıt boşsa veya beklenen formatta değilse kontrol et
        if response and response.text:
            return response.text
        else:
            return "Yanıt alınamadı veya boş. Lütfen prompt'u kontrol edin."
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg or "TooManyRequests" in error_msg:
            return f"Hata: Kota aşımı! Lütfen daha sonra tekrar deneyin veya kota durumunuzu kontrol edin. Detay: {e}"
        elif "api key not valid" in error_msg.lower() or "authentication error" in error_msg.lower():
            return "Hata: API anahtarı geçersiz veya yetkilendirme hatası. Lütfen anahtarınızı kontrol edin."
        else:
            return f"Hata: API Hatası: {e}"

# --- AI Görsel Yorumlama Fonksiyonu (Gemini Vision) ---
@st.cache_data
def interpret_image_gemini_vision(pil_image_object, prompt_text="Bu resimde ne görüyorsun?"):
    model = genai.GenerativeModel('gemini-1.5-flash')
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
            return "Hata: API anahtarı geçersiz veya yetkilendirme hatası. Lütfen anahtarınızı kontrol edin."
        else:
            return f"Hata: Görsel yorumlama hatası: {e}"

# --- AI Görsel Oluşturma Fonksiyonu (DALL-E 3) ---
# Bu fonksiyon dış API çağrısı yaptığı için st.cache_data dikkatli kullanılmalı, her çağrıda yeni görsel istiyorsak kaldırılabilir.
def generate_image_dalle(image_prompt_text):
    if not openai_client:
        return "Hata: OpenAI istemcisi başlatılamadı."
    try:
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=image_prompt_text,
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
    format_prompt = ""
    if platform == "Instagram":
        format_prompt = (
            f"Aşağıdaki metni görsel odaklı Instagram için ilgi çekici, kısa ve öz bir gönderi metnine dönüştür. "
            f"Okunabililiği artırmak için kısa paragraflar, emoji ve trend hashtagler kullan. "
            f"Metal evler, prefabrik ve tiny house kültürüne ilgi duyan Avrupa'daki kitleye hitap et. "
            f"Harekete geçirici (CTA) ifadeler ekle. Metni orijinal anlamını koruyarak, Instagram'ın karakter sınırlamalarına uygun ama bilgilendirici olacak şekilde düzenle. "
            f"Çıktıyı {target_language} dilinde ver. Metin: \n\n{text}"
        )
    elif platform == "Facebook":
        format_prompt = (
            f"Aşağıdaki metni Facebook topluluğu için samimi, bilgilendirici ve etkileşim odaklı bir gönderiye dönüştür. "
            f"Paylaşımı teşvik eden sorular, topluluk odaklı ifadeler ve uygun hashtagler kullan. "
            f"Metal evler, prefabrik ve tiny house kültürüne ilgi duyan Avrupa'daki kitleye hitap et. "
            f"Video veya görsel içeriğe eşlik edebilecek, paylaşılabilir ve sohbeti başlatacak bir metin yaz. "
            f"Çıktıyı {target_language} dilinde ver. Metin: \n\n{text}"
        )
    elif platform == "LinkedIn":
        format_prompt = (
            f"Aşağıdaki metni LinkedIn profesyonel ağı için bilgilendirici, otoriter ve düşündürücü bir gönderiye dönüştür. "
            f"Sektörel içgörüler, profesyonel terimler ve konuyla ilgili hashtagler kullan. "
            f"Metal evler, prefabrik ve tiny house sektöründeki veya bu alana yatırım yapmayı düşünen Avrupa'daki profesyonellere hitap et. "
            f"Değer katan bilgiler sun ve tartışmayı teşvik et. "
            f"Çıktıyı {target_language} dilinde ver. Metin: \n\n{text}"
        )
    elif platform == "Genel Blog Yazısı":
        format_prompt = (
            f"Aşağıdaki metni bir blog yazısı formatına dönüştür. Blogun ana başlığını, alt başlıklarını ve paragraflarını açıkça belirt. "
            f"Okunabililiği artırmak için giriş, gelişme (alt başlıklar kullanarak) ve sonuç bölümleri oluştur. "
            f"Anahtar kelimelerle zenginleştirilmiş, bilgilendirici ve SEO dostu bir yapı kur. "
            f"Metal evler, prefabrik ve tiny house kültürüne ilgi duyan Avrupa'daki okuyucular için kapsamlı ve akıcı bir anlatım sağla. "
            f"Çıktıyı {target_language} dilinde ver. Metin: \n\n{text}"
        )
    elif platform == "E-posta Bülteni":
        format_prompt = (
            f"Aşağıdaki metni kısa, öz ve okuyucuyu harekete geçiren bir e-posta bülteni içeriğine dönüştür. "
            f"Net bir konu başlığı (subject line) öner, kısa giriş, ana faydaları vurgulayan maddeler veya kısa paragraflar ve net bir harekete geçirici mesaj (CTA) içer. "
            f"Metal evler, prefabrik ve tiny house kültürüyle ilgilenen Avrupa'daki abonelere hitap et. "
            f"Çıktıyı {target_language} dilinde ver. Metin: \n\n{text}"
        )
    else: # Varsayılan veya bilinmeyen platformlar için
        format_prompt = (
            f"Aşağıdaki metni genel bir sosyal medya platformu için uygun, ilgi çekici ve etkileşim artırıcı bir gönderi formatında yeniden yaz. "
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
            return "Hata: API anahtarı geçersiz veya yetkilendirme hatası. Lütfen anahtarınızı kontrol edin."
        return f"Hata: Metin formatlama hatası (AI): {e}"

# --- YouTube Video Fikri Oluşturma Fonksiyonu ---
@st.cache_data
def generate_youtube_idea(prompt_text, target_language="Türkçe"):
    model = genai.GenerativeModel('gemini-2.0-flash')
    youtube_prompt = (
        f"'{prompt_text}' konusunda bir YouTube videosu fikri oluştur. "
        f"Başlık önerileri, anahtar noktalar (video içeriği), kısa bir senaryo taslağı (giriş, gelişme, sonuç) ve potansiyel görsel/çekim fikirleri içermeli. "
        f"Çıktıyı {target_language} dilinde ver."
    )
    try:
        response = model.generate_content(youtube_prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "api key not valid" in error_msg.lower() or "authentication error" in error_msg.lower():
            return "Hata: API anahtarı geçersiz veya yetkilendirme hatası. Lütfen anahtarınızı kontrol edin."
        return f"Hata: YouTube video fikri oluşturma hatası (AI): {e}"

# --- Placeholder for AI Short Video Generation ---
def generate_short_video_placeholder(video_prompt_text, target_language="Türkçe"):
    return (f"Video oluşturma özelliği henüz entegre edilmedi. "
            f"'{video_prompt_text}' konusunda {target_language} dilinde bir video oluşturmak için "
            f"**RunwayML API**, **Pictory.ai API** veya **Synthesys.io API** gibi platformların API'leri gereklidir. "
            f"Bu işlem maliyetli olabilir ve uzun sürebilir.")

# --- Placeholder for Social Media Statistics ---
def fetch_social_media_stats_placeholder():
    return ("Sosyal Medya İstatistikleri (Geliştirme Aşamasında):\n\n"
            "Bu bölüm, API entegrasyonları tamamlandığında tüm sosyal medya hesaplarınızdaki (Facebook, Instagram, LinkedIn, YouTube) "
            "toplam takipçi, etkileşim, izlenme gibi anahtar metrikleri görselleştirecektir. "
            "Bu verilere erişim için her platformdan özel izinler ve kimlik doğrulama gereklidir.")


# --- Streamlit Uygulama Arayüzü ---
st.set_page_config(layout="wide") # Geniş sayfa düzeni
st.title("Premium Home AI Sosyal Medya Asistanı 🚀")

st.markdown("""
    Bu asistan, Premium Home için sosyal medya içerikleri oluşturmanıza, görselleri yorumlamanıza ve yeni fikirler üretmenize yardımcı olur.
    Metal evler, prefabrik yapılar ve Tiny House kültürü odaklı içerikler üretir.
    ---
""")

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
        st.session_state.last_generated_text = generated_content # Metni session state'e kaydet
        st.session_state.last_selected_language = selected_language
        st.markdown("### Oluşturulan Metin:")
        st.code(generated_content, language='markdown') # Metni kod bloğu olarak göster

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
            st.code(formatted_text, language='markdown') # Metni kod bloğu olarak göster

            # Sosyal Medya Paylaşım Linkleri
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
                <p style="font-size:12px; color:#666; margin-top:10px;"><i>Not: Instagram ve LinkedIn işletme sayfası paylaşımları için API entegrasyonu gereklidir. Bu butonlar manuel paylaşıma yönlendirir.</i></p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Önce 'Metin Oluştur' bölümünden bir metin oluşturun.")

# --- Görsel Yükle ve Yorumla Bölümü ---
st.header("Görsel Yükle ve Yorumla")
uploaded_file = st.file_uploader("Yorumlamak için bir görsel yükleyin", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Görsel', use_column_width=True)
    
    if st.button('Görseli Yorumla', type="secondary", key='interpret_image_button'):
        with st.spinner("Görsel yorumlanıyor..."):
            interpretation = interpret_image_gemini_vision(image)
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
        # İndirme butonu
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
            youtube_idea = generate_youtube_idea(youtube_prompt, "Türkçe")
        st.session_state.last_youtube_idea = youtube_idea
        st.markdown("### Oluşturulan YouTube Video Fikri:")
        st.code(youtube_idea, language='markdown')
    
with col6:
    # "YouTube Fikrini Video İçin Kullan" butonu
    if st.button('YouTube Fikrini Video İçin Kullan', type="secondary", key='use_for_video_creation_button'):
        if 'last_youtube_idea' in st.session_state and st.session_state.last_youtube_idea:
            st.session_state.video_creation_prompt_input_value = st.session_state.last_youtube_idea
            st.success("YouTube Fikri video istemine kopyalandı. Şimdi aşağıdan 'Video Oluştur' butonuna tıklayabilirsiniz.")
        else:
            st.error("Önce bir YouTube Fikri oluşturmanız gerekiyor.")

# --- AI ile Kısa Video Oluşturma (Placeholder) Bölümü ---
st.header("AI ile Kısa Video Oluştur (Geliştirme Aşamasında)")
st.markdown("<p style='font-size:13px; color:#555;'>*Yukarıdaki 'YouTube Video Fikri Oluştur' bölümünde üretilen son fikri kullanır.</p>", unsafe_allow_html=True)

video_creation_prompt_input = st.text_area(
    'Video Oluşturma İstem:',
    value=st.session_state.get('video_creation_prompt_input_value', ''), # Otomatik doldurma için
    placeholder='Video oluşturma istemi giriniz (Örn: Bir Tiny House\'un 15 saniyelik tanıtım videosu).',
    height=150,
    key='video_creation_prompt_input'
)

if st.button('Video Oluştur (API Gerekli)', type="secondary", key='generate_short_video_button'): # type="danger" -> "secondary"
    if not video_creation_prompt_input.strip():
        st.error("Lütfen video oluşturmak için bir istem girin veya YouTube fikri oluşturun.")
        st.stop()
    
    with st.spinner(f"Video oluşturma isteği: '{video_creation_prompt_input[:50]}...'"):
        generated_video_info = generate_short_video_placeholder(video_creation_prompt_input, "Türkçe")
    st.markdown("### Oluşturulan Video (Placeholder):")
    st.code(generated_video_info, language='markdown')

# --- Sosyal Medya İstatistikleri (Placeholder) Bölümü ---
st.header("Sosyal Medya İstatistikleri (Geliştirme Aşamasında)")
st.markdown("<p style='font-size:13px; color:#555;'>*API entegrasyonları tamamlandığında sosyal medya istatistikleriniz burada gösterilecektir.</p>", unsafe_allow_html=True)

if st.button('İstatistikleri Çek (API Gerekli)', type="secondary", key='fetch_stats_button'): # type="danger" -> "secondary"
    with st.spinner("İstatistikler çekiliyor... (API entegrasyonu gerekli)"):
        stats_text = fetch_social_media_stats_placeholder()
    st.markdown("### Toplam Sosyal Medya İstatistikleri:")
    st.code(stats_text, language='markdown')

st.markdown("---")
st.markdown("Developed with ❤️ by Premium Home AI Assistant")

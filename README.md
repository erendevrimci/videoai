# VideoAI Automation Pipeline

An automated pipeline for generating AI videos and publishing them to YouTube.

## Features

- Script generation using OpenAI
- Voice-over generation using ElevenLabs
- Automatic caption generation
- Video editing with clip selection based on script content
- YouTube title and description generation
- Automated YouTube uploads
- Multi-channel support

## Setup

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   ELEVENLABS_API_KEY=your_elevenlabs_key
   YOUTUBE_API_KEY=your_youtube_key
   YOUTUBE_CLIENT_ID=your_client_id
   YOUTUBE_CLIENT_SECRET=your_client_secret
   ```

3. Place video clips in the `clips/` directory and define them in `clips/clips_label.md`

4. Add background music MP3 files to the `background_music/` directory

## Usage

### Running the Full Pipeline

```bash
# Run the complete pipeline for all channels
python main.py

# Run for a specific channel
python main.py --channel 1

# Run specific steps only
python main.py --steps script,voice,video

# Run with a custom delay between channels
python main.py --delay 120
```

### Running Individual Steps

```bash
# Generate script
python write_script.py

# Generate voice for a specific channel
python voice_over.py 1

# Generate captions
python captions.py

# Edit video
python video_edit.py

# Generate title/description
python write_title_desc.py

# Upload to YouTube
python upload_video.py
```

## Configuration

The system uses a centralized configuration system in `config.py`. You can modify settings there instead of changing the code directly.

Key configuration options:
- API settings (models, parameters)
- Channel-specific settings (voice IDs, YouTube credentials)
- File paths
- Video editing parameters

## Directory Structure

- `clips/` - Video clip files and metadata
- `voice/` - Generated voice files
- `background_music/` - Background music files
- `outputs/` - Channel-specific output directories
  - `channel_1/` - Channel 1 outputs
  - `channel_2/` - Channel 2 outputs
  - `channel_3/` - Channel 3 outputs

## Channels

The system supports multiple YouTube channels, each with its own:
- Voice configuration
- YouTube credentials
- Output files

# RVL-CDIP Veri Seti Sunum Hazırlık Metni

Merhaba! RVL-CDIP veri seti hakkında sunumunuzda sorulabilecek muhtemel soruları ve cevaplarını içeren bir metin hazırladım. Bu bilgiler sunumunuz sırasında size yardımcı olacaktır.

## Temel Sorular ve Cevaplar

### RVL-CDIP nedir ve ne anlama gelir?
RVL-CDIP, "Ryerson Vision Lab Complex Document Information Processing" ifadesinin kısaltmasıdır. Bu, doküman görüntülerini sınıflandırmak için oluşturulmuş kapsamlı bir veri setidir.

### Veri seti kim tarafından geliştirilmiştir?
Veri seti Adam W. Harley, Alex Ufkes ve Konstantinos G. Derpanis tarafından geliştirilmiştir. Bu araştırmacılar 2015 yılında bu veri seti ile ilgili bir makale yayınlamışlardır.

### Veri setinin boyutu ve içeriği nedir?
- Toplam 400.000 gri tonlamalı görüntü içerir
- 16 farklı doküman sınıfı vardır
- Her sınıf için 25.000 görüntü bulunmaktadır
- 320.000 eğitim, 40.000 doğrulama ve 40.000 test görüntüsü içerir
- Görüntüler, en büyük boyutları 1000 pikseli geçmeyecek şekilde boyutlandırılmıştır

### Veri setindeki sınıflar nelerdir?
Veri setinde şu 16 sınıf bulunmaktadır:
1. Mektup (letter)
2. Form
3. E-posta
4. El yazısı
5. Reklam
6. Bilimsel rapor
7. Bilimsel yayın
8. Şartname
9. Dosya klasörü
10. Haber makalesi
11. Bütçe
12. Fatura
13. Sunum
14. Anket
15. Özgeçmiş
16. Not (memo)

### Bu veri seti nereden türetilmiştir?
RVL-CDIP, IIT-CDIP Test Collection 1.0'ın bir alt kümesidir. IIT-CDIP ise Legacy Tobacco Document Library'den (LTDL) alınmıştır. LTDL, Kaliforniya Üniversitesi, San Francisco tarafından 2007'de oluşturulmuştur.

### Veri setine nasıl erişebilirim?
Veri seti HuggingFace Datasets Kütüphanesi'nde bulunmaktadır. Ayrıca, Google Drive üzerinden de indirilebilir. Siteye göre iki dosya vardır:
- rvl-cdip.tar.gz (37GB)
- labels_only.tar.gz (6.1MB)

## Teknik Sorular ve Cevaplar

### Etiket dosyaları hangi formatta düzenlenmiştir?
Etiket dosyaları şu formatta görüntüleri ve kategorilerini listeler:
`path/to/the/image.tif category`
Kategoriler 0'dan 15'e kadar numaralandırılmıştır.

### Bu veri seti hangi çalışmalarda kullanılmıştır?
İlk olarak "Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval" adlı çalışmada kullanılmıştır. Bu çalışma, ICDAR 2015'te sunulmuştur.

### Veri setini kullanırken nasıl atıf yapmalıyım?
Veri setini kullanırken şu makaleye atıf yapılmalıdır:
A. W. Harley, A. Ufkes, K. G. Derpanis, "Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval," in ICDAR, 2015

### Veri setinin lisans durumu nedir?
RVL-CDIP, IIT-CDIP'in bir alt kümesidir ve Legacy Tobacco Document Library'den alınmıştır. Lisans bilgileri LTDL web sitesinde bulunabilir.

## Uygulama ve Araştırma Soruları

### Bu veri seti hangi yapay zeka uygulamalarında kullanılabilir?
- Doküman sınıflandırma sistemleri
- Optik Karakter Tanıma (OCR) modelleri geliştirme
- Doküman yapısı analizi
- Doküman geri erişim sistemleri
- Belge işleme otomasyonu

### Bu veri setiyle çalışırken karşılaşılabilecek zorluklar nelerdir?
- Görüntülerin gri tonlamalı olması bazı renk tabanlı ipuçlarının kaybolmasına neden olabilir
- 16 sınıf çeşitliliği, modellerin doğru sınıflandırma yapmasını zorlaştırabilir
- Veri setinin büyüklüğü (37GB) işleme ve depolama açısından zorluk çıkarabilir
- Tarihsel dokümanlar olduğu için modern doküman formatlarına tam uyum sağlamayabilir

### Bu veri seti ile ne tür modeller eğitilebilir?
- Evrişimli Sinir Ağları (CNN) tabanlı doküman sınıflandırıcıları
- Transfer öğrenimi ile doküman anlama modelleri
- Doküman görüntü işleme sistemleri
- OCR öncesi doküman hazırlama ve kategorizasyon araçları

### Veri setinin güncel araştırmalardaki konumu nedir?
Doküman anlama ve sınıflandırma alanında hala önemli bir kıyaslama (benchmark) veri seti olarak kullanılmaktadır. 2022'de HuggingFace Datasets kütüphanesine eklenmesi, güncelliğini ve önemini koruduğunu göstermektedir.
### Görüntü Özellikleri
- Tüm görüntüler gri tonlamalı TIFF formatındadır
- Çözünürlükler değişkendir ancak en büyük boyut 1000 piksel ile sınırlandırılmıştır
- Dosyaların çoğu taranmış dokümanlardan oluşmaktadır
- Görüntü kalitesi değişkendir, bazı dokümanlar düşük tarama kalitesine sahiptir

[Kaynak: https://adamharley.com/rvl-cdip/](https://adamharley.com/rvl-cdip/)

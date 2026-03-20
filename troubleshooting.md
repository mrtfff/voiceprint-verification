# 🔧 Ses İmzası Sistemi - Sorun ve Çözüm Rehberi

Geliştirme sırasında karşılaşılabilecek sorunlar ve çözümleri.

---

## 1. SpeechBrain / torchaudio Uyumluluk Sorunu

**Sorun:** `AttributeError: module 'torchaudio' has no attribute 'list_audio_backends'`

**Neden:** SpeechBrain 1.0.3, torchaudio 2.10.0 ile uyumsuz. `list_audio_backends` fonksiyonu torchaudio'nun yeni sürümlerinde kaldırıldı.

**Çözümler (öncelik sırasıyla):**

1. **SpeechBrain'i güncelle:**
   ```powershell
   .\venv\Scripts\pip.exe install -U speechbrain
   ```

2. **Eğer güncelleme çalışmazsa - Monkey-patch:**
   ```python
   import torchaudio
   if not hasattr(torchaudio, 'list_audio_backends'):
       torchaudio.list_audio_backends = lambda: ['soundfile']
   ```

3. **Son çare - torchaudio sürümünü düşür:**
   ```powershell
   .\venv\Scripts\pip.exe install torchaudio==2.1.0
   ```

---

## 2. Mikrofon Erişim Sorunları

**Sorun:** `sounddevice.PortAudioError: Error opening InputStream` veya boş kayıt

**Çözümler:**

1. **Windows mikrofon izni kontrol et:**
   - Ayarlar → Gizlilik → Mikrofon → Uygulamalara mikrofon erişimi izni ver

2. **Doğru cihaz seçimi:**
   ```python
   import sounddevice as sd
   print(sd.query_devices())  # Mevcut cihazları listele
   sd.default.device = 1       # İstenen cihaz ID'sini ayarla
   ```

3. **Bluetooth kulaklık sorunu:**
   - Bluetooth kulaklık mikrofonu genellikle düşük kaliteledir (8kHz SCO profili)
   - Mümkünse kablolu mikrofon kullanın

---

## 3. Düşük Kalite Kayıt / Çok Sessiz Ses

**Sorun:** Embedding kalitesi düşük, doğrulama skoru güvenilmez

**Belirtiler:** Kayıt dalgaformu çok düz, SNR < 10dB

**Çözümler:**

1. **Mikrofon mesafesi:** 15-30 cm arası ideal
2. **Gain/volume kontrolü:** Windows ses ayarlarından mikrofon seviyesini artır
3. **Ön-işleme ekle:**
   ```python
   import numpy as np
   # DC offset kaldırma
   audio = audio - np.mean(audio)
   # Peak normalization
   audio = audio / (np.max(np.abs(audio)) + 1e-8)
   ```

---

## 4. Gürültülü Ortamda Düşük Doğruluk

**Sorun:** Eşleşme skoru çok düşük çıkıyor, aynı kişi bile tanınmıyor

**Çözümler:**

1. **Kayıt ortamını iyileştir:**
   - Sessiz bir odada kayıt yap
   - Klima, fan gibi sürekli gürültü kaynaklarını kapat

2. **Birden fazla segment ortalaması:**
   - Tek bir kayıt yerine 3-5 segment kaydedip embedding ortalaması al
   - Bu, gürültüye karşı direnci artırır

3. **SNR kontrolü ekle:**
   ```python
   def kontrol_snr(audio, esik=10):
       sinyal = np.mean(audio**2)
       # Sessiz bölümleri gürültü olarak tahmin et
       sessiz = audio[np.abs(audio) < np.percentile(np.abs(audio), 10)]
       gurultu = np.mean(sessiz**2) + 1e-10
       snr = 10 * np.log10(sinyal / gurultu)
       return snr > esik
   ```

---

## 5. Farklı Mikrofon Kullanımı

**Sorun:** Kayıt bir mikrofonla yapılıp doğrulama başka bir mikrofonla yapıldığında skor düşüyor

**Neden:** Her mikrofonun farklı frekans yanıtı var, bu da embedding'i etkiliyor

**Çözümler:**

1. **Aynı mikrofonu kullanmaya çalış** (en basit çözüm)
2. **Threshold'u biraz gevşet:** Farklı mikrofon durumunda eşik değerini 0.25'ten 0.35'e çıkar
3. **Birden fazla mikrofon ile enrollment:** Kayıt aşamasında farklı mikrofonlarla segment kaydet

---

## 6. Model Yükleme Yavaş

**Sorun:** ECAPA-TDNN modeli her seferinde yüklendiğinde 5-10 saniye bekliyor

**Çözüm:** Modeli singleton olarak yükle, uygulamanın başlangıcında bir kez yükle:
```python
class VoiceEncoder:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

---

## 7. CUDA / GPU Sorunları

**Sorun:** `RuntimeError: CUDA out of memory` veya GPU algılanmıyor

**Bilgi:** Mevcut kurulumda `torch 2.10.0+cpu` yüklü, yani sadece CPU kullanılıyor.

**Çözüm:** ECAPA-TDNN embedding çıkarma işlemi CPU'da yeterince hızlıdır (~1 saniye). GPU gerekmez. Eğer GPU kullanmak isterseniz:
```powershell
.\venv\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 8. Embedding Boyutu Uyumsuzluğu

**Sorun:** Kayıtlı embedding ile yeni çıkarılan embedding boyutları eşleşmiyor

**Neden:** Farklı model kullanılmış olabilir

**Çözüm:** Her embedding'in metadata'sında model adı ve boyutunu sakla, karşılaştırma öncesi kontrol et:
```python
if saved_embedding.shape != new_embedding.shape:
    raise ValueError(f"Embedding boyutu uyumsuz: {saved_embedding.shape} != {new_embedding.shape}")
```

---

## 9. WAV Dosyası Format Sorunları

**Sorun:** `ValueError: File format not supported` veya bozuk okuma

**Çözümler:**

1. **Sadece 16-bit PCM WAV kullan** (en güvenli format)
2. **torchaudio yerine scipy kullan:** torchaudio backend sorunlarından kaçınmak için:
   ```python
   from scipy.io.wavfile import read
   sr, audio = read("dosya.wav")
   ```
3. **Sample rate kontrol et:** Model 16kHz bekliyor, farklıysa resample et:
   ```python
   import torchaudio
   audio, sr = torchaudio.load("dosya.wav")
   if sr != 16000:
       resampler = torchaudio.transforms.Resample(sr, 16000)
       audio = resampler(audio)
   ```

---

## 10. Cosine Similarity Yorumlama

**Kılavuz:**

| Skor Aralığı | Anlam |
|---------------|-------|
| 0.65 - 1.00 | Kesinlikle aynı kişi |
| 0.45 - 0.65 | Muhtemelen aynı kişi |
| 0.30 - 0.45 | Belirsiz bölge (threshold'a bağlı) |
| 0.15 - 0.30 | Muhtemelen farklı kişi |
| < 0.15 | Kesinlikle farklı kişi |

> **Not:** Bu değerler ECAPA-TDNN modeline özgüdür ve ortam koşullarına göre değişebilir. Gerçek threshold değerini kendi test verilerinizle kalibre edin.

import time
import sounddevice as sd

from scipy.io.wavfile import write

def ses_kaydet(dosya_adi, saniye=5):
    fs = 16000  # Örnekleme hızı (SpeechBrain 16kHz formatını tercih eder)
    
    print(f"Lütfen {saniye} saniye boyunca konuşun... (Kayıt başlıyor)")
    # Mikrofonu dinle
    ses_verisi = sd.rec(int(saniye * fs), samplerate=fs, channels=1)
    sd.wait()  # Kaydın bitmesini bekle
    
    # Dosyayı kaydet
    write(dosya_adi, fs, ses_verisi)
    print(f"Kayıt tamamlandı ve '{dosya_adi}' olarak kaydedildi.\n")


# Kendi orjinal sesinizi kaydedin
input   ("Kendi sesinizi kaydedin. Enter'a basın...")
time.sleep(3)
ses_kaydet("benim_referans_sesim.wav", saniye=30)

# Test etmek için başka bir ses veya kendi sesinizi tekrar kaydedin
input("Test etmek için başka bir ses veya kendi sesinizi tekrar kaydedin. Enter'a basın...")
ses_kaydet("test_edilecek_ses.wav", saniye=7)
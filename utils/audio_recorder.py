"""
Gelişmiş ses kaydı modülü - kalite kontrolü ve geri bildirim ile
"""
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import SAMPLE_RATE, CHANNELS
from utils.audio_utils import normalize_audio, check_audio_quality, trim_silence


class AudioRecorder:
    """Gelişmiş ses kayıt sınıfı."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
    
    def list_devices(self) -> None:
        """Mevcut ses cihazlarını listele."""
        print("\n🎤 Mevcut Ses Cihazları:")
        print("=" * 60)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                marker = " ← VARSAYILAN" if i == sd.default.device[0] else ""
                print(f"  [{i}] {dev['name']} (Giriş: {dev['max_input_channels']} kanal){marker}")
        print()
    
    def record(self, duration: float, show_countdown: bool = True) -> np.ndarray:
        """
        Belirtilen süre boyunca ses kaydet.
        
        Args:
            duration: Kayıt süresi (saniye)
            show_countdown: Geri sayım göster
            
        Returns:
            numpy array - float32 formatında ses verisi
        """
        if show_countdown:
            print(f"\n🔴 Kayıt {duration} saniye sürecek...")
            for i in range(3, 0, -1):
                print(f"   {i}...")
                time.sleep(1)
            print("   🎙️ KONUŞUN!\n")
        
        # Kayıt yap
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32'
        )
        
        # Kayıt süresini göster
        start_time = time.time()
        while sd.get_stream().active:
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)
            print(f"\r   ⏱️ Kalan süre: {remaining:.1f}s ", end="", flush=True)
            time.sleep(0.1)
        
        sd.wait()
        print(f"\r   ✅ Kayıt tamamlandı!          ")
        
        # Mono'ya çevir
        audio = audio.flatten()
        
        return audio
    
    def record_with_quality_check(self, duration: float, 
                                    max_retries: int = 3) -> np.ndarray | None:
        """
        Ses kaydı yap ve kalite kontrolü uygula. 
        Kalite yetersizse kullanıcıya tekrar kayıt imkanı sun.
        
        Returns:
            numpy array veya None (kullanıcı iptal ettiyse)
        """
        for attempt in range(max_retries):
            audio = self.record(duration)
            
            # Kalite kontrolü
            quality = check_audio_quality(audio)
            
            if quality['passed']:
                print(f"   📊 Ses kalitesi: İYİ (SNR: {quality['snr']}dB, RMS: {quality['rms']})")
                return normalize_audio(trim_silence(audio))
            
            # Kalite sorunlarını göster
            print(f"\n   ⚠️ Ses kalitesi sorunları tespit edildi:")
            for issue in quality['issues']:
                print(f"      • {issue}")
            
            if attempt < max_retries - 1:
                retry = input(f"\n   Tekrar kayıt yapmak ister misiniz? ({max_retries - attempt - 1} deneme kaldı) [E/h]: ")
                if retry.lower() == 'h':
                    print("   Mevcut kayıt kullanılacak.")
                    return normalize_audio(trim_silence(audio))
            else:
                print("   Maksimum deneme sayısına ulaşıldı. Mevcut kayıt kullanılacak.")
                return normalize_audio(trim_silence(audio))
        
        return None
    
    def save_wav(self, audio: np.ndarray, filepath: str) -> None:
        """Ses verisini WAV dosyası olarak kaydet."""
        # float32 -> int16 dönüşümü
        audio_int16 = (audio * 32767).astype(np.int16)
        wav_write(filepath, self.sample_rate, audio_int16)
        print(f"   💾 Kaydedildi: {filepath}")

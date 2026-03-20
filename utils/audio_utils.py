"""
Ses yardımcı fonksiyonları - ön-işleme, kalite kontrolü, normalizasyon
"""
import numpy as np
from scipy.io.wavfile import read as wav_read, write as wav_write


def load_wav(file_path: str) -> tuple[int, np.ndarray]:
    """
    WAV dosyasını yükle ve float32 formatına dönüştür.
    
    Returns:
        (sample_rate, audio_data) - audio_data [-1, 1] aralığında float32
    """
    sr, audio = wav_read(file_path)
    
    # int16 -> float32 dönüşümü
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Stereo -> Mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    return sr, audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """DC offset kaldırma ve peak normalization."""
    # DC offset kaldır
    audio = audio - np.mean(audio)
    
    # Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95  # Küçük headroom bırak
    
    return audio


def calculate_snr(audio: np.ndarray) -> float:
    """
    Sinyal-gürültü oranını tahmin et (dB).
    En sessiz %10'luk bölümü gürültü olarak kabul eder.
    """
    abs_audio = np.abs(audio)
    threshold = np.percentile(abs_audio, 10)
    
    noise = audio[abs_audio <= threshold]
    signal = audio[abs_audio > threshold]
    
    if len(noise) == 0 or len(signal) == 0:
        return 0.0
    
    noise_power = np.mean(noise ** 2) + 1e-10
    signal_power = np.mean(signal ** 2) + 1e-10
    
    return 10 * np.log10(signal_power / noise_power)


def check_clipping(audio: np.ndarray, threshold: float = 0.99) -> float:
    """Ses verisindeki clipping oranını hesapla."""
    clipped = np.sum(np.abs(audio) >= threshold)
    return clipped / len(audio)


def check_audio_quality(audio: np.ndarray, min_snr: float = 10.0, 
                         min_level: float = 0.01, max_clip: float = 0.01) -> dict:
    """
    Ses kalitesi kontrolleri.
    
    Returns:
        dict: {'passed': bool, 'snr': float, 'rms': float, 'clipping': float, 'issues': list}
    """
    rms = np.sqrt(np.mean(audio ** 2))
    snr = calculate_snr(audio)
    clipping = check_clipping(audio)
    
    issues = []
    
    if rms < min_level:
        issues.append(f"Ses seviyesi çok düşük (RMS: {rms:.4f}, minimum: {min_level})")
    
    if snr < min_snr:
        issues.append(f"Sinyal-gürültü oranı düşük (SNR: {snr:.1f}dB, minimum: {min_snr}dB)")
    
    if clipping > max_clip:
        issues.append(f"Clipping tespit edildi ({clipping*100:.1f}%, maksimum: {max_clip*100:.1f}%)")
    
    return {
        'passed': len(issues) == 0,
        'snr': round(snr, 2),
        'rms': round(rms, 4),
        'clipping': round(clipping, 4),
        'issues': issues
    }


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Basit resample fonksiyonu (scipy kullanarak)."""
    if orig_sr == target_sr:
        return audio
    
    from scipy.signal import resample as scipy_resample
    
    num_samples = int(len(audio) * target_sr / orig_sr)
    return scipy_resample(audio, num_samples).astype(np.float32)


def trim_silence(audio: np.ndarray, threshold: float = 0.02, 
                  frame_length: int = 1024) -> np.ndarray:
    """
    Başlangıç ve sondaki sessiz kısımları kes.
    Çok kısa veya tamamen sessiz kayıtlarda orijinali döndürür.
    """
    abs_audio = np.abs(audio)
    
    # Frame bazlı enerji hesapla
    n_frames = len(abs_audio) // frame_length
    if n_frames == 0:
        return audio
    
    energies = np.array([
        np.mean(abs_audio[i * frame_length:(i + 1) * frame_length])
        for i in range(n_frames)
    ])
    
    # Sessiz olmayan frame'leri bul
    active_frames = np.where(energies > threshold)[0]
    
    if len(active_frames) == 0:
        return audio  # Tamamen sessiz - orijinali döndür
    
    start = max(0, active_frames[0] * frame_length - frame_length)
    end = min(len(audio), (active_frames[-1] + 2) * frame_length)
    
    return audio[start:end]

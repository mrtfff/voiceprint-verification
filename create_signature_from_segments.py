"""
Tek seferlik kullanım: Mevcut WAV segmentlerinden ERes2Net ses imzası oluştur.

Kullanım:
    .\\venv\\Scripts\\python.exe create_signature_from_segments.py
"""
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import fix_torchaudio_compat, MODEL_ERES2NET
fix_torchaudio_compat()

from models.eres2net_encoder import ERes2NetEncoder
from models.signature_store import SignatureStore
from utils.audio_utils import load_wav, normalize_audio, trim_silence_vad, resample
from config.settings import SAMPLE_RATE

# ─── Ayarlar ─────────────────────────────────────────────────────────────────
SEGMENTS_DIR = r"c:\Users\Mert\Documents\projelerim\ses-imzam\signatures\eres2net\Mert2_segments"
USER_ID = "Mert2"
# ─────────────────────────────────────────────────────────────────────────────


def load_segment(wav_path: str):
    """WAV dosyasını yükle, ön-işle ve embedding için hazırla."""
    sr, audio = load_wav(wav_path)
    if sr != SAMPLE_RATE:
        print(f"   ⚠️ Resample: {sr}Hz → {SAMPLE_RATE}Hz")
        audio = resample(audio, sr, SAMPLE_RATE)
    
    trimmed, speech_dur = trim_silence_vad(audio)
    audio_clean = normalize_audio(trimmed)
    
    return audio_clean, speech_dur


def main():
    print("=" * 60)
    print("   🔧 WAV SEGMENTLERİNDEN İMZA OLUŞTUR (Tek Seferlik)")
    print("=" * 60)
    print(f"\n   Kaynak : {SEGMENTS_DIR}")
    print(f"   Kullanıcı: {USER_ID}")
    print(f"   Model  : ERes2Net (ModelScope)\n")
    
    # WAV dosyalarını sıralı yükle
    wav_files = sorted(glob.glob(os.path.join(SEGMENTS_DIR, "segment_*.wav")))
    
    if not wav_files:
        print(f"❌ {SEGMENTS_DIR} dizininde segment_*.wav dosyası bulunamadı.")
        return
    
    print(f"📂 Bulunan segmentler: {len(wav_files)}")
    
    segments = []
    for wav_path in wav_files:
        fname = os.path.basename(wav_path)
        audio, speech_dur = load_segment(wav_path)
        print(f"   ✅ {fname} — net konuşma: {speech_dur:.1f}s, örnekler: {len(audio)}")
        segments.append(audio)
    
    # Model yükle
    print(f"\n{'─' * 50}")
    encoder = ERes2NetEncoder()
    
    # Embedding çıkar
    print(f"\n{'─' * 50}")
    print(f"🧠 {len(segments)} segmentten kalite ağırlıklı embedding çıkarılıyor...")
    embedding = encoder.extract_multi_segment_embedding(segments)
    
    print(f"\n   ✅ Embedding boyutu : {embedding.shape[0]}")
    print(f"   ✅ L2 norm          : {float(embedding @ embedding):.4f}")
    
    # Kaydet
    store = SignatureStore(model_name=MODEL_ERES2NET)
    
    if store.exists(USER_ID):
        overwrite = input(f"\n⚠️  '{USER_ID}' zaten kayıtlı. Üzerine yaz? [e/H]: ").strip()
        if overwrite.lower() != 'e':
            print("İptal edildi.")
            return
    
    metadata = {
        "segment_count": len(segments),
        "segments_dir": SEGMENTS_DIR,
        "source": "created_from_existing_wavs",
    }
    save_path = store.save(USER_ID, embedding, metadata)
    
    print(f"\n💾 İmza kaydedildi: {save_path}")
    print(f"\n✅ '{USER_ID}' için ERes2Net ses imzası hazır!")
    print(f"   Artık main.py üzerinden doğrulama testi yapabilirsiniz.")
    print("=" * 60)


if __name__ == "__main__":
    main()

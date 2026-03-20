"""
Ses İmzası Kayıt (Enrollment) Pipeline'ı

Bu script:
1. Kullanıcıdan isim alır
2. 3 ayrı segment (farklı cümleler) kaydeder
3. Her segmentten ECAPA-TDNN embedding çıkarır
4. Ortalama embedding'i hesaplar (daha güçlü imza)
5. İmzayı signatures/ dizinine kaydeder
6. Doğrulama testi: hemen kısa bir kayıt alıp eşleşme skoru gösterir
"""
import sys
import os

# Proje kökünü path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    fix_torchaudio_compat, 
    ENROLLMENT_DURATION, 
    ENROLLMENT_SEGMENTS,
    EMBEDDING_DIM,
    VERIFICATION_DURATION,
    VERIFICATION_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
)
fix_torchaudio_compat()

from utils.audio_recorder import AudioRecorder
from models.voice_encoder import VoiceEncoder
from models.signature_store import SignatureStore


# ─── Yardımcı Cümleler ───────────────────────────────────────────────────────
ENROLLMENT_PROMPTS = [
    "1️⃣  Doğal konuşma hızınızda, herhangi bir konu hakkında konuşun.\n"
    "     (Örnek: Bugün neler yaptığınızı anlatın)",
    
    "2️⃣  Farklı bir tonla veya farklı bir konu hakkında konuşun.\n"
    "     (Örnek: Bir anınızı veya planınızı anlatın)",
    
    "3️⃣  Son segment: sayıları sayın veya bir şiir okuyun.\n"
    "     (Örnek: 1'den 20'ye kadar sayın)",
]


def main():
    print("=" * 60)
    print("   🎤 SES İMZASI KAYIT SİSTEMİ (Enrollment)")
    print("   ECAPA-TDNN | 192-dim Voiceprint")
    print("=" * 60)
    
    recorder = AudioRecorder()
    store = SignatureStore()
    
    # ─── Mevcut kullanıcıları göster ──────────────────────────────────────────
    existing_users = store.list_users()
    if existing_users:
        print("\n📋 Kayıtlı kullanıcılar:")
        for user in existing_users:
            print(f"   • {user['user_id']} (Kayıt: {user.get('created_at', 'N/A')[:10]})")
    
    # ─── Kullanıcı bilgisi ────────────────────────────────────────────────────
    print()
    user_id = input("👤 Adınızı girin (ses imzası ID'si): ").strip()
    
    if not user_id:
        print("❌ Geçerli bir isim girmelisiniz.")
        return
    
    if store.exists(user_id):
        print(f"\n⚠️  '{user_id}' zaten kayıtlı.")
        print(f"   [1] Sadece doğrulama testi yap")
        print(f"   [2] Üzerine yaz (yeni kayıt)")
        print(f"   [3] İptal")
        choice = input("   Seçiminiz [1/2/3]: ").strip()
        
        if choice == '1':
            # Sadece test modu
            encoder = VoiceEncoder()
            saved_embedding, _ = store.load(user_id)
            print(f"\n🔍 '{user_id}' imzası yüklendi. Test kaydı alınacak.")
            print(f"   (Süre: {VERIFICATION_DURATION} saniye)")
            input("   Hazır olduğunuzda Enter'a basın...")
            
            test_audio = recorder.record_with_quality_check(VERIFICATION_DURATION, max_retries=2)
            if test_audio is not None:
                test_embedding = encoder.extract_embedding(test_audio)
                score = VoiceEncoder.cosine_similarity(saved_embedding, test_embedding)
                interpretation = VoiceEncoder.interpret_score(score)
                
                print(f"\n{'─' * 50}")
                print(f"📊 SONUÇ")
                print(f"   Benzerlik skoru : {score:.4f}")
                print(f"   Yorum           : {interpretation}")
                print(f"   Eşik değeri     : {VERIFICATION_THRESHOLD}")
                
                if score >= VERIFICATION_THRESHOLD:
                    print(f"\n   ✅ DOĞRULANDI!")
                    if score >= HIGH_CONFIDENCE_THRESHOLD:
                        print(f"   🌟 Yüksek güvenle eşleşme!")
                else:
                    print(f"\n   ⚠️ Skor eşik değerinin altında.")
            return
        elif choice == '2':
            print("   Yeni kayıt başlatılıyor...")
        else:
            print("İptal edildi.")
            return
    
    # ─── Mikrofon kontrolü ────────────────────────────────────────────────────
    recorder.list_devices()
    input("Enter'a basarak kayda başlayın...")
    
    # ─── Model yükle ──────────────────────────────────────────────────────────
    encoder = VoiceEncoder()
    
    # ─── Segment kayıtları ────────────────────────────────────────────────────
    segments = []
    segment_wavs_dir = os.path.join("signatures", user_id + "_segments")
    os.makedirs(segment_wavs_dir, exist_ok=True)
    
    for i in range(ENROLLMENT_SEGMENTS):
        print(f"\n{'─' * 50}")
        print(f"📝 Segment {i+1}/{ENROLLMENT_SEGMENTS}")
        print(ENROLLMENT_PROMPTS[i] if i < len(ENROLLMENT_PROMPTS) else "Herhangi bir şey konuşun.")
        print(f"   (Süre: {ENROLLMENT_DURATION} saniye)")
        
        input("   Hazır olduğunuzda Enter'a basın...")
        
        audio = recorder.record_with_quality_check(ENROLLMENT_DURATION)
        
        if audio is None:
            print("❌ Kayıt başarısız. İptal edildi.")
            return
        
        segments.append(audio)
        
        # Backup olarak WAV kaydet
        wav_path = os.path.join(segment_wavs_dir, f"segment_{i+1}.wav")
        recorder.save_wav(audio, wav_path)
    
    # ─── Embedding çıkarma ────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("🧠 Ses imzası çıkarılıyor...")
    
    embedding = encoder.extract_multi_segment_embedding(segments)
    
    print(f"   ✅ İmza boyutu: {embedding.shape[0]} boyut")
    print(f"   ✅ L2 norm: {float(embedding @ embedding):.4f}")
    
    # ─── İmzayı kaydet ────────────────────────────────────────────────────────
    metadata = {
        "segment_count": len(segments),
        "segment_duration": ENROLLMENT_DURATION,
        "total_duration": ENROLLMENT_DURATION * len(segments),
        "segments_dir": segment_wavs_dir,
    }
    
    save_path = store.save(user_id, embedding, metadata)
    print(f"\n💾 Ses imzası kaydedildi: {save_path}")
    
    # ─── Doğrulama testi ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("🔍 DOĞRULAMA TESTİ")
    print(f"   Şimdi kısa bir kayıt alıp imzanızla karşılaştıracağız.")
    print(f"   (Süre: {VERIFICATION_DURATION} saniye)")
    
    do_test = input("\n   Doğrulama testi yapmak ister misiniz? [E/h]: ").strip()
    
    if do_test.lower() != 'h':
        input("   Hazır olduğunuzda Enter'a basın...")
        
        test_audio = recorder.record_with_quality_check(VERIFICATION_DURATION, max_retries=2)
        
        if test_audio is not None:
            test_embedding = encoder.extract_embedding(test_audio)
            score = VoiceEncoder.cosine_similarity(embedding, test_embedding)
            interpretation = VoiceEncoder.interpret_score(score)
            
            print(f"\n{'─' * 50}")
            print(f"📊 SONUÇ")
            print(f"   Benzerlik skoru : {score:.4f}")
            print(f"   Yorum           : {interpretation}")
            print(f"   Eşik değeri     : {VERIFICATION_THRESHOLD}")
            
            if score >= VERIFICATION_THRESHOLD:
                print(f"\n   ✅ SES İMZASI BAŞARIYLA DOĞRULANDI!")
                if score >= HIGH_CONFIDENCE_THRESHOLD:
                    print(f"   🌟 Yüksek güvenle eşleşme!")
            else:
                print(f"\n   ⚠️ Skor eşik değerinin altında.")
                print(f"   İpucu: Kayıt ortamınızın sessiz olduğundan emin olun.")
    
    # ─── Özet ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"📋 KAYIT ÖZETİ")
    print(f"   Kullanıcı    : {user_id}")
    print(f"   Segmentler   : {len(segments)} x {ENROLLMENT_DURATION}s")
    print(f"   İmza boyutu  : {EMBEDDING_DIM} boyut")
    print(f"   Konum        : {save_path}")
    print(f"\n   Gelecekte bu imza, gerçek zamanlı konuşucu doğrulama")
    print(f"   sisteminde kullanılabilir.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

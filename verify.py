"""
Ses İmzası Doğrulama (Verification) Script'i

Kayıtlı bir kullanıcının ses imzasını kısa bir kayıtla doğrular.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    fix_torchaudio_compat,
    VERIFICATION_DURATION,
    VERIFICATION_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
)
fix_torchaudio_compat()

from utils.audio_recorder import AudioRecorder
from models.voice_encoder import VoiceEncoder
from models.signature_store import SignatureStore


def main():
    print("=" * 60)
    print("   🔍 SES İMZASI DOĞRULAMA (Verification)")
    print("=" * 60)

    store = SignatureStore()
    recorder = AudioRecorder()

    # ─── Kayıtlı kullanıcıları göster ────────────────────────────────────────
    users = store.list_users()
    if not users:
        print("\n❌ Henüz kayıtlı kullanıcı yok. Önce enroll.py çalıştırın.")
        return

    print("\n📋 Kayıtlı kullanıcılar:")
    for i, user in enumerate(users):
        print(f"   [{i+1}] {user['user_id']} (Kayıt: {user.get('created_at', 'N/A')[:10]})")

    # ─── Kullanıcı seç ────────────────────────────────────────────────────────
    choice = input("\n👤 Doğrulanacak kullanıcı adı veya numarası: ").strip()

    # Numara ile seçim
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(users):
            user_id = users[idx]['user_id']
        else:
            print("❌ Geçersiz numara.")
            return
    except ValueError:
        user_id = choice

    if not store.exists(user_id):
        print(f"❌ '{user_id}' adında kayıtlı kullanıcı bulunamadı.")
        return

    # ─── İmzayı yükle ────────────────────────────────────────────────────────
    saved_embedding, profile = store.load(user_id)
    print(f"\n✅ '{user_id}' imzası yüklendi.")
    print(f"   Kayıt tarihi  : {profile.get('created_at', 'N/A')[:19]}")
    print(f"   Segment sayısı : {profile.get('segment_count', 'N/A')}")

    # ─── Model yükle ─────────────────────────────────────────────────────────
    encoder = VoiceEncoder()

    # ─── Test döngüsü ────────────────────────────────────────────────────────
    while True:
        print(f"\n{'─' * 50}")
        print(f"🎙️ {VERIFICATION_DURATION} saniyelik test kaydı alınacak.")
        input("   Hazır olduğunuzda Enter'a basın...")

        test_audio = recorder.record_with_quality_check(VERIFICATION_DURATION, max_retries=2)

        if test_audio is None:
            print("❌ Kayıt başarısız.")
            continue

        test_embedding = encoder.extract_embedding(test_audio)
        score = VoiceEncoder.cosine_similarity(saved_embedding, test_embedding)
        interpretation = VoiceEncoder.interpret_score(score)

        print(f"\n{'─' * 50}")
        print(f"📊 SONUÇ")
        print(f"   Karşılaştırılan : {user_id}")
        print(f"   Benzerlik skoru : {score:.4f}")
        print(f"   Yorum           : {interpretation}")
        print(f"   Eşik değeri     : {VERIFICATION_THRESHOLD}")

        if score >= VERIFICATION_THRESHOLD:
            print(f"\n   ✅ DOĞRULANDI — Bu ses '{user_id}' kişisine ait!")
            if score >= HIGH_CONFIDENCE_THRESHOLD:
                print(f"   🌟 Yüksek güvenle eşleşme!")
        else:
            print(f"\n   ❌ DOĞRULANAMADI — Bu ses '{user_id}' kişisine ait değil.")

        again = input("\n   Tekrar test etmek ister misiniz? [E/h]: ").strip()
        if again.lower() == 'h':
            break

    print("\n👋 Doğrulama tamamlandı.")


if __name__ == "__main__":
    main()

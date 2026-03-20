"""
Ses İmzası Sistemi - Ana Menü

Tüm işlemleri tek bir yerden yönetin:
  1. Yeni ses imzası kaydet (Enrollment)
  2. Ses doğrulama testi (Verification)
  3. Kayıtlı kullanıcıları listele
  4. Kullanıcı sil
"""
import sys
import os

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
SEGMENT_NOTE = (
    "   💡 Metni tamamen okumak zorunda değilsiniz!\n"
    "      Önemli olan tonlamanıza ve doğal konuşmanıza dikkat etmeniz."
)

ENROLLMENT_PROMPTS = [
    "1️⃣  Aşağıdaki metni normal konuşma hızınızda okuyun.\n"
    "     (Türkçedeki 29 harfin tamamını içerir)\n\n"
    '     "Güneşli bir gökyüzünde, fırtınanın habercisi olan siyah\n'
    '      bulutlar aniden belirdi. Çiftçi, toprağın sertleştiğini\n'
    '      fark edince küreğini bırakıp, küçük kulübesinde mola verdi.\n'
    '      Jandarma ekibi ise yağan yağmurun şiddetini ölçerken,\n'
    '      vadi boyunca uzanan dar yolları ve sarp yamaçları dikkatle\n'
    '      izledi. Pijamalı çocuk, öğle vakti jöleli tatlısını yerken\n'
    '      dışarıdaki bu karmaşayı şaşkınlıkla seyrediyordu. Herkes\n'
    '      için güç bir gündü, fakat doğa tüm görkemiyle hâlâ\n'
    '      büyüleyiciydi."',

    "2️⃣  Heyecanlı bir tonla, sanki bir arkadaşınıza şaşırdığınız\n"
    "     bir olayı anlatır gibi okuyun.\n\n"
    '     "Biliyor musun, dün akşam tam köşeyi dönerken inanılmaz\n'
    '      bir manzarayla karşılaştım! Gökyüzü o kadar parlaktı ki,\n'
    '      sanki tüm yıldızlar bir araya gelmiş ve muhteşem bir şölen\n'
    '      başlatmış gibiydi; gerçekten büyüleyici bir andı!"',

    "3️⃣  Bu metni sakin, tane tane ve her kelimenin sonundaki harfi\n"
    "     netleştirerek okuyun.\n\n"
    '     "Bir, iki, üç, dört, beş...\n'
    '      Gökyüzünün mavisinde, süzülen beyaz bir bulut gibiyim.\n'
    '      Sert rüzgarlar esse de, yolumdan asla dönmem.\n'
    '      Dokuz, on ve bitti!"',
]


def banner():
    print("\n" + "=" * 60)
    print("   🎤 SES İMZASI SİSTEMİ")
    print("   ECAPA-TDNN | 192-dim Voiceprint")
    print("=" * 60)


def menu():
    print("\n   [1] 🆕  Yeni ses imzası kaydet")
    print("   [2] 🔍  Ses doğrulama testi")
    print("   [3] 📋  Kayıtlı kullanıcıları listele")
    print("   [4] 🗑️  Kullanıcı sil")
    print("   [0] 🚪  Çıkış")
    return input("\n   Seçiminiz: ").strip()


# ─── 1. ENROLLMENT ───────────────────────────────────────────────────────────
def do_enroll(recorder: AudioRecorder, encoder: VoiceEncoder, store: SignatureStore):
    print(f"\n{'=' * 60}")
    print("   🆕 YENİ SES İMZASI KAYDI")
    print(f"{'=' * 60}")

    user_id = input("\n👤 Adınızı girin: ").strip()
    if not user_id:
        print("❌ Geçerli bir isim girmelisiniz.")
        return

    if store.exists(user_id):
        overwrite = input(f"⚠️  '{user_id}' zaten kayıtlı. Üzerine yazmak ister misiniz? [e/H]: ").strip()
        if overwrite.lower() != 'e':
            print("İptal edildi.")
            return

    recorder.list_devices()
    input("Enter'a basarak kayda başlayın...")

    # Segment kayıtları
    segments = []
    segment_wavs_dir = os.path.join("signatures", user_id + "_segments")
    os.makedirs(segment_wavs_dir, exist_ok=True)

    for i in range(ENROLLMENT_SEGMENTS):
        print(f"\n{'─' * 50}")
        print(f"📝 Segment {i+1}/{ENROLLMENT_SEGMENTS}")
        print(ENROLLMENT_PROMPTS[i] if i < len(ENROLLMENT_PROMPTS) else "Herhangi bir şey konuşun.")
        print(f"\n{SEGMENT_NOTE}")
        print(f"\n   (Süre: {ENROLLMENT_DURATION} saniye)")

        input("   Hazır olduğunuzda Enter'a basın...")
        audio = recorder.record_with_quality_check(ENROLLMENT_DURATION)

        if audio is None:
            print("❌ Kayıt başarısız. İptal edildi.")
            return

        segments.append(audio)
        wav_path = os.path.join(segment_wavs_dir, f"segment_{i+1}.wav")
        recorder.save_wav(audio, wav_path)

    # Embedding çıkar ve kaydet
    print(f"\n{'─' * 50}")
    print("🧠 Ses imzası çıkarılıyor...")

    embedding = encoder.extract_multi_segment_embedding(segments)
    print(f"   ✅ İmza boyutu: {embedding.shape[0]} boyut")
    print(f"   ✅ L2 norm: {float(embedding @ embedding):.4f}")

    metadata = {
        "segment_count": len(segments),
        "segment_duration": ENROLLMENT_DURATION,
        "total_duration": ENROLLMENT_DURATION * len(segments),
        "segments_dir": segment_wavs_dir,
    }

    save_path = store.save(user_id, embedding, metadata)

    print(f"\n💾 Ses imzası kaydedildi: {save_path}")
    print(f"   Kullanıcı    : {user_id}")
    print(f"   Segmentler   : {len(segments)} x {ENROLLMENT_DURATION}s")
    print(f"   İmza boyutu  : {EMBEDDING_DIM} boyut")


# ─── 2. VERIFICATION ─────────────────────────────────────────────────────────
def do_verify(recorder: AudioRecorder, encoder: VoiceEncoder, store: SignatureStore):
    print(f"\n{'=' * 60}")
    print("   🔍 SES DOĞRULAMA TESTİ")
    print(f"{'=' * 60}")

    users = store.list_users()
    if not users:
        print("\n❌ Henüz kayıtlı kullanıcı yok. Önce kayıt yapın.")
        return

    print("\n📋 Kayıtlı kullanıcılar:")
    for i, user in enumerate(users):
        print(f"   [{i+1}] {user['user_id']} (Kayıt: {user.get('created_at', 'N/A')[:10]})")

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

    saved_embedding, profile = store.load(user_id)
    print(f"\n✅ '{user_id}' imzası yüklendi.")

    # Test döngüsü
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


# ─── 3. LIST USERS ────────────────────────────────────────────────────────────
def do_list(store: SignatureStore):
    print(f"\n{'=' * 60}")
    print("   📋 KAYITLI KULLANICILAR")
    print(f"{'=' * 60}")

    users = store.list_users()
    if not users:
        print("\n   Henüz kayıtlı kullanıcı yok.")
        return

    for user in users:
        print(f"\n   👤 {user['user_id']}")
        print(f"      Kayıt tarihi  : {user.get('created_at', 'N/A')[:19]}")
        print(f"      Segment sayısı : {user.get('segment_count', 'N/A')}")
        print(f"      Model          : {user.get('model', 'N/A')}")

    print(f"\n   Toplam: {len(users)} kullanıcı")


# ─── 4. DELETE USER ───────────────────────────────────────────────────────────
def do_delete(store: SignatureStore):
    print(f"\n{'=' * 60}")
    print("   🗑️ KULLANICI SİL")
    print(f"{'=' * 60}")

    users = store.list_users()
    if not users:
        print("\n   Henüz kayıtlı kullanıcı yok.")
        return

    print("\n📋 Kayıtlı kullanıcılar:")
    for i, user in enumerate(users):
        print(f"   [{i+1}] {user['user_id']}")

    choice = input("\n   Silinecek kullanıcı adı veya numarası: ").strip()

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
        print(f"❌ '{user_id}' bulunamadı.")
        return

    confirm = input(f"⚠️  '{user_id}' silinecek. Emin misiniz? [e/H]: ").strip()
    if confirm.lower() == 'e':
        store.delete(user_id)
        print(f"✅ '{user_id}' silindi.")
    else:
        print("İptal edildi.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    banner()

    recorder = AudioRecorder()
    store = SignatureStore()
    encoder = None  # Lazy load — sadece gerekince yükle

    while True:
        choice = menu()

        if choice == '1':
            if encoder is None:
                encoder = VoiceEncoder()
            do_enroll(recorder, encoder, store)

        elif choice == '2':
            if encoder is None:
                encoder = VoiceEncoder()
            do_verify(recorder, encoder, store)

        elif choice == '3':
            do_list(store)

        elif choice == '4':
            do_delete(store)

        elif choice == '0':
            print("\n👋 Güle güle!")
            break

        else:
            print("❌ Geçersiz seçim.")


if __name__ == "__main__":
    main()

"""
Ses İmzası Sistemi - Ana Menü

Model seçimi: ECAPA-TDNN (SpeechBrain) veya ERes2Net (ModelScope)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    fix_torchaudio_compat,
    ENROLLMENT_DURATION,
    ENROLLMENT_SEGMENTS,
    VERIFICATION_DURATION,
    VERIFICATION_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    ERES2NET_THRESHOLD,
    ERES2NET_HIGH_THRESHOLD,
    ERES2NET_ENROLLMENT_SEGMENTS,
    ERES2NET_SEGMENT_DURATION,
    ERES2NET_MIN_SPEECH_DURATION,
    MODEL_ECAPA,
    MODEL_ERES2NET,
)
fix_torchaudio_compat()

from utils.audio_recorder import AudioRecorder
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

    "4️⃣  Haber spikeri veya belgesel seslendirmeni gibi ciddi, otoriter\n"
    "     ve net bir ton kullanın. Kelimelerin üzerine basarak okuyun.\n\n"
    '     "Teknolojinin hızla geliştiği bu yeni çağda, yapay zeka\n'
    '      sistemleri hayatımızın her alanına entegre olmaya devam\n'
    '      ediyor. Veri analizi, ses tanıma ve makine öğrenmesi gibi\n'
    '      alanlarda katedilen mesafe, geleceğin dijital dünyasını\n'
    '      bugünden şekillendiriyor. Karmaşık algoritmalar, insan\n'
    '      sesini en ince ayrıntısına kadar analiz ederek gerçeğe en\n'
    '      yakın sonuçları üretmeyi hedefliyor."\n\n'
    "     Tüyo: Kendinden emin, net bir tonla. Kelimeler arası hafif\n"
    "     duraksamalar yapabilirsiniz.",

    "5️⃣  Düşünceli, derin ve sakin bir ton. Sanki birine çok önemli\n"
    "     bir tavsiye veriyormuşsunuz gibi yavaş okuyun.\n\n"
    '     "Zaman, bazen çok hızlı akıp giden bir nehir, bazen de\n'
    '      durup dinlendiğimiz sakin bir liman gibidir. Önemli olan\n'
    '      bu koşturmacanın içinde kendi sesimizi duyabilmek ve\n'
    '      geride anlamlı izler bırakabilmektir. Her kelime, her\n'
    '      cümle aslında ruhumuzun bir yansımasıdır. Ve işte şimdi,\n'
    '      bu uzun yolculuğun sonuna geliyoruz."\n\n'
    "     Tüyo: Çok yavaş okuyun. Cümle aralarında 1-2 saniye bekleyin\n"
    '     ve nefes alışverişinizin doğal duyulmasına izin verin.',
]


# ─── Yardımcı fonksiyonlar ────────────────────────────────────────────────────
def get_encoder(model_name: str):
    """Seçilen modele göre encoder yükle."""
    if model_name == MODEL_ECAPA:
        from models.voice_encoder import VoiceEncoder
        return VoiceEncoder()
    else:
        from models.eres2net_encoder import ERes2NetEncoder
        return ERes2NetEncoder()


def get_enrollment_params(model_name: str) -> tuple:
    """(n_segments, segment_duration) modelye göre döndür."""
    if model_name == MODEL_ERES2NET:
        return ERES2NET_ENROLLMENT_SEGMENTS, ERES2NET_SEGMENT_DURATION
    return ENROLLMENT_SEGMENTS, ENROLLMENT_DURATION


def get_threshold(model_name: str) -> float:
    return ERES2NET_THRESHOLD if model_name == MODEL_ERES2NET else VERIFICATION_THRESHOLD


def get_high_threshold(model_name: str) -> float:
    return ERES2NET_HIGH_THRESHOLD if model_name == MODEL_ERES2NET else HIGH_CONFIDENCE_THRESHOLD


def interpret_score(encoder, score: float) -> str:
    return encoder.interpret_score(score)


# ─── Model Seçim Menüsü ───────────────────────────────────────────────────────
def select_model() -> str:
    print("\n" + "=" * 60)
    print("   🎤 SES İMZASI SİSTEMİ")
    print("=" * 60)
    print("\n   Model seçin:")
    print("   [1] ECAPA-TDNN  — SpeechBrain, VoxCeleb (192-dim)")
    print("   [2] ERes2Net    — ModelScope, 200k konuşmacı (daha güçlü)")
    print("   [0] Çıkış")

    while True:
        choice = input("\n   Seçiminiz [1/2]: ").strip()
        if choice == '1':
            return MODEL_ECAPA
        elif choice == '2':
            return MODEL_ERES2NET
        elif choice == '0':
            sys.exit(0)
        else:
            print("   ❌ Geçersiz seçim, tekrar deneyin.")


def banner(model_name: str):
    label = "ECAPA-TDNN (SpeechBrain)" if model_name == MODEL_ECAPA else "ERes2Net (ModelScope)"
    thr = get_threshold(model_name)
    print("\n" + "=" * 60)
    print(f"   🎤 SES İMZASI SİSTEMİ")
    print(f"   Model   : {label}")
    print(f"   Eşik    : {thr}")
    print("=" * 60)


def menu(model_name: str) -> str:
    label = "ECAPA-TDNN" if model_name == MODEL_ECAPA else "ERes2Net"
    print(f"\n   [1] 🆕  Yeni ses imzası kaydet  ({label})")
    print(f"   [2] 🔍  Ses doğrulama testi     ({label})")
    print("   [3] 📋  Kayıtlı kullanıcıları listele")
    print("   [4] 🗑️  Kullanıcı sil")
    print("   [5] 🔄  Model değiştir")
    print("   [0] 🚪  Çıkış")
    return input("\n   Seçiminiz: ").strip()


# ─── 1. ENROLLMENT ───────────────────────────────────────────────────────────
def do_enroll(recorder: AudioRecorder, encoder, store: SignatureStore, model_name: str):
    print(f"\n{'=' * 60}")
    print("   🆕 YENİ SES İMZASI KAYDI")
    print(f"{'=' * 60}")

    user_id = input("\n👤 Adınızı girin: ").strip()
    if not user_id:
        print("❌ Geçerli bir isim girmelisiniz.")
        return

    if store.exists(user_id):
        overwrite = input(f"⚠️  '{user_id}' bu model için zaten kayıtlı. Üzerine yazmak ister misiniz? [e/H]: ").strip()
        if overwrite.lower() != 'e':
            print("İptal edildi.")
            return

    recorder.list_devices()

    n_segments, seg_duration = get_enrollment_params(model_name)

    if model_name == MODEL_ERES2NET:
        print(f"\n   ℹ️  ERes2Net modu: {n_segments} segment × {seg_duration}s")
        print(f"   Daha fazla ve daha uzun kayıt → daha güçlü ses imzası")

    input("Enter'a basarak kayda başlayın...")

    segments = []
    segment_wavs_dir = os.path.join("signatures", store.model_name, user_id + "_segments")
    os.makedirs(segment_wavs_dir, exist_ok=True)

    for i in range(n_segments):
        print(f"\n{'─' * 50}")
        print(f"📝 Segment {i+1}/{n_segments}")
        print(ENROLLMENT_PROMPTS[i] if i < len(ENROLLMENT_PROMPTS) else "Herhangi bir şey konuşun.")
        print(f"\n{SEGMENT_NOTE}")
        print(f"\n   (Süre: {seg_duration} saniye)")

        input("   Hazır olduğunuzda Enter'a basın...")
        audio = recorder.record_with_quality_check(seg_duration)

        if audio is None:
            print("❌ Kayıt başarısız. İptal edildi.")
            return

        # Konuşma süresini göster (ERes2Net için önemli)
        if model_name == MODEL_ERES2NET:
            from utils.audio_utils import trim_silence_vad
            _, speech_dur = trim_silence_vad(audio)
            print(f"   🗣️  Net konuşma süresi: {speech_dur:.1f}s", end="")
            if speech_dur < ERES2NET_MIN_SPEECH_DURATION:
                print(f" ⚠️ (minimum {ERES2NET_MIN_SPEECH_DURATION}s önerilen)")
            else:
                print(" ✅")

        segments.append(audio)
        wav_path = os.path.join(segment_wavs_dir, f"segment_{i+1}.wav")
        recorder.save_wav(audio, wav_path)

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
    print(f"   Model        : {store.model_name}")
    print(f"   İmza boyutu  : {embedding.shape[0]} boyut")


# ─── 2. VERIFICATION ─────────────────────────────────────────────────────────
def do_verify(recorder: AudioRecorder, encoder, store: SignatureStore, model_name: str):
    print(f"\n{'=' * 60}")
    print("   🔍 SES DOĞRULAMA TESTİ")
    print(f"{'=' * 60}")

    users = store.list_users()
    if not users:
        print("\n❌ Bu model için kayıtlı kullanıcı yok. Önce kayıt yapın.")
        return

    print("\n📋 Kayıtlı kullanıcılar:")
    for i, user in enumerate(users):
        print(f"   [{i+1}] {user['user_id']} (Kayıt: {user.get('created_at', 'N/A')[:10]})")

    choice = input("\n👤 Doğrulanacak kullanıcı adı veya numarası: ").strip()

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

    saved_embedding, profile = store.load(user_id)
    threshold = get_threshold(model_name)
    high_threshold = get_high_threshold(model_name)
    print(f"\n✅ '{user_id}' imzası yüklendi. (boyut: {saved_embedding.shape[0]})")

    while True:
        print(f"\n{'─' * 50}")
        print(f"🎙️ {VERIFICATION_DURATION} saniyelik test kaydı alınacak.")
        input("   Hazır olduğunuzda Enter'a basın...")

        test_audio = recorder.record_with_quality_check(VERIFICATION_DURATION, max_retries=2)
        if test_audio is None:
            print("❌ Kayıt başarısız.")
            continue

        test_embedding = encoder.extract_embedding(test_audio)
        score = encoder.cosine_similarity(saved_embedding, test_embedding)
        interpretation = encoder.interpret_score(score)

        print(f"\n{'─' * 50}")
        print(f"📊 SONUÇ")
        print(f"   Karşılaştırılan : {user_id}")
        print(f"   Benzerlik skoru : {score:.4f}")
        print(f"   Yorum           : {interpretation}")
        print(f"   Eşik değeri     : {threshold}")

        if score >= threshold:
            print(f"\n   ✅ DOĞRULANDI — Bu ses '{user_id}' kişisine ait!")
            if score >= high_threshold:
                print(f"   🌟 Yüksek güvenle eşleşme!")
        else:
            print(f"\n   ❌ DOĞRULANAMADI — Bu ses '{user_id}' kişisine ait değil.")

        again = input("\n   Tekrar test etmek ister misiniz? [E/h]: ").strip()
        if again.lower() == 'h':
            break


# ─── 3. LIST USERS ────────────────────────────────────────────────────────────
def do_list(store: SignatureStore):
    print(f"\n{'=' * 60}")
    print(f"   📋 KAYITLI KULLANICILAR ({store.model_name})")
    print(f"{'=' * 60}")

    users = store.list_users()
    if not users:
        print("\n   Bu model için henüz kayıtlı kullanıcı yok.")
        return

    for user in users:
        print(f"\n   👤 {user['user_id']}")
        print(f"      Kayıt tarihi   : {user.get('created_at', 'N/A')[:19]}")
        print(f"      Segment sayısı : {user.get('segment_count', 'N/A')}")
        print(f"      Embedding dim  : {user.get('embedding_dim', 'N/A')}")

    print(f"\n   Toplam: {len(users)} kullanıcı")


# ─── 4. DELETE USER ───────────────────────────────────────────────────────────
def do_delete(store: SignatureStore):
    print(f"\n{'=' * 60}")
    print(f"   🗑️ KULLANICI SİL ({store.model_name})")
    print(f"{'=' * 60}")

    users = store.list_users()
    if not users:
        print("\n   Bu model için henüz kayıtlı kullanıcı yok.")
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
    recorder = AudioRecorder()

    model_name = select_model()
    encoder = None

    while True:
        banner(model_name)
        store = SignatureStore(model_name=model_name)
        choice = menu(model_name)

        if choice == '1':
            if encoder is None:
                encoder = get_encoder(model_name)
            do_enroll(recorder, encoder, store, model_name)

        elif choice == '2':
            if encoder is None:
                encoder = get_encoder(model_name)
            do_verify(recorder, encoder, store, model_name)

        elif choice == '3':
            do_list(store)

        elif choice == '4':
            do_delete(store)

        elif choice == '5':
            model_name = select_model()
            encoder = None  # Yeni model için sıfırla

        elif choice == '0':
            print("\n👋 Güle güle!")
            break

        else:
            print("❌ Geçersiz seçim.")


if __name__ == "__main__":
    main()

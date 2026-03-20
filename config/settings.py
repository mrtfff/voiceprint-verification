"""
Ses İmzası Sistemi - Merkezi Ayarlar
"""
import os

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Ses Kaydı Ayarları ───────────────────────────────────────────────────────
SAMPLE_RATE = 16000            # Hz - ECAPA-TDNN'in beklediği format
CHANNELS = 1                   # Mono
ENROLLMENT_DURATION = 10       # Her segment için kayıt süresi (saniye)
ENROLLMENT_SEGMENTS = 3        # Kayıt için segment sayısı
VERIFICATION_DURATION = 5      # Doğrulama için kayıt süresi (saniye)

# ─── Model Ayarları ───────────────────────────────────────────────────────────
EMBEDDING_DIM = 192            # ECAPA-TDNN çıktı boyutu
PRETRAINED_MODEL_DIR = os.path.join(PROJECT_ROOT, "pretrained_models", "spkrec-ecapa-voxceleb")

# ─── İmza Saklama Ayarları ────────────────────────────────────────────────────
SIGNATURES_DIR = os.path.join(PROJECT_ROOT, "signatures")

# ─── Doğrulama Eşikleri ──────────────────────────────────────────────────────
# Cosine similarity tabanlı eşikler (0-1 arası, 1 = tam eşleşme)
VERIFICATION_THRESHOLD = 0.55  # Bu değerin üstü = aynı kişi
HIGH_CONFIDENCE_THRESHOLD = 0.70  # Bu değerin üstü = yüksek güvenle aynı kişi

# ─── Ses Kalitesi Ayarları ────────────────────────────────────────────────────
MIN_SNR_DB = 10                # Minimum sinyal-gürültü oranı (dB)
MIN_AUDIO_LEVEL = 0.01        # Minimum ses seviyesi (RMS)
MAX_CLIPPING_RATIO = 0.01     # Maksimum clipping oranı

# ─── SpeechBrain Uyumluluk ────────────────────────────────────────────────────
def fix_torchaudio_compat():
    """SpeechBrain 1.0.3 / torchaudio 2.10+ uyumluluk düzeltmesi."""
    import torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['soundfile']

# ─── Model Sabitleri ──────────────────────────────────────────────────────────
MODEL_ECAPA = "ecapa-tdnn"
MODEL_ERES2NET = "eres2net"
MODEL_W2VBERT = "w2v-bert-2.0"

ERES2NET_MODEL_ID = "iic/speech_eres2net_sv_zh-cn_16k-common"
ERES2NET_MODEL_REVISION = "v1.0.5"

# Eşik değerleri: two-stage (enroll→save→verify) kullanım için kalibre edilmiş
ERES2NET_THRESHOLD = 0.55         # pipeline'ın 0.365'i two-stage içi kullanım için uygun değil
ERES2NET_HIGH_THRESHOLD = 0.70

# w2v-BERT 2.0 değerleri
W2V_BERT_THRESHOLD = 0.65
W2V_BERT_HIGH_THRESHOLD = 0.80

# ERes2Net / W2V-BERT için enrollment parametreleri
ERES2NET_ENROLLMENT_SEGMENTS = 5  # 3 → 5
ERES2NET_SEGMENT_DURATION = 15    # 10s → 15s
ERES2NET_MIN_SPEECH_DURATION = 3.0  # Minimum net konuşma süresi (saniye)


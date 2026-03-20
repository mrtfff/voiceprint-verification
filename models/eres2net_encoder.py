"""
ModelScope ERes2Net tabanlı ses imzası çıkarma modülü.
ECAPA-TDNN ile aynı arayüzü sağlar.
"""
import os
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    SAMPLE_RATE,
    ERES2NET_MODEL_ID, ERES2NET_MODEL_REVISION,
    ERES2NET_THRESHOLD, ERES2NET_HIGH_THRESHOLD,
)


class ERes2NetEncoder:
    """
    ModelScope ERes2Net modeli ile ses imzası çıkarma.
    Singleton pattern — model bir kez yüklenir.

    Threshold notu: modelscope pipeline'ın varsayılan 0.365 eşiği, iki dosyayı
    doğrudan karşılaştıran tek aşamalı kullanım içindir. Bizim iki aşamalı
    sistemimizde (enroll → kaydet → verify → karşılaştır) 0.55 daha doğru sonuç verir.
    """

    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ERes2NetEncoder._pipeline is None:
            self._load_model()

    def _load_model(self):
        print("🔄 ERes2Net modeli yükleniyor (ModelScope)...")
        print(f"   Model: {ERES2NET_MODEL_ID} @ {ERES2NET_MODEL_REVISION}")

        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        ERes2NetEncoder._pipeline = pipeline(
            task=Tasks.speaker_verification,
            model=ERES2NET_MODEL_ID,
            model_revision=ERES2NET_MODEL_REVISION,
        )
        print("✅ ERes2Net modeli başarıyla yüklendi!")

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Ses verisinden embedding çıkar.

        Args:
            audio: float32 numpy array, [-1, 1] aralığında, 16kHz mono

        Returns:
            L2 normalize edilmiş embedding vektörü
        """
        import tempfile
        import soundfile as sf

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # ERes2Net pipeline dosya yolu bekliyor — geçici WAV yaz
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            audio_int16 = (audio * 32767).astype(np.int16)
            sf.write(tmp_path, audio_int16, SAMPLE_RATE, subtype='PCM_16')

            result = ERes2NetEncoder._pipeline(
                [tmp_path, tmp_path],
                output_emb=True
            )
            embedding = np.array(result['embs'][0]).flatten().astype(np.float32)
        finally:
            os.unlink(tmp_path)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def extract_embedding_from_file(self, wav_path: str) -> np.ndarray:
        """WAV dosyasından embedding çıkar."""
        from utils.audio_utils import load_wav, normalize_audio, resample, trim_silence_vad

        sr, audio = load_wav(wav_path)
        if sr != SAMPLE_RATE:
            print(f"   ⚠️ Resample: {sr}Hz → {SAMPLE_RATE}Hz")
            from utils.audio_utils import resample
            audio = resample(audio, sr, SAMPLE_RATE)

        audio = normalize_audio(trim_silence_vad(audio))
        return self.extract_embedding(audio)

    def extract_multi_segment_embedding(self, segments: list) -> np.ndarray:
        """
        Çok segmentli kalite ağırlıklı ortalama embedding.
        Daha yüksek RMS (daha güçlü ses) olan segmentlere daha fazla ağırlık verilir.
        """
        from utils.audio_utils import trim_silence_vad, normalize_audio

        embeddings = []
        weights = []

        for i, segment in enumerate(segments):
            print(f"   📊 Segment {i+1}/{len(segments)} işleniyor...")

            # VAD ile sessiz bölümleri kırp
            active, _ = trim_silence_vad(segment)
            active = normalize_audio(active)

            # RMS bazlı ağırlık: daha güçlü ve net ses → daha yüksek ağırlık
            rms = float(np.sqrt(np.mean(active ** 2)))
            weights.append(max(rms, 1e-6))

            emb = self.extract_embedding(active)
            embeddings.append(emb)
            print(f"      RMS: {rms:.4f} → ağırlık: {rms:.4f}")

        # Ağırlıklı ortalama
        weights = np.array(weights)
        weights = weights / weights.sum()  # normalize weights

        avg = np.zeros_like(embeddings[0])
        for emb, w in zip(embeddings, weights):
            avg += emb * w

        # L2 normalize
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm

        return avg

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        dot = np.dot(emb1, emb2)
        n1 = np.linalg.norm(emb1)
        n2 = np.linalg.norm(emb2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(dot / (n1 * n2))

    @staticmethod
    def interpret_score(score: float) -> str:
        """
        ERes2Net iki aşamalı (enroll→verify) kullanım için skor yorumu.
        Threshold: 0.55
        """
        if score >= ERES2NET_HIGH_THRESHOLD:
            return "✅ Kesinlikle aynı kişi"
        elif score >= ERES2NET_THRESHOLD:
            return "✅ Muhtemelen aynı kişi"
        elif score >= 0.40:
            return "⚠️ Belirsiz"
        elif score >= 0.20:
            return "❌ Muhtemelen farklı kişi"
        else:
            return "❌ Kesinlikle farklı kişi"

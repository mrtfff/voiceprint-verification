"""
ECAPA-TDNN tabanlı ses imzası (embedding) çıkarma modülü.
SpeechBrain pretrained modelini kullanır.
"""
import os
import numpy as np
import torch

# SpeechBrain uyumluluk düzeltmesi (import'tan önce çağrılmalı)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import fix_torchaudio_compat, PRETRAINED_MODEL_DIR, EMBEDDING_DIM, SAMPLE_RATE
fix_torchaudio_compat()

from speechbrain.inference.speaker import SpeakerRecognition


class VoiceEncoder:
    """
    ECAPA-TDNN modeli ile ses imzası çıkarma.
    
    Singleton pattern: Model bir kez yüklenir ve tekrar kullanılır.
    Bu, her çağrıda 5-10 saniyelik model yükleme süresinden kaçınmayı sağlar.
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if VoiceEncoder._model is None:
            self._load_model()
    
    def _load_model(self):
        """Pretrained ECAPA-TDNN modelini yükle."""
        print("🔄 ECAPA-TDNN modeli yükleniyor...")
        
        if not os.path.exists(PRETRAINED_MODEL_DIR):
            raise FileNotFoundError(
                f"Pretrained model bulunamadı: {PRETRAINED_MODEL_DIR}\n"
                f"Model dosyalarını indirmeniz gerekiyor."
            )
        
        VoiceEncoder._model = SpeakerRecognition.from_hparams(
            source=PRETRAINED_MODEL_DIR,
            savedir=PRETRAINED_MODEL_DIR,
        )
        
        print("✅ Model başarıyla yüklendi!")
    
    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Ses verisinden 192 boyutlu embedding çıkar.
        
        Args:
            audio: float32 numpy array, [-1, 1] aralığında, 16kHz mono
            
        Returns:
            192 boyutlu numpy embedding vektörü
        """
        # numpy -> torch tensor
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        waveform = torch.tensor(audio).unsqueeze(0)  # [1, T]
        
        # Embedding çıkar
        with torch.no_grad():
            embedding = VoiceEncoder._model.encode_batch(waveform)
        
        # torch tensor -> numpy
        embedding = embedding.squeeze().cpu().numpy()
        
        assert embedding.shape == (EMBEDDING_DIM,), \
            f"Beklenen embedding boyutu {EMBEDDING_DIM}, alınan: {embedding.shape}"
        
        return embedding
    
    def extract_embedding_from_file(self, wav_path: str) -> np.ndarray:
        """
        WAV dosyasından embedding çıkar.
        
        Args:
            wav_path: WAV dosya yolu
            
        Returns:
            192 boyutlu numpy embedding vektörü
        """
        from utils.audio_utils import load_wav, normalize_audio, resample, trim_silence
        
        sr, audio = load_wav(wav_path)
        
        # Gerekirse resample
        if sr != SAMPLE_RATE:
            print(f"   ⚠️ Resample ediliyor: {sr}Hz → {SAMPLE_RATE}Hz")
            audio = resample(audio, sr, SAMPLE_RATE)
        
        # Ön-işleme
        audio = normalize_audio(trim_silence(audio))
        
        return self.extract_embedding(audio)
    
    def extract_multi_segment_embedding(self, segments: list[np.ndarray]) -> np.ndarray:
        """
        Birden fazla ses segmentinden ortalama embedding çıkar.
        Bu yöntem, tek bir segmente göre daha güçlü ve güvenilir bir imza üretir.
        
        Args:
            segments: float32 numpy array listesi
            
        Returns:
            192 boyutlu ortalama embedding vektörü (L2 normalize edilmiş)
        """
        embeddings = []
        
        for i, segment in enumerate(segments):
            print(f"   📊 Segment {i+1}/{len(segments)} işleniyor...")
            emb = self.extract_embedding(segment)
            embeddings.append(emb)
        
        # Ortalama embedding
        avg_embedding = np.mean(embeddings, axis=0)
        
        # L2 normalization (cosine similarity için)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        return avg_embedding
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        İki embedding arasındaki cosine similarity hesapla.
        
        Returns:
            float: -1 ile 1 arası benzerlik skoru (1 = tam eşleşme)
        """
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    @staticmethod
    def interpret_score(score: float) -> str:
        """
        Benzerlik skorunu yorumla.
        ECAPA-TDNN cosine similarity için gerçekçi aralıklar:
        - Aynı kişi tipik: 0.45 - 0.80+
        - Farklı kişi tipik: -0.10 - 0.30
        """
        if score >= 0.65:
            return "✅ Kesinlikle aynı kişi"
        elif score >= 0.45:
            return "✅ Muhtemelen aynı kişi"
        elif score >= 0.30:
            return "⚠️ Belirsiz - eşik değerine bağlı"
        elif score >= 0.15:
            return "❌ Muhtemelen farklı kişi"
        else:
            return "❌ Kesinlikle farklı kişi"

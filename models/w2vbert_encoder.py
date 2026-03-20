"""
Facebook W2v-BERT 2.0 tabanlı, Speaker Verification (SV) için özel adapte edilmiş
ve MFA (Multi-layer Feature Aggregation) ile LoRA (Low-Rank Adaptation) içeren encoder modülü.

Referans: "Enhancing Speaker Verification with w2v-BERT 2.0" (arXiv: 2510.04213)
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    SAMPLE_RATE,
    W2V_BERT_THRESHOLD,
    W2V_BERT_HIGH_THRESHOLD
)

# Ağırlıkların bulunacağı klasör
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "model_lmft_0.14.pth")


class ASP(nn.Module):
    """
    Attentive Statistics Pooling (ASP) katmanı.
    Zaman eksenindeki özellikleri dikkat mekanizmasıyla ağırlıklandırarak ortalama 
    ve standart sapmalarını hesaplar.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(ASP, self).__init__()
        self.expansion = 2
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor):
        # x shape: [Batch, Time, Dim]
        w = self.attention(x.transpose(1, 2)).transpose(1, 2)  # [B, T, D]
        mu = torch.sum(x * w, dim=1) # [B, D]
        sg = torch.sqrt((torch.sum((x**2) * w, dim=1) - mu**2).clamp(min=1e-5)) # [B, D]
        return torch.cat([mu, sg], dim=1) # [B, 2D]


class W2vBertSVModel(nn.Module):
    """
    w2v-BERT 2.0 + MFA + LoRA + ASP SV Mimari Uyarlaması.
    """
    def __init__(self, adapter_dim=128, embd_dim=256):
        super(W2vBertSVModel, self).__init__()
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

        print("   -> W2v-BERT 2.0 altyapısı kuruluyor (Base Model)...")
        # Önceden eğitilmiş model konfigurasyonunu al (weights download olmadan sadece skeleton için)
        config = Wav2Vec2BertConfig.from_pretrained('facebook/w2v-bert-2.0')
        self.encoder = Wav2Vec2BertModel(config)
        
        # Orijinal repodaki gibi mikroskopik hataları önlemek için parametre silinir
        if hasattr(self.encoder, 'masked_spec_embed'):
            delattr(self.encoder, 'masked_spec_embed')
            
        self.d_model = self.encoder.config.hidden_size # genelde 1024
        self.n_mfa_layers = self.encoder.config.num_hidden_layers + 1 # 24 + 1 = 25
        
        # Layer Adapter (Çok Katmanlı Özellik Çıkarımı için)
        print(f"   -> Multi-layer Feature Aggregation ({self.n_mfa_layers} katman) hazırlanıyor...")
        self.adapter_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.ReLU(True),
                nn.Linear(adapter_dim, adapter_dim),
            ) for _ in range(self.n_mfa_layers)
        ])
        
        # ASP ve Bottleneck
        feat_dim = adapter_dim * self.n_mfa_layers
        self.pooling = ASP(feat_dim, adapter_dim)
        self.bottleneck = nn.Linear(feat_dim * self.pooling.expansion, embd_dim)

    def forward(self, input_features, attention_mask=None):
        # W2V-BERT projeksiyonu
        x = self.encoder.feature_projection(input_features)[0]
        
        # Conformer Katmanları
        hidden_states = [x]
        for layer in self.encoder.encoder.layers:
            x = layer(x)[0]
            hidden_states.append(x)
            
        # MFA + Layer Adapter
        layer_outputs = []
        x_hidden = hidden_states[-self.n_mfa_layers:]
        for i in range(self.n_mfa_layers):
            layer_outputs.append(self.adapter_layers[i](x_hidden[i]))
            
        x_cat = torch.cat(layer_outputs, dim=-1)
        
        # Havuzlama (Pooling)
        x_pool = self.pooling(x_cat)
        
        # Çıktı Embedding (Genelde 256 boyut)
        x_out = self.bottleneck(x_pool)
        return x_out


class W2vBertEncoder:
    """
    W2V-BERT Singleton Sınıfı. ModelScope ve SpeechBrain ile aynı arayüze sahiptir.
    """
    _instance = None
    _model = None
    _feature_extractor = None
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if W2vBertEncoder._model is None:
            self._load_model()

    def _load_model(self):
        print("🔄 W2v-BERT 2.0 SV modeli yükleniyor... Bu işlem parçaları birleştirir, biraz sürebilir.")
        from transformers import AutoFeatureExtractor

        if not os.path.exists(WEIGHTS_PATH):
            print(f"\n❌ Kritik Hata: {WEIGHTS_PATH} bulunamadı!")
            print("Lütfen ilk önce 'model_lmft_0.14.pth' dosyasının başarıyla indirildiğinden emin olun.")
            sys.exit(1)

        # Özellik çıkarıcı yükleniyor (huggingface cache'den veya otomatik inet'den)
        W2vBertEncoder._feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')

        # Model mimarisini oluştur
        model = W2vBertSVModel(adapter_dim=128, embd_dim=256)
        
        # SV ağırlıklarını Load et (LoRA ve Adapterler dahil)
        print("   -> Önceden eğitilmiş SV (Speaker Verification) ağırlıkları belleğe alınıyor...")
        ckpt_data = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)
        
        # dict key uyarlaması (Github repo: "modules['spk_model']" -> front.encoder... -> self.encoder...)
        ckpt_state_dict = ckpt_data['modules']['spk_model']
        adapted_state_dict = {}
        for k, v in ckpt_state_dict.items():
            if k.startswith('front.encoder.'):
                new_k = k.replace('front.encoder.', 'encoder.')
            elif k.startswith('front.'):
                # Olası diğer front öğeleri atlanır ya da uyarlanır, şu anki yapı için gerekmez
                continue
            else:
                new_k = k
            adapted_state_dict[new_k] = v

        # Eşleşenleri atama
        curr_state_dict = model.state_dict()
        mismatched = []
        for k in curr_state_dict.keys():
            if k in adapted_state_dict and curr_state_dict[k].shape == adapted_state_dict[k].shape:
                curr_state_dict[k] = adapted_state_dict[k]
            else:
                mismatched.append(k)

        if mismatched:
            print(f"   ⚠️ Uyarı: Bazı parametreler eşleşmedi: {len(mismatched)} adet.")
        else:
            print("   ✅ Tüm SV modeli parametreleri kusursuz eşleşti.")

        model.load_state_dict(curr_state_dict)
        model.to(self._device)
        model.eval()
        
        W2vBertEncoder._model = model
        print("✅ w2v-BERT 2.0 (SV Adaptation) modeli başarıyla hazırlandı!")

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        features = W2vBertEncoder._feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )

        input_features = features.input_features.to(self._device)

        with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16 if self._device.type == "cuda" else torch.float32):
            with torch.no_grad():
                emb = W2vBertEncoder._model(input_features).float().detach().cpu().numpy()

        emb = emb.flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb

    def extract_embedding_from_file(self, wav_path: str) -> np.ndarray:
        from utils.audio_utils import load_wav, normalize_audio, resample, trim_silence_vad
        sr, audio = load_wav(wav_path)
        if sr != SAMPLE_RATE:
            audio = resample(audio, sr, SAMPLE_RATE)

        audio, _ = trim_silence_vad(audio)
        audio = normalize_audio(audio)
        return self.extract_embedding(audio)

    def extract_multi_segment_embedding(self, segments: list) -> np.ndarray:
        from utils.audio_utils import trim_silence_vad, normalize_audio
        embeddings = []
        weights = []

        for i, segment in enumerate(segments):
            print(f"   📊 Segment {i+1}/{len(segments)} işleniyor...")
            active, _ = trim_silence_vad(segment)
            active = normalize_audio(active)
            rms = float(np.sqrt(np.mean(active ** 2)))
            weights.append(max(rms, 1e-6))
            emb = self.extract_embedding(active)
            embeddings.append(emb)

        weights = np.array(weights)
        weights = weights / weights.sum()

        avg = np.zeros_like(embeddings[0])
        for emb, w in zip(embeddings, weights):
            avg += emb * w

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
        """w2v-BERT 2.0 (SV) aşırı hassastır. Eşik değerleri özeldir."""
        if score >= W2V_BERT_HIGH_THRESHOLD:
            return "✅ Kesinlikle aynı kişi (Yüksek Güven)"
        elif score >= W2V_BERT_THRESHOLD:
            return "✅ Muhtemelen aynı kişi"
        elif score >= 0.50:
            return "⚠️ Belirsiz"
        elif score >= 0.30:
            return "❌ Muhtemelen farklı kişi"
        else:
            return "❌ Kesinlikle farklı kişi"


"""
Ses imzası saklama ve yükleme modülü.
Model bazında ayrı dizinlerde saklar: signatures/{model_name}/{user_id}/
"""
import os
import json
import shutil
import numpy as np
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import SIGNATURES_DIR


class SignatureStore:
    """Ses imzalarını model bazında ayrı dizinlerde yönetir."""

    def __init__(self, base_dir: str = SIGNATURES_DIR, model_name: str = "ecapa-tdnn"):
        self.base_dir = os.path.join(base_dir, model_name)
        self.model_name = model_name
        os.makedirs(self.base_dir, exist_ok=True)

    def _user_dir(self, user_id: str) -> str:
        safe_id = "".join(c if c.isalnum() or c == '_' else '_' for c in user_id)
        return os.path.join(self.base_dir, safe_id)

    def save(self, user_id: str, embedding: np.ndarray,
             metadata: dict | None = None) -> str:
        user_dir = self._user_dir(user_id)
        os.makedirs(user_dir, exist_ok=True)

        np.save(os.path.join(user_dir, "embedding.npy"), embedding)

        profile = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "embedding_dim": int(embedding.shape[0]),
            "model": self.model_name,
            "embedding_file": "embedding.npy",
        }
        if metadata:
            profile.update(metadata)

        with open(os.path.join(user_dir, "profile.json"), 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        return user_dir

    def load(self, user_id: str) -> tuple[np.ndarray, dict]:
        user_dir = self._user_dir(user_id)
        emb_path = os.path.join(user_dir, "embedding.npy")

        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"'{user_id}' için imza bulunamadı: {emb_path}")

        embedding = np.load(emb_path)

        profile_path = os.path.join(user_dir, "profile.json")
        profile = {}
        if os.path.exists(profile_path):
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)

        return embedding, profile

    def exists(self, user_id: str) -> bool:
        return os.path.exists(os.path.join(self._user_dir(user_id), "embedding.npy"))

    def list_users(self) -> list[dict]:
        users = []
        if not os.path.exists(self.base_dir):
            return users

        for dirname in os.listdir(self.base_dir):
            user_dir = os.path.join(self.base_dir, dirname)
            if not os.path.isdir(user_dir):
                continue
            profile_path = os.path.join(user_dir, "profile.json")
            if os.path.exists(profile_path):
                with open(profile_path, 'r', encoding='utf-8') as f:
                    users.append(json.load(f))
            elif os.path.exists(os.path.join(user_dir, "embedding.npy")):
                users.append({"user_id": dirname, "created_at": "N/A"})

        return users

    def delete(self, user_id: str) -> bool:
        user_dir = self._user_dir(user_id)
        if not os.path.exists(user_dir):
            return False
        shutil.rmtree(user_dir)
        return True

# ECAPA-TDNN Voiceprint Verification

Speaker verification / voiceprint enrollment and testing using **ECAPA-TDNN** (SpeechBrain) with a simple microphone-based CLI.

## What it does

1. **Enrollment**: records multiple speech segments from your microphone, extracts embeddings, and stores an averaged voiceprint.
2. **Verification**: records a short test sample and compares its embedding to the saved voiceprint via cosine similarity.
3. **Storage**: saves per-user embeddings and metadata under `signatures/` (by default).

## Key features

- Uses **SpeechBrain ECAPA-TDNN** embeddings (192-dim)
- Includes microphone recording + simple quality checks (SNR/RMS/clipping)
- Stores signatures as `embedding.npy` + `profile.json`
- Ignores large/local assets (models, pretrained model cache, and user recordings)

## Prerequisites

- Python 3.9+ recommended
- Microphone access (Windows privacy settings)
- Optional but recommended: run on CPU-only first (it works, and keeps setup simpler)

## Install dependencies

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pretrained model

`models/voice_encoder.py` expects the SpeechBrain pretrained files to exist in:

`pretrained_models/spkrec-ecapa-voxceleb`

If you don’t have them yet, download them once with:

```powershell
python -c "from speechbrain.inference.speaker import SpeakerRecognition; SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='pretrained_models/spkrec-ecapa-voxceleb')"
```

## Run

### Full menu (recommended)

```powershell
python main.py
```

### Enrollment only

```powershell
python enroll.py
```

### Verification only

```powershell
python verify.py
```

### Quick manual recording test

```powershell
python record.py
```

## Where data is stored

- `signatures/`: saved voiceprints (embeddings + metadata)
- `pretrained_models/`: cached SpeechBrain model files (ignored by Git)

This repo also ignores user recording folders:

- `seslerim/`
- `ses imzam/`

## Thresholds

The default cosine similarity thresholds are set in `config/settings.py`:

- `VERIFICATION_THRESHOLD`
- `HIGH_CONFIDENCE_THRESHOLD`

Depending on your microphone and room noise, you may want to calibrate them.

## Troubleshooting

See `troubleshooting.md` for common SpeechBrain / torchaudio issues and recording quality tips.


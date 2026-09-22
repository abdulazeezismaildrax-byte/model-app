# ============================================================
# VocalForge — Backend (Modal + FastAPI + Qwen3-TTS 1.7B Base)
# Deploy: modal deploy modal_app.py
#
# GPU:          Nvidia L4 (24GB VRAM, native FlashAttention-2)
# Concurrency:  4 users per L4, max 5 L4s in the workspace
# Caching:      Base model weights baked into the image at build time
#               Generated WAVs in a persistent Volume
#
# Model:
#   - Base → voice cloning (user-uploaded reference audio)
# ============================================================

import modal

# ---------- Modal App & Volumes ----------
app = modal.App("vocal-forge")

data_vol = modal.Volume.from_name("vocal-forge-data", create_if_missing=True)

DATA_DIR   = "/data"
OUTPUT_DIR = f"{DATA_DIR}/outputs"

# ---------- Base Image ----------
base_image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime")
    .apt_install("ffmpeg", "sox", "libsox-dev", "rubberband-cli")
    .pip_install(
        "qwen-tts",
        "fastapi",
        "python-multipart",
        "soundfile",
        "sox",
        "transformers",
        "uvicorn[standard]",
        "aiofiles",
        "huggingface_hub",
        "pyrubberband",   # ← for speed control (time-stretching)
    )
)

# ---------- One-time model download (runs at image build) ----------
def _download_model():
    import os
    from huggingface_hub import snapshot_download
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    repo = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    print(f"Pre-downloading {repo} ...")
    snapshot_download(repo_id=repo, cache_dir=cache_dir)
    print("✅ Base model cached at image build time.")

image = base_image.run_function(_download_model)


# ============================================================
# Modal class-based app (required for concurrency controls)
# ============================================================
@app.cls(
    image=image,
    gpu="L4",
    cpu=2.0,
    memory=8192,
    volumes={
        DATA_DIR: data_vol,
    },
    timeout=1800,
    max_containers=5,
    scaledown_window=60,          # ← shut down after 1 minute idle
)
@modal.concurrent(max_inputs=4)
class VocalForge:
    @modal.enter()
    def load_model(self):
        import os, torch
        from qwen_tts import Qwen3TTSModel

        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("Loading Qwen3-TTS 1.7B Base (voice cloning) …")
        self.base_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("✅ Base model ready.")

    # ------------- HTTP app -------------
    @modal.asgi_app()
    def web(self):
        import os, io, re, uuid
        import numpy as np
        import soundfile as sf
        import pyrubberband as pyrb
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException
        from fastapi.responses import FileResponse
        from fastapi.middleware.cors import CORSMiddleware

        fapp = FastAPI()
        fapp.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @fapp.get("/")
        def root():
            return {"status": "ok", "service": "VocalForge", "mode": "voice-clone"}

        # -------- Voice cloning (Base model) --------
        @fapp.post("/api/clone")
    def clone(  # ← no async
        ref_audio: UploadFile = File(...),
        ref_text: str = Form(""),
        target_text: str = Form(...),
        speed: float = Form(1.0),
    ):
        import os, io, uuid
        import numpy as np
        import soundfile as sf
        import pyrubberband as pyrb
    
        target_text = target_text.strip()
        if not target_text:
            raise HTTPException(400, "Target text cannot be empty")
        if len(target_text) > 500:
            raise HTTPException(400, "Target text exceeds 500 characters")
    
        speed = max(0.5, min(2.0, float(speed)))
    
        # Sync read from the uploaded file object
        data = ref_audio.file.read()
        try:
            arr, sr = sf.read(io.BytesIO(data))
        except Exception as e:
            raise HTTPException(400, f"Could not read reference audio: {e}")
    
        ref_path = f"/tmp/ref_{uuid.uuid4().hex}.wav"
        sf.write(ref_path, arr, sr)
    
        try:
            wavs, sr = self.base_model.generate_voice_clone(
                text=target_text,
                language="English",
                ref_audio=ref_path,
                ref_text=ref_text.strip() if ref_text and ref_text.strip() else None,
            )
        except Exception as e:
            raise HTTPException(500, f"Cloning failed: {e}")
        finally:
            try:
                os.remove(ref_path)
            except OSError:
                pass
    
        audio = np.asarray(wavs[0]).squeeze()
        if abs(speed - 1.0) > 0.01:
            audio = pyrb.time_stretch(audio, sr, speed)
    
        fname = f"{uuid.uuid4().hex}.wav"
        out_path = os.path.join(OUTPUT_DIR, fname)
        sf.write(out_path, audio, sr)
        data_vol.commit()  # now safe — runs in the thread pool
    
        return {
            "audio_url": f"/api/audio/{fname}",
            "speed": speed,
        }

        # -------- Serve generated audio --------
        @fapp.get("/api/audio/{fname}")
        def get_audio(fname: str):
            if not re.match(r"^[a-f0-9]+\.wav$", fname):
                raise HTTPException(400, "Invalid filename")
            p = os.path.join(OUTPUT_DIR, fname)
            if not os.path.exists(p):
                raise HTTPException(404, "Audio not found")
            return FileResponse(p, media_type="audio/wav")

        return fapp

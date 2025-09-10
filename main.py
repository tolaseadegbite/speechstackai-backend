import os
import re
import uuid
from typing import List, Dict

import modal
import boto3
import numpy as np
import soundfile as sf
from fastapi import HTTPException
from pydantic import BaseModel

# --- Configuration ---
MODEL_DIR = "/models"
HF_REPO_YV = "tolasekelvin/YV"
HF_REPO_AF_LANG = "tolasekelvin/Af_lang"
S3_AUDIO_PREFIX = "african-tts-audio"

# --- Modal App Setup ---
app = modal.App("multi-language-tts-api")

# Image definition
image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install_from_requirements("requirements.txt")
    .apt_install("libsndfile1")
    # [FIXED] Added --revision flag to bust the cache and download the latest version of the repo.
    .run_commands(
        f"huggingface-cli download {HF_REPO_YV} --revision 1b1eef1 --local-dir {MODEL_DIR}/YV --local-dir-use-symlinks False",
        f"huggingface-cli download {HF_REPO_AF_LANG} --local-dir {MODEL_DIR}/Af_lang --local-dir-use-symlinks False"
    )
)

# Secret should contain: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME
speechstackai_secret = modal.Secret.from_name("speechstackai-secret")


# --- Pydantic Models for API ---
class GenerateSpeechRequest(BaseModel):
    text: str
    voice: str

class GenerateSpeechResponse(BaseModel):
    audio_url: str
    s3_key: str

# --- Helper Function for Text Chunking (Unchanged) ---
def text_chunker(text: str, max_chunk_size: int = 500):
    if len(text) <= max_chunk_size: return [text]
    chunks, current_pos, text_len = [], 0, len(text)
    while current_pos < text_len:
        chunk_end = min(current_pos + max_chunk_size, text_len)
        if chunk_end == text_len:
            chunks.append(text[current_pos:]); break
        search_text = text[current_pos:chunk_end]
        sentence_ends = [m.end() for m in re.finditer(r'[.!?]+', search_text)]
        if sentence_ends:
            last_sentence_end = sentence_ends[-1]
            chunks.append(text[current_pos:current_pos + last_sentence_end])
            current_pos += last_sentence_end
        else:
            last_space = search_text.rfind(' ')
            if last_space > 0:
                chunks.append(text[current_pos:current_pos + last_space])
                current_pos += last_space + 1
            else:
                chunks.append(text[current_pos:chunk_end])
                current_pos = chunk_end
        while current_pos < text_len and text[current_pos].isspace():
            current_pos += 1
    return [chunk for chunk in chunks if chunk]


@app.cls(
    image=image,
    secrets=[speechstackai_secret],
    scaledown_window=15,
    gpu="L4"
)
class TTSPlatform:
    # Voice configuration is correct
    VOICE_CONFIG = {
        
        "Abọ́sẹ̀dé": {"path": "YV/female_yoruba_tts", "checkpoint": "checkpoint_500000.pth.tar"},
        "Adéṣínà":  {"path": "YV/male_yoruba_tts",   "checkpoint": "checkpoint_500000.pth.tar"},
        
        "Ọmọ́wùnmí": {"path": "YV/bible_to_female", "checkpoint": "checkpoint_420000.pth.tar"},
        "Akinolú":  {"path": "YV/bible_to_male",   "checkpoint": "checkpoint_420000.pth.tar"},
        
        "Àrẹ̀mú":      {"path": "Af_lang/yor",      "checkpoint": "checkpoint_1100000.pth"},
        "Kofi":       {"path": "Af_lang/ewe",      "checkpoint": "checkpoint_1100000.pth"},
        "Danjuma":     {"path": "Af_lang/hau",      "checkpoint": "checkpoint_1100000.pth"},
        "Uzima":   {"path": "Af_lang/lin",      "checkpoint": "checkpoint_1100000.pth"},
        "Kwesi":    {"path": "Af_lang/twi-aku",  "checkpoint": "checkpoint_1100000.pth"},
        "Kwame":     {"path": "Af_lang/twi-asa",  "checkpoint": "checkpoint_1100000.pth"},
        
    }

    @modal.enter()
    def load_models(self):
        from TTS.utils.synthesizer import Synthesizer
        print("Initializing all TTS synthesizers...")
        self.synthesizers = {}
        for voice_name, config in self.VOICE_CONFIG.items():
            try:
                print(f"-> Loading voice: {voice_name}")
                model_path = os.path.join(MODEL_DIR, config["path"])
                checkpoint_path = os.path.join(model_path, config["checkpoint"])
                config_path = os.path.join(model_path, "config.json")
                if not os.path.exists(checkpoint_path) or not os.path.exists(config_path):
                    print(f"   [WARNING] Model files not found for {voice_name} at {model_path}. Skipping.")
                    continue
                self.synthesizers[voice_name] = Synthesizer(
                    checkpoint_path, config_path, use_cuda=True
                )
                print(f"   [SUCCESS] Loaded {voice_name}")
            except Exception as e:
                print(f"   [ERROR] Failed to load model for {voice_name}: {e}")
        print(f"Successfully loaded {len(self.synthesizers)} out of {len(self.VOICE_CONFIG)} voices.")
        self.s3_client = boto3.client("s3")
        self.s3_bucket = os.environ["S3_BUCKET_NAME"]

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: GenerateSpeechRequest) -> GenerateSpeechResponse:
        if request.voice not in self.synthesizers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice '{request.voice}'. Please use one of the available voices from the /voices endpoint."
            )
        try:
            if len(request.text) > 5000:
                raise HTTPException( status_code=400, detail="Text length exceeds 5000 characters.")
            synthesizer = self.synthesizers[request.voice]
            sample_rate = synthesizer.output_sample_rate
            print(f"Received request for '{request.voice}' voice. Chunking text...")
            text_chunks = text_chunker(request.text)
            print(f"Text split into {len(text_chunks)} chunks.")
            audio_segments = []
            for i, chunk in enumerate(text_chunks):
                print(f"Synthesizing chunk {i+1}/{len(text_chunks)}...")
                wav_chunk = synthesizer.tts(chunk)
                audio_segments.append(np.array(wav_chunk))
                if i < len(text_chunks) - 1:
                    silence = np.zeros(int(sample_rate * 0.3))
                    audio_segments.append(silence)
            print("Stitching audio chunks together...")
            full_audio = np.concatenate(audio_segments)
            audio_id = str(uuid.uuid4())
            local_path = f"/tmp/{audio_id}.wav"
            s3_key = f"{S3_AUDIO_PREFIX}/{audio_id}.wav"
            sf.write(local_path, full_audio, sample_rate)
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            os.remove(local_path)
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object', Params={'Bucket': self.s3_bucket, 'Key': s3_key}, ExpiresIn=3600
            )
            return GenerateSpeechResponse(audio_url=presigned_url, s3_key=s3_key)
        except Exception as e:
            print(f"Error during speech generation: {e}")
            raise HTTPException(status_code=500, detail="Internal server error.")

@app.local_entrypoint()
def main():
    import requests
    model = TTSPlatform()
    
    # Let's test one of the new voices to be sure it works
    test_voice = "Adéṣínà"
    print(f"--- Testing /generate endpoint for '{test_voice}' voice ---")
    generate_url = model.generate.get_web_url()
    yoruba_text = "Àkọ́lé: Àṣà Ìsọmọlórúkọ ní Ilẹ̀ Yorùbá: Ohun Pàtàkì Tó Kọjá Orúkọ Lásán"
    
    payload = {"text": yoruba_text, "voice": test_voice}
    response = requests.post(generate_url, json=payload)
    response.raise_for_status()
    result = response.json()
    print(f"✅ Success! Audio generated for {test_voice}.")
    print(f"   S3 Key: {result['s3_key']}")
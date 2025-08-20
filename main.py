import os
import re
import uuid
from typing import List, Literal

import modal
import boto3
import numpy as np
import soundfile as sf
from fastapi import HTTPException
from pydantic import BaseModel

# --- Configuration ---
MODEL_DIR = "/models"
HF_REPO_ID = "tolasekelvin/YV"
S3_AUDIO_PREFIX = "yoruba-tts-audio" # A folder in your S3 bucket for organization

# --- Modal App Setup ---
app = modal.App("yoruba-tts-api-service")

# Image definition
image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install_from_requirements("requirements.txt")
    .apt_install("libsndfile1")
    .run_commands(f"huggingface-cli download {HF_REPO_ID} --local-dir {MODEL_DIR} --local-dir-use-symlinks False")
)

# Secret should contain: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME
speechstackai_secret = modal.Secret.from_name("speechstackai-secret")

# --- Pydantic Models for API ---

# CHANGE 1: Renamed 'gender' to 'voice' and updated the allowed names.
class GenerateSpeechRequest(BaseModel):
    text: str
    voice: Literal["Adéṣínà", "Abọ́sẹ̀dé"]

class GenerateSpeechResponse(BaseModel):
    audio_url: str
    s3_key: str

# --- Helper Function for Text Chunking ---

def text_chunker(text: str, max_chunk_size: int = 500):
    """
    Splits text into chunks that are safe for the TTS model.
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    current_pos = 0
    text_len = len(text)
    
    while current_pos < text_len:
        chunk_end = min(current_pos + max_chunk_size, text_len)
        if chunk_end == text_len:
            chunks.append(text[current_pos:])
            break
        
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
class YorubaTTS:
    @modal.enter()
    def load_models(self):
        """
        Loads the TTS models and creates a mapping from public names to internal keys.
        """
        from TTS.utils.synthesizer import Synthesizer
        
        # CHANGE 2: Create a map from the public voice names to the internal model keys.
        self.voice_map = {
            "Adéṣínà": "male",    # Adéṣínà is the male voice
            "Abọ́sẹ̀dé": "female"  # Abọ́sẹ̀dé is the female voice
        }

        print("Initializing TTS Synthesizers...")
        self.synthesizers = {}
        try:
            # The internal loading logic remains the same, using "male" and "female" keys.
            female_model_dir = os.path.join(MODEL_DIR, "female_yoruba_tts")
            self.synthesizers["female"] = Synthesizer(
                os.path.join(female_model_dir, "checkpoint_500000.pth.tar"),
                os.path.join(female_model_dir, "config.json"),
                use_cuda=True,
            )
            male_model_dir = os.path.join(MODEL_DIR, "male_yoruba_tts")
            self.synthesizers["male"] = Synthesizer(
                os.path.join(male_model_dir, "checkpoint_500000.pth.tar"),
                os.path.join(male_model_dir, "config.json"),
                use_cuda=True,
            )
            print("Synthesizers loaded successfully.")
        except Exception as e:
            print(f"FATAL: Failed to load models: {e}")
            self.synthesizers = {}

        self.s3_client = boto3.client("s3")
        self.s3_bucket = os.environ["S3_BUCKET_NAME"]

    # --- API Endpoints ---

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        """Checks if the models are loaded and the service is healthy."""
        # This check remains internal and doesn't need to change.
        if "male" in self.synthesizers and "female" in self.synthesizers:
            return {"status": "healthy", "models_loaded": list(self.synthesizers.keys())}
        return {"status": "unhealthy", "reason": "One or more TTS models failed to load."}

    @modal.fastapi_endpoint(method="GET")
    def voices(self) -> dict[str, List[str]]:
        """Returns the list of available, user-friendly voice names."""
        # CHANGE 3: Return the public names from the voice_map.
        return {"voices": list(self.voice_map.keys())}

    @modal.fastapi_endpoint(method="POST")
    def generate(self, request: GenerateSpeechRequest) -> GenerateSpeechResponse:
        """
        Generates speech from text using the selected voice name.
        """
        if not self.synthesizers:
            raise HTTPException(
                status_code=503, detail="Service is unavailable: Models are not loaded."
            )
        try:
            if len(request.text) > 5000:
                raise HTTPException(
                    status_code=400, detail="Text length exceeds the limit of 5000 characters."
                )

            # CHANGE 4: Use the voice_map to get the correct internal model key.
            internal_key = self.voice_map[request.voice]
            synthesizer = self.synthesizers[internal_key]
            
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
            
            print(f"Saving concatenated audio to {local_path}...")
            sf.write(local_path, full_audio, sample_rate)

            print(f"Uploading {s3_key} to S3 bucket '{self.s3_bucket}'...")
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            
            os.remove(local_path)

            print("Generating presigned URL...")
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket, 'Key': s3_key},
                ExpiresIn=3600
            )

            return GenerateSpeechResponse(audio_url=presigned_url, s3_key=s3_key)

        except Exception as e:
            print(f"Error during speech generation: {e}")
            raise HTTPException(
                status_code=500, detail="An internal error occurred while generating speech."
            )

@app.local_entrypoint()
def main():
    """
    This function runs an end-to-end test of the API locally.
    """
    import requests

    model = YorubaTTS()

    print("--- Testing /voices endpoint ---")
    voices_url = model.voices.get_web_url()
    response = requests.get(voices_url)
    response.raise_for_status()
    print(f"✅ Success! Available voices: {response.json()['voices']}")
    print("-" * 30)

    print("--- Testing /health endpoint ---")
    health_url = model.health.get_web_url()
    response = requests.get(health_url)
    response.raise_for_status()
    print(f"✅ Success! Health status: {response.json()['status']}")
    print("-" * 30)

    # CHANGE 5: Update the test to use the new voice names.
    print("--- Testing /generate endpoint for 'Abọ́sẹ̀dé' voice ---")
    generate_url = model.generate.get_web_url()

    newscast_text="""
                    Ẹ kú àṣálẹ́ o, ẹ̀yin olùgbọ́ wa níbikíbi tí ẹ bá ti ń gbọ́ wa káàkiri agbami ayé. Orí ètò ìròyìn alẹ́ ọjọ́rú láti ilé-iṣẹ́ ìgbóhùnsáfẹ́fẹ́ 'Ohùn Àgbáyé' ni ẹ ti ń gbọ́ wa.
                """
    
    payload = {
        "text": newscast_text,
        "voice": "Abọ́sẹ̀dé"
    }

    response = requests.post(generate_url, json=payload)
    response.raise_for_status()
    result = response.json()
    
    print(f"✅ Success! Audio generated.")
    print(f"   S3 Key: {result['s3_key']}")
    print(f"   Presigned URL (valid for 1 hour): {result['audio_url']}")
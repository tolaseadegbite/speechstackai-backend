import os
from coqpit import Coqpit
from TTS.utils.synthesizer import Synthesizer

# --- Helper function to load and synthesize (to avoid repetition) ---
def synthesize_voice(model_dir: str, text_to_synthesize: str, output_audio_file: str, gpu: bool = False):
    """
    Loads a TTS model using the Synthesizer class and synthesizes speech.

    Args:
        model_dir (str): Path to the directory containing checkpoint and config files.
        text_to_synthesize (str): The text to convert to speech.
        output_audio_file (str): Path to save the synthesized audio file.
        gpu (bool): Whether to use GPU for inference. Defaults to False (CPU).
    """
    checkpoint_file = os.path.join(model_dir, "checkpoint_500000.pth.tar")
    config_file = os.path.join(model_dir, "config.json")

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    print(f"Initializing TTS Synthesizer from {model_dir}... This might take a moment.")

    # The Synthesizer class directly takes the paths to the checkpoint and config file.
    # It internally handles loading the model, tokenizer, audio processor, and setting up the device.
    synthesizer = Synthesizer(
        tts_checkpoint=checkpoint_file,
        tts_config_path=config_file,
        use_cuda=gpu
    )

    print("TTS Synthesizer loaded successfully.")

    # Perform synthesis using the synthesizer's high-level tts method
    print(f"Synthesizing speech for: '{text_to_synthesize}'")

    # The .tts() method returns the raw waveform as a NumPy array
    waveform = synthesizer.tts(
        text=text_to_synthesize,
        speaker_name=None, # Set this if you have a multi-speaker model and specific speaker names
        language_name=None # Set this if you have a multi-lingual model and specific language names
    )

    # Save the audio using the synthesizer's built-in save_wav method
    synthesizer.save_wav(waveform, output_audio_file)

    print(f"Speech synthesized and saved to {output_audio_file}")


## Model Configurations

# --- Configuration for Female Voice ---
# IMPORTANT: This path must match where your working model and config files are located,
# as verified by your successful tts-server run.
female_model_dir = "models/female_yoruba_tts/"
output_audio_file_female = "output_female_speech.wav"
text_to_synthesize_female = 'Ẹ jẹ́ kí n fi Ọ̀gbẹ́ni Tọ́láṣe Adégbìtẹ́ hàn yín, ọ̀kan lára àwọn ọmọ Yorùbá tó ń fi òye iṣẹ́-ẹ̀rọ mú ìdàgbàsókè bá ilẹ̀ wa.'

# --- Perform Female Voice Synthesis ---
synthesize_voice(female_model_dir, text_to_synthesize_female, output_audio_file_female, gpu=False)

print("\n" + "="*50 + "\n") # Separator

# --- Configuration for Male Voice ---
male_model_dir = "models/male_yoruba_tts/"
output_audio_file_male = "output_male_speech.wav"
text_to_synthesize_male = 'Ẹ jẹ́ kí n fi Ọ̀gbẹ́ni Tọ́láṣe Adégbìtẹ́ hàn yín, ọ̀kan lára àwọn ọmọ Yorùbá tó ń fi òye iṣẹ́-ẹ̀rọ mú ìdàgbàsókè bá ilẹ̀ wa.'

# --- Perform Male Voice Synthesis ---
synthesize_voice(male_model_dir, text_to_synthesize_male, output_audio_file_male, gpu=False)
import os
import subprocess
import httpx
import base64
from typing import Tuple

ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
ELEVEN_LABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"

async def opus_to_wav(input_path: str, output_path: str) -> None:
    """Convert OGG/Opus to WAV/PCM 16kHz mono using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path
    ], check=True)

async def wav_to_opus(input_path: str, output_path: str) -> None:
    """Convert WAV to OGG/Opus using ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-c:a", "libopus", output_path
    ], check=True)

async def sarvam_stt(wav_file: str) -> tuple[str, str]:
    """Convert speech to text using Sarvam AI Saarika STT."""
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not found in environment")
    
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
    }
    
    files = {
        "file": (wav_file, open(wav_file, "rb"), "audio/wav")
    }
    
    data = {
        "model": "saarika:v2.5",
        "language_code": "unknown",  # Auto-detect language
        "speaker_diarization": "false",
        "enable_noise_reduction": "true"
    }
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:  # Longer timeout for speech processing
            response = await client.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            print(f"Sarvam STT Response: {result}")  # Debug log
            
            # Extract transcript and language from Sarvam response
            transcript = result.get("transcript", "")
            detected_language = result.get("language_code", "en-IN")
            
            # Convert language code format (hi-IN -> hi, en-IN -> en)
            lang_code = detected_language.split("-")[0] if "-" in detected_language else detected_language
            
            return transcript, lang_code
            
    except httpx.HTTPStatusError as e:
        print(f"Sarvam STT HTTP Error: {e.response.status_code} - {e.response.text}")
        raise Exception(f"Sarvam STT failed with status {e.response.status_code}")
    except Exception as e:
        print(f"Sarvam STT error: {e}")
        raise Exception(f"Sarvam STT error: {e}")
    finally:
        # Close the file
        if "file" in files:
            files["file"][1].close()


async def sarvam_tts(text: str, output_wav: str, language_code: str = "en-IN") -> None:
    """Synthesize text to speech using Sarvam AI Bulbul TTS."""
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY not found in environment")
    
    url = "https://api.sarvam.ai/text-to-speech"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Map language codes to Sarvam format (add -IN suffix if not present)
    if "-" not in language_code:
        if language_code == "en":
            language_code = "en-IN"
        elif language_code == "hi":
            language_code = "hi-IN"
        elif language_code == "bn":
            language_code = "bn-IN"
        elif language_code == "ta":
            language_code = "ta-IN"
        elif language_code == "te":
            language_code = "te-IN"
        elif language_code == "gu":
            language_code = "gu-IN"
        elif language_code == "kn":
            language_code = "kn-IN"
        elif language_code == "ml":
            language_code = "ml-IN"
        elif language_code == "mr":
            language_code = "mr-IN"
        elif language_code == "pa":
            language_code = "pa-IN"
        elif language_code == "od":
            language_code = "or-IN"  # Odia uses 'or' in Sarvam
        else:
            language_code = "en-IN"  # Default fallback
    
    # Use correct speakers available for bulbul:v2
    speakers = {
        "en-IN": "anushka",      # Female speakers: anushka, manisha, vidya, arya
        "hi-IN": "abhilash",     # Male speakers: abhilash, karun, hitesh
        "bn-IN": "anushka",      
        "ta-IN": "vidya",        
        "te-IN": "vidya",        
        "gu-IN": "manisha",      
        "kn-IN": "vidya",        
        "ml-IN": "vidya",        
        "mr-IN": "manisha",      
        "pa-IN": "karun",        
        "or-IN": "anushka"       
    }
    
    speaker = speakers.get(language_code, "anushka")  # Default to anushka
    
    # Format text properly - limit to 1500 chars (updated from docs)
    formatted_text = text[:1500]
    
    payload = {
        "text": formatted_text,
        "target_language_code": language_code,
        "speaker": speaker,
        "pitch": 0.0,
        "pace": 1.0,
        "loudness": 1.0,
        "speech_sample_rate": 22050,
        "enable_preprocessing": True,  # Better for mixed-language text
        "model": "bulbul:v2"
    }
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse JSON response
            result = response.json()
            print(f"Sarvam TTS Response keys: {list(result.keys())}")
            
            # Extract base64 audio data from "audios" array (not "audio")
            audios = result.get("audios", [])
            if not audios or len(audios) == 0:
                print(f"No 'audios' array in response: {result}")
                raise Exception("No audio data received from Sarvam TTS")
            
            # Get first audio from the array
            audio_base64 = audios[0]
            if not audio_base64:
                print(f"Empty audio data in audios array: {audios}")
                raise Exception("Empty audio data received from Sarvam TTS")
            
            # Decode base64 audio to binary
            try:
                audio_data = base64.b64decode(audio_base64)
                print(f"Successfully decoded {len(audio_data)} bytes of audio data")
            except Exception as decode_error:
                print(f"Failed to decode base64 audio: {decode_error}")
                raise Exception(f"Failed to decode audio data: {decode_error}")
            
            # Save audio data to WAV file
            with open(output_wav, "wb") as f:
                f.write(audio_data)
            
            print(f"Saved {len(audio_data)} bytes of audio to {output_wav}")
                
    except httpx.HTTPStatusError as e:
        print(f"Sarvam TTS HTTP Error: {e.response.status_code} - {e.response.text}")
        # Fallback to ElevenLabs if available
        if ELEVEN_LABS_API_KEY:
            print("Falling back to ElevenLabs TTS...")
            await elevenlabs_tts_fallback(text, output_wav)
        else:
            raise Exception(f"Sarvam TTS failed with status {e.response.status_code}")
    except Exception as e:
        print(f"Sarvam TTS error: {e}")
        # Fallback to ElevenLabs if available
        if ELEVEN_LABS_API_KEY:
            print("Falling back to ElevenLabs TTS...")
            await elevenlabs_tts_fallback(text, output_wav)
        else:
            raise Exception(f"Sarvam TTS error: {e}")

async def elevenlabs_tts_fallback(text: str, output_wav: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> None:
    """Fallback TTS using ElevenLabs."""
    headers = {
        "xi-api-key": ELEVEN_LABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    url = f"{ELEVEN_LABS_TTS_URL}/{voice_id}"
    
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        with open(output_wav, "wb") as f:
            f.write(response.content)

# Main TTS function - uses Sarvam by default
async def text_to_speech(text: str, output_wav: str, voice_id: str = "default", language_code: str = "en-IN") -> None:
    """Use Sarvam AI Bulbul for TTS with ElevenLabs as fallback."""
    await sarvam_tts(text, output_wav, language_code)
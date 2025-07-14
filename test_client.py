import requests
import time
from pathlib import Path


class TTSClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def health_check(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Service is {health_data['status']}")
                print(f"   Model loaded: {health_data['model_loaded']}")
                print(f"   GPU available: {health_data['gpu_available']}")
                print(f"   Sample rate: {health_data['sample_rate']}")
                print(f"   Supported languages: {health_data['supported_languages']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Could not connect to service: {e}")
            return False

    def synthesize(self,
                   text: str,
                   valence: float,
                   arousal: float,
                   reference_audio_path: str,
                   language: str = "en",
                   temperature: float = 0.75,
                   length_penalty: float = 1.0,
                   repetition_penalty: float = 2.0,
                   top_p: float = 0.8,
                   output_file: str = None):
        """
        Synthesize speech with reference audio

        Args:
            text: Text to synthesize
            valence: Emotional valence (-1.0 to 1.0)
            arousal: Emotional arousal (-1.0 to 1.0)
            reference_audio_path: Path to reference audio file
            language: Target language
            temperature: Generation temperature
            length_penalty: Length penalty
            repetition_penalty: Repetition penalty
            top_p: Top-p sampling
            output_file: Output filename (auto-generated if None)
        """

        if not Path(reference_audio_path).exists():
            print(f"❌ Reference audio file not found: {reference_audio_path}")
            return None

        # Form data for the API
        data = {
            "text": text,
            "valence": valence,
            "arousal": arousal,
            "language": language,
            "temperature": temperature,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "top_p": top_p
        }

        with open(reference_audio_path, "rb") as f:
            files = {"reference_audio": f}

            print(f"🎤 Synthesizing: '{text[:50]}...'")
            print(f"   Reference: {reference_audio_path}")
            print(f"   Valence: {valence}, Arousal: {arousal}")

            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/synthesize",  # Updated endpoint
                    data=data,
                    files=files
                )
                duration = time.time() - start_time

                if response.status_code == 200:
                    if output_file is None:
                        output_file = f"output_{int(time.time())}.wav"

                    with open(output_file, "wb") as f:
                        f.write(response.content)

                    print(f"✅ Audio saved: {output_file} (took {duration:.2f}s)")
                    return output_file
                else:
                    print(f"❌ Error: {response.status_code} - {response.text}")
                    return None

            except Exception as e:
                print(f"❌ Request failed: {e}")
                return None


def main():
    """Test the TTS API with different emotional states"""
    client = TTSClient()

    print("=== TTS API Test ===")

    # Health check
    if not client.health_check():
        print("Service not available, exiting...")
        return

    print("\n=== Testing different emotional states ===")

    test_cases = [
        {
            "text": "Hello, how are you doing today?",
            "valence": 0.8,  # Happy
            "arousal": 0.6,  # Excited
            "description": "Happy/Excited",
            "reference_audio_path": "reference_audio/stef_voice.wav"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['description']} ---")
        client.synthesize(
            text=test_case["text"],
            valence=test_case["valence"],
            arousal=test_case["arousal"],
            reference_audio_path=test_case["reference_audio_path"],
            output_file=f"test_{i}_{test_case['description'].lower().replace('/', '_').replace(' ', '_')}.wav"
        )
        time.sleep(1)  # Brief pause between requests

    print("\n🎉 Testing complete!")


if __name__ == "__main__":
    main()
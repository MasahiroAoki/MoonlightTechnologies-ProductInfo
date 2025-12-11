# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki
#
# This script demonstrates how to submit a multimodal (text + image + audio)
# prompt to the distributed brain simulation using the SDK.

import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from evospikenet import EvoSpikeNetAPIClient
from requests.exceptions import RequestException

# Use the dummy files created at the project root
DUMMY_IMAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dummy_image.png'))
DUMMY_AUDIO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dummy_audio.wav'))


def main():
    """
    Main function to run the multimodal simulation query workflow.
    """
    print("--- Multimodal Distributed Brain Simulation SDK Example ---")

    # 1. Initialize the client
    client = EvoSpikeNetAPIClient()

    # 2. Check server health
    if not client.is_server_healthy():
        print("Error: API server is not healthy or not running.")
        return

    print("✅ API server is healthy.")

    # 3. Verify that the dummy media files exist
    if not os.path.exists(DUMMY_IMAGE_PATH):
        print(f"Error: Dummy image not found at {DUMMY_IMAGE_PATH}")
        return
    if not os.path.exists(DUMMY_AUDIO_PATH):
        print(f"Error: Dummy audio not found at {DUMMY_AUDIO_PATH}")
        return

    print(f"✅ Found dummy image: {DUMMY_IMAGE_PATH}")
    print(f"✅ Found dummy audio: {DUMMY_AUDIO_PATH}")

    try:
        # 4. Submit a multimodal prompt
        print("\nSubmitting a multimodal prompt (text, image, audio)...")
        prompt_text = "Describe the image and analyze the sound."
        submission_response = client.submit_prompt(
            prompt=prompt_text,
            image_path=DUMMY_IMAGE_PATH,
            audio_path=DUMMY_AUDIO_PATH
        )
        print(f"✅ Prompt submitted successfully. Server response: {submission_response}")

        # 5. Poll for the result
        print("\nPolling for the simulation result...")
        result = client.poll_for_result(timeout=120, interval=5)

        # 6. Print the final result
        if result and result.get("response"):
            print("\n--- Simulation Result Received ---")
            print(f"Response: {result['response']}")
            print(f"Timestamp: {result['timestamp']}")
            print("------------------------------------")
        else:
            print("\n❌ Could not retrieve a result within the timeout period.")

    except RequestException as e:
        print(f"\nAn error occurred during the API request: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki
#
# This script demonstrates a complete workflow for interacting with the
# distributed brain simulation via the EvoSpikeNet Python SDK.

import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from evospikenet import EvoSpikeNetAPIClient
from requests.exceptions import RequestException

def main():
    """
    Main function to run the simulation query workflow.
    """
    print("--- Distributed Brain Simulation SDK Example ---")

    # 1. Initialize the client
    client = EvoSpikeNetAPIClient()

    # 2. Check if the API server is running
    if not client.is_server_healthy():
        print("Error: API server is not healthy or not running. Please start the server.")
        return

    print("✅ API server is healthy.")

    try:
        # 3. Submit a text prompt to the simulation
        print("\nSubmitting a prompt to the simulation...")
        prompt_text = "What is the capital of Japan?"
        submission_response = client.submit_prompt(prompt=prompt_text)
        print(f"✅ Prompt submitted successfully. Server response: {submission_response}")

        # 4. Poll for the result
        print("\nPolling for the simulation result...")
        result = client.poll_for_result(timeout=120, interval=5)

        # 5. Print the final result
        if result and result.get("response"):
            print("\n--- Simulation Result Received ---")
            print(f"Response: {result['response']}")
            print(f"Timestamp: {result['timestamp']}")
            print("------------------------------------")
        else:
            print("\n❌ Could not retrieve a result within the timeout period.")
            print("Checking simulation status for debugging...")
            try:
                status = client.get_simulation_status()
                print("Current simulation status:")
                import json
                print(json.dumps(status, indent=2))
            except RequestException as e:
                print(f"Could not get simulation status: {e}")

    except RequestException as e:
        print(f"\nAn error occurred during the API request: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

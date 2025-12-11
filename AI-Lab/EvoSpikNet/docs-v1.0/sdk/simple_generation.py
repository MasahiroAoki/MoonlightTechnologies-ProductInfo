# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki
#
# This script demonstrates the basic usage of the EvoSpikeNetAPIClient to generate text.
# It initializes the client, sends a simple prompt, and prints the response.

import os
import sys

# Add the project root to the Python path to allow importing evospikenet
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evospikenet import EvoSpikeNetAPIClient
from requests.exceptions import RequestException

def main():
    """
    Main function to run the simple generation example.
    """
    print("--- EvoSpikeNet API SDK: Simple Generation Example ---")

    # Initialize the client. Assumes the API server is running on the default URL.
    try:
        client = EvoSpikeNetAPIClient()
        print("API client initialized.")

        prompt_text = "Once upon a time, in a land of spiking neurons,"
        print(f"\\nSending prompt: '{prompt_text}'")

        # Call the generate method
        response = client.generate(prompt=prompt_text, max_length=75)

        print("\\n--- API Response ---")
        print(f"Original Prompt: {response['prompt']}")
        print(f"Generated Text:  {response['generated_text']}")
        print("--------------------")

    except RequestException as e:
        print(f"\\n❌ Error: Could not connect to the API server at {client.base_url}")
        print("Please ensure the API server is running by executing './scripts/run_api_server.sh'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

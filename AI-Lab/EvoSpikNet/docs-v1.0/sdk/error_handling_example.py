# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki
#
# This script demonstrates how the EvoSpikeNetAPIClient handles connection errors.
# It attempts to connect to a non-existent server to show how the SDK raises
# a requests.exceptions.RequestException.

import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evospikenet import EvoSpikeNetAPIClient
from requests.exceptions import RequestException

def main():
    """
    Main function to run the error handling example.
    """
    print("--- EvoSpikeNet API SDK: Error Handling Example ---")

    # Initialize the client with a non-existent URL
    invalid_url = "http://localhost:9999"
    client = EvoSpikeNetAPIClient(base_url=invalid_url)
    print(f"API client initialized with an invalid URL: {invalid_url}")

    prompt_text = "This request is expected to fail."
    print(f"\\nAttempting to send prompt: '{prompt_text}'")

    try:
        # This call is expected to raise a RequestException
        client.generate(prompt=prompt_text)

        # This part should not be reached
        print("\\n❌ Error: The API call succeeded unexpectedly.")

    except RequestException as e:
        print(f"\\n✅ Success: The API call failed as expected.")
        print("This demonstrates that the SDK correctly raises an exception when it cannot connect to the server.")
        print(f"Caught exception: {type(e).__name__}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

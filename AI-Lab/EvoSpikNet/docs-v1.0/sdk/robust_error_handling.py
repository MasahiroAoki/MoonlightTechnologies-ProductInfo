# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki
#
# This script demonstrates robust error handling and retries with the EvoSpikeNetAPIClient.
# It shows how to handle connection failures and implement retry logic.

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evospikenet import EvoSpikeNetAPIClient
from requests.exceptions import RequestException


def main():
    """
    Demonstrates error handling, validation, and retry logic.
    """
    print("--- EvoSpikeNet API SDK: Advanced Error Handling Example ---\n")

    try:
        client = EvoSpikeNetAPIClient()

        # 1. Wait for server to be healthy with retry
        print("[1] Waiting for API server to be healthy...")
        if not client.wait_for_server(timeout=30, interval=2):
            print("❌ Could not connect to server after retries")
            return

        # 2. Get server information
        print("\n[2] Retrieving server information...")
        server_info = client.get_server_info()
        if server_info:
            print(f"✓ Server info: {server_info}")
        else:
            print("⚠ Server info not available")

        # 3. Validate prompt before submission
        print("\n[3] Validating prompts...")
        test_prompts = [
            "Valid prompt",
            "",  # Invalid: empty
            "x" * 15000  # Invalid: too long
        ]

        for prompt in test_prompts:
            if client.validate_prompt(prompt):
                print(f"✓ Valid: {prompt[:30]}...")
            else:
                print(f"✗ Invalid: {prompt[:30] if prompt else '(empty)'}...")

        # 4. Use with_error_handling for robust text generation
        print("\n[4] Generating text with automatic retry...")
        prompt = "The future of technology will be"
        result = client.with_error_handling(
            client.generate,
            prompt=prompt,
            max_length=50,
            retries=3
        )

        if result:
            print(f"✓ Successfully generated text:")
            print(f"  Input:  {result['prompt']}")
            print(f"  Output: {result['generated_text']}")
        else:
            print("✗ Failed to generate text after all retries")

        print("\n--- Example Complete ---")

    except RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()

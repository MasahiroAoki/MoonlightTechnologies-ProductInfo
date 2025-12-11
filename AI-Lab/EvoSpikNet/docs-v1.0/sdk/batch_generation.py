# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki
#
# This script demonstrates batch text generation using the EvoSpikeNetAPIClient.
# It processes multiple prompts and collects all results.

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evospikenet import EvoSpikeNetAPIClient
from requests.exceptions import RequestException


def main():
    """
    Demonstrates batch text generation with multiple prompts.
    """
    print("--- EvoSpikeNet API SDK: Batch Generation Example ---\n")

    try:
        client = EvoSpikeNetAPIClient()

        # List of prompts to process
        prompts = [
            "In the beginning,",
            "The future of artificial intelligence is",
            "Neural networks are fascinating because"
        ]

        print(f"Processing {len(prompts)} prompts...\n")

        # Use batch_generate to process all prompts
        results = client.batch_generate(prompts, max_length=50)

        print("\n--- Batch Generation Results ---")
        for i, result in enumerate(results, 1):
            if "error" in result:
                print(f"\n[Prompt {i}] ❌ Error: {result['error']}")
            else:
                print(f"\n[Prompt {i}]")
                print(f"  Input:  {result['prompt']}")
                print(f"  Output: {result['generated_text']}")

    except RequestException as e:
        print(f"❌ Error: Could not connect to the API server")
        print("Please ensure the API server is running")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()

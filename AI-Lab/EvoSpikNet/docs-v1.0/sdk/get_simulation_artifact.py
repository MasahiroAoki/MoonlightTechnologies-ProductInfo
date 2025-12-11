# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki
#
# This script provides an example of how to use the EvoSpikeNetAPIClient
# to find and retrieve data artifacts that were uploaded by a distributed
# brain simulation.

import sys
import os
import io
import torch

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from evospikenet.sdk import EvoSpikeNetAPIClient

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"

def main():
    """
    Main function to find the latest simulation session, list its artifacts,
    and download a selected artifact for analysis.
    """
    print("--- EvoSpikeNet SDK: Get Simulation Artifact Example ---")

    try:
        # 1. Initialize the API client
        client = EvoSpikeNetAPIClient(base_url=API_BASE_URL)
        print(f"Connected to API server at {API_BASE_URL}")

        # 2. Find the most recent execution session
        # In a real scenario, you might have a more specific way to find the
        # session you care about, but for this example, we'll just get the latest one.
        sessions = client.get_all_sessions()
        if not sessions:
            print("No execution sessions found in the database.")
            return

        # Sort sessions by start time to find the most recent
        latest_session = sorted(sessions, key=lambda s: s['start_time'], reverse=True)[0]
        session_id = latest_session['session_id']
        print(f"\nFound latest session: {session_id} (Started at: {latest_session['start_time']})")

        # 3. Get all artifacts associated with that session
        artifacts = client.get_session_artifacts(session_id)

        # Filter for only simulation data artifacts for clarity
        sim_artifacts = [a for a in artifacts if a['artifact_type'] == 'simulation_data']

        if not sim_artifacts:
            print("No simulation data artifacts found for this session.")
            return

        # 4. Display artifacts and prompt user for selection
        print("\nAvailable simulation artifacts:")
        for i, artifact in enumerate(sim_artifacts):
            print(f"  [{i}] ID: {artifact['artifact_id']} | Name: {artifact['name']}")

        try:
            selection = int(input("\nEnter the number of the artifact to download: "))
            if not 0 <= selection < len(sim_artifacts):
                raise ValueError("Invalid selection.")
            selected_artifact_id = sim_artifacts[selection]['artifact_id']
        except (ValueError, IndexError):
            print("Invalid input. Exiting.")
            return

        # 5. Download the selected artifact
        print(f"\nDownloading artifact {selected_artifact_id}...")
        artifact_data = client.get_artifact(selected_artifact_id)

        # The data is returned as raw bytes. We need to load it into a buffer.
        buffer = io.BytesIO(artifact_data)

        # 6. Load the tensor data and analyze it
        # Note: Using weights_only=True is a security best practice.
        data_tensor = torch.load(buffer, weights_only=True)

        print("\n--- Artifact Analysis ---")
        print(f"Successfully loaded tensor from artifact.")
        print(f"Tensor Shape: {data_tensor.shape}")
        print(f"Tensor DType: {data_tensor.dtype}")
        print(f"Tensor Mean: {data_tensor.float().mean().item():.4f}")
        print("-------------------------\n")


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the API server is running and accessible.")

if __name__ == "__main__":
    main()

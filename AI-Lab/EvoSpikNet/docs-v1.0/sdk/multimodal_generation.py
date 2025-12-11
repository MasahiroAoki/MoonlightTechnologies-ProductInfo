# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki
#
# This script demonstrates multimodal processing with text, image, and audio inputs
# using the EvoSpikeNetAPIClient.

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evospikenet import EvoSpikeNetAPIClient


def main():
    """
    Demonstrates multimodal generation with text, image, and audio inputs.
    """
    print("--- EvoSpikeNet API SDK: Multimodal Processing Example ---\n")

    try:
        client = EvoSpikeNetAPIClient()

        # Wait for server to be ready
        print("[1] Verifying server is available...")
        if not client.wait_for_server(timeout=30):
            print("❌ Server not available")
            return

        # Example 1: Text-only generation
        print("\n[2] Text-only Generation:")
        text_prompt = "Describe the benefits of renewable energy"
        result = client.generate(text_prompt, max_length=100)
        if result:
            print(f"✓ Generated: {result['generated_text'][:100]}...")

        # Example 2: Multimodal with image (base64 encoded)
        print("\n[3] Multimodal Generation (Text + Image):")
        # Note: In production, load actual image and encode to base64
        image_data = None  # Would be: base64.b64encode(open('image.jpg', 'rb').read())
        
        result = client.submit_prompt(
            prompt="What is depicted in this image?",
            text_input="Additional context about the image",
            image_data=image_data  # Would contain base64 encoded image
        )
        
        if result:
            task_id = result.get('task_id')
            print(f"✓ Multimodal task submitted: {task_id}")
            
            # Poll for result with timeout
            print("  Waiting for result...")
            final_result = client.poll_for_result(task_id, timeout=60, interval=2)
            if final_result:
                print(f"  ✓ Result: {final_result.get('output', '')[:100]}...")
            else:
                print("  ⚠ Timeout waiting for result")

        # Example 3: Multimodal with audio
        print("\n[4] Multimodal Generation (Text + Audio):")
        # Note: In production, load actual audio and encode to base64
        audio_data = None  # Would be: base64.b64encode(open('audio.wav', 'rb').read())
        
        result = client.submit_prompt(
            prompt="Transcribe and describe this audio",
            text_input="Audio context: interview recording",
            audio_data=audio_data  # Would contain base64 encoded audio
        )
        
        if result:
            task_id = result.get('task_id')
            print(f"✓ Audio analysis task submitted: {task_id}")
            
            # Get result with custom status checking
            status = client.get_simulation_status(task_id)
            print(f"  Status: {status}")

        # Example 4: Processing multiple multimodal inputs with error handling
        print("\n[5] Batch Multimodal Processing with Error Handling:")
        
        multimodal_prompts = [
            {"prompt": "Describe this scene", "text_input": "Scene 1"},
            {"prompt": "Analyze this audio", "text_input": "Audio 1"},
            {"prompt": "Process this input", "text_input": "Input 3"}
        ]
        
        success_count = 0
        for i, item in enumerate(multimodal_prompts):
            try:
                # Validate prompt first
                if not client.validate_prompt(item['prompt']):
                    print(f"  ✗ Item {i+1}: Invalid prompt")
                    continue
                
                result = client.submit_prompt(**item)
                if result:
                    print(f"  ✓ Item {i+1}: Task submitted")
                    success_count += 1
                else:
                    print(f"  ⚠ Item {i+1}: Failed to submit")
                    
            except Exception as e:
                print(f"  ✗ Item {i+1}: Error - {e}")
        
        print(f"\n✓ Successfully submitted {success_count}/{len(multimodal_prompts)} tasks")
        print("\n--- Example Complete ---")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

# Copyright 2025 Moonlight Technologies Inc.
# Auth Masahiro Aoki
#
# This script demonstrates asynchronous patterns and long-running task management
# with the EvoSpikeNetAPIClient.

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evospikenet import EvoSpikeNetAPIClient


def main():
    """
    Demonstrates async patterns, task polling, and result retrieval.
    """
    print("--- EvoSpikeNet API SDK: Asynchronous Patterns Example ---\n")

    try:
        client = EvoSpikeNetAPIClient()

        # Ensure server is ready
        print("[1] Checking server availability...")
        if not client.wait_for_server(timeout=30):
            print("❌ Server not available")
            return

        # Pattern 1: Submit task and poll with custom interval
        print("\n[2] Pattern 1: Submit Task and Async Polling")
        print("-" * 50)
        
        prompt = "Explain the concept of neuroplasticity in detail"
        print(f"Submitting: {prompt}")
        
        result = client.generate(prompt, max_length=500)
        if result and 'task_id' in result:
            task_id = result['task_id']
            print(f"✓ Task submitted: {task_id}")
            
            # Custom polling logic with timeout
            max_wait = 120  # seconds
            poll_interval = 5  # seconds
            start_time = time.time()
            
            print("Polling for results...")
            while time.time() - start_time < max_wait:
                status = client.get_simulation_status(task_id)
                print(f"  ↻ Status: {status.get('state', 'unknown')}")
                
                if status.get('state') == 'completed':
                    print(f"✓ Task completed!")
                    final_result = client.get_simulation_result(task_id)
                    if final_result:
                        print(f"  Result: {str(final_result)[:100]}...")
                    break
                
                time.sleep(poll_interval)
            else:
                print(f"⚠ Task did not complete within {max_wait}s")

        # Pattern 2: Submit multiple tasks and collect results
        print("\n[3] Pattern 2: Multiple Async Tasks")
        print("-" * 50)
        
        prompts = [
            "What is artificial intelligence?",
            "Explain machine learning algorithms",
            "Describe deep neural networks"
        ]
        
        tasks = []
        print("Submitting multiple tasks...")
        for i, prompt in enumerate(prompts):
            result = client.generate(prompt, max_length=100)
            if result and 'task_id' in result:
                tasks.append({
                    'id': result['task_id'],
                    'prompt': prompt,
                    'submitted': time.time()
                })
                print(f"  ✓ Task {i+1}: {result['task_id']}")
            else:
                print(f"  ✗ Task {i+1}: Failed to submit")

        # Wait for all tasks with individual timeouts
        print("\nWaiting for all tasks...")
        completed = 0
        for task in tasks:
            timeout = 120
            start = time.time()
            
            while time.time() - start < timeout:
                status = client.get_simulation_status(task['id'])
                if status.get('state') == 'completed':
                    print(f"  ✓ {task['id']}: Complete")
                    completed += 1
                    break
                time.sleep(2)
            else:
                print(f"  ⚠ {task['id']}: Timeout")
        
        print(f"\n✓ {completed}/{len(tasks)} tasks completed")

        # Pattern 3: Submit with validation and error handling
        print("\n[4] Pattern 3: Validated Async Submission")
        print("-" * 50)
        
        test_prompts = [
            "Short prompt",
            "A much longer prompt that provides detailed context for the model to generate high-quality responses",
            "Another example"
        ]
        
        for prompt in test_prompts:
            # Validate before submission
            if not client.validate_prompt(prompt):
                print(f"  ✗ Rejected: {prompt[:40]}...")
                continue
            
            # Submit with retry logic
            result = client.with_error_handling(
                client.generate,
                prompt=prompt,
                max_length=200,
                retries=3
            )
            
            if result:
                print(f"  ✓ Accepted: {prompt[:40]}...")
            else:
                print(f"  ✗ Failed: {prompt[:40]}...")

        # Pattern 4: Log retrieval for long-running tasks
        print("\n[5] Pattern 4: Log Retrieval During Task Execution")
        print("-" * 50)
        
        # Create a log session for monitoring
        log_session = client.create_log_session()
        if log_session:
            session_id = log_session.get('session_id')
            print(f"✓ Created log session: {session_id}")
            
            # Submit a task
            result = client.generate(
                "Run a complex analysis task",
                max_length=500
            )
            
            if result and 'task_id' in result:
                task_id = result['task_id']
                
                # Retrieve logs while task is running
                print("Retrieving logs...")
                logs = client.get_remote_log(session_id, task_id)
                if logs:
                    print(f"  ✓ Retrieved {len(logs)} log entries")
                    for i, log in enumerate(logs[:3]):
                        print(f"    - {log}")
                else:
                    print("  ⚠ No logs available yet")

        print("\n--- Example Complete ---")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

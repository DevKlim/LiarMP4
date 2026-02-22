import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:8005" 
MOUNT_POINT = "/a2a"

def test_agent_handshake():
    print(f"--- Testing A2A Agent Connection ---")
    print(f"Targeting Base URL: {BASE_URL}")
    
    # Check Health First
    try:
        health = requests.get(f"{BASE_URL}/health")
        print(f"System Health Check: {health.status_code} | {health.text}")
    except requests.exceptions.ConnectionError:
        print(f"Health Check Failed: Could not connect to {BASE_URL}. Is the container running?")
        return

    # List of common A2A JSON-RPC methods to try
    methods_to_try = ["agent.process", "agent.generate", "model.generate", "a2a.generate"]
    
    # ADK usually mounts at the root of the mount point
    url = f"{BASE_URL}{MOUNT_POINT}"
    
    success_method = None

    for method in methods_to_try:
        print(f"\n[Attempt] Method: '{method}' -> {url}")
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": {
                "input": "Hello agent, are you online?"
            },
            "id": 1
        }

        try:
            response = requests.post(
                url, 
                json=payload, 
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            print(f"  > Status Code: {response.status_code}")
            
            if response.status_code == 404:
                # If /a2a fails, try /a2a/jsonrpc (rare but possible)
                alt_url = f"{BASE_URL}{MOUNT_POINT}/jsonrpc"
                print(f"  > 404 at root. Retrying at {alt_url}...")
                response = requests.post(alt_url, json=payload, headers={"Content-Type": "application/json"})
                print(f"  > Alt Status: {response.status_code}")
                if response.status_code == 200:
                    url = alt_url # Switch URL for future

            try:
                data = response.json()
                
                # Check for Method Not Found Error
                if "error" in data and data["error"].get("code") == -32601:
                    print(f"  > [FAIL] Method '{method}' not found.")
                    continue
                
                # Check for valid result or other error (like param error)
                if "result" in data:
                    print(f"  > [SUCCESS] Method '{method}' worked!")
                    print(json.dumps(data, indent=2))
                    success_method = method
                    break
                elif "error" in data:
                    print(f"  > [PARTIAL] Method recognized but errored: {data['error']}")
                    # If it's not 'Method not found', we assume the method name is correct
                    success_method = method
                    break
                    
            except json.JSONDecodeError:
                print(f"  > [FAIL] Invalid JSON response.")

        except requests.exceptions.ConnectionError:
            print(f"  > [NETWORK ERROR] Could not connect.")
            break

    if success_method:
        print(f"\n✅ SUCCESS! The working JSON-RPC method is: {success_method}")
        print(f"Please update your frontend configuration if necessary.")
    else:
        print("\n❌ FAILURE. No known methods accepted.")
        print("Check server logs for A2A mount details.")

if __name__ == "__main__":
    test_agent_handshake()

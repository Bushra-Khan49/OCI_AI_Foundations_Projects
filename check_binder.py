import json
import urllib.request
import sys

url = "https://mybinder.org/build/gh/Bushra-Khan49/OCI_AI_Foundations_Projects/main"
req = urllib.request.Request(url)

try:
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            line = line.decode('utf-8').strip()
            if line.startswith('data: '):
                data = json.loads(line[6:])
                phase = data.get('phase', '')
                message = data.get('message', '').strip()
                print(f"[{phase}] {message}")
                if phase == 'ready':
                    print("\nBinder build successfully completed! Environment is ready.")
                    sys.exit(0)
                elif phase == 'failed':
                    print("\nBinder build failed!")
                    sys.exit(1)
except Exception as e:
    print(f"Error fetching Binder build status: {e}")
    sys.exit(1)

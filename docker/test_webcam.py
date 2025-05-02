# This file has been moved to the webcam/ directory
# Please use the new location: docker/webcam/test_webcam.py

# This file will be removed in a future update
print("WARNING: This file has been moved to docker/webcam/test_webcam.py")
print("Please update your references to use the new location.")

# Import the actual implementation from the new location
import sys
import os

new_path = os.path.join(os.path.dirname(__file__), "webcam", "test_webcam.py")

if os.path.exists(new_path):
    print(f"Redirecting to: {new_path}")
    
    # Add the parent directory to the path
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Execute the module directly
    with open(new_path) as f:
        exec(f.read())
else:
    print(f"ERROR: New module location not found: {new_path}")
    print("The webcam directory may have been moved or deleted.")
    sys.exit(1)

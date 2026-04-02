#!/usr/bin/env python3
"""
Example usage of the Face Recognition API

This script demonstrates how to use the system:
1. Register a person with multiple images
2. Recognize faces in a test image
3. List registered people

Run this after starting the API server.
"""
import requests
from pathlib import Path

API_BASE = "http://localhost:5001"


def example_register():
    """Example: Register a new person"""
    print("\n=== Example 1: Register a Person ===")

    # Prepare registration data
    name = "John Doe"
    image_paths = list(Path("data/raw/John/").glob("*.jpg"))[:5]  # Use up to 5 images

    if not image_paths:
        print(f"Warning: No images found in data/raw/John/. Skipping registration.")
        print("Add some face images to data/raw/John/ to test registration.")
        return False

    files = [('images', open(p, 'rb')) for p in image_paths]
    data = {'name': name}

    response = requests.post(f"{API_BASE}/api/register", files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Successfully registered {name}")
        print(f"  Embeddings extracted: {result['data']['embeddings_extracted']}")
        print(f"  Total embeddings: {result['data']['total_embeddings']}")
        return True
    else:
        print(f"✗ Registration failed: {response.text}")
        return False


def example_recognize():
    """Example: Recognize faces in an image"""
    print("\n=== Example 2: Recognize Faces ===")

    test_image = Path("face1.jpg")  # Use existing image from project root
    if not test_image.exists():
        print(f"Warning: {test_image} not found. Skipping recognition.")
        return

    with open(test_image, 'rb') as f:
        files = {'image': f}
        params = {'threshold': 0.6}  # Optional

        response = requests.post(f"{API_BASE}/api/recognize", files=files, params=params)

    if response.status_code == 200:
        result = response.json()
        faces = result['data']['faces']
        print(f"✓ Detected {len(faces)} face(s)")

        for i, face in enumerate(faces):
            print(f"\nFace #{i+1}:")
            print(f"  BBox: {face['bbox']}")
            print(f"  Name: {face['name']}")
            print(f"  Confidence: {face['confidence']:.2%}")
    else:
        print(f"✗ Recognition failed: {response.text}")


def example_list_people():
    """Example: List all registered people"""
    print("\n=== Example 3: List Registered People ===")

    response = requests.get(f"{API_BASE}/api/people")

    if response.status_code == 200:
        result = response.json()
        people = result['data']['people']
        print(f"✓ Total registered people: {len(people)}")
        for person in people:
            print(f"  - {person}")
    else:
        print(f"✗ Failed: {response.text}")


def example_detect_only():
    """Example: Only detect faces (no recognition)"""
    print("\n=== Example 4: Detect Faces Only ===")

    test_image = Path("face1.jpg")
    if not test_image.exists():
        print(f"Warning: {test_image} not found.")
        return

    with open(test_image, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{API_BASE}/api/detect", files=files)

    if response.status_code == 200:
        result = response.json()
        faces = result['data']['faces']
        print(f"✓ Detected {len(faces)} face(s)")
        for face in faces:
            print(f"  BBox: {face['bbox']}, Quality: {face.get('quality_score', 'N/A'):.2f}")
    else:
        print(f"✗ Detection failed: {response.text}")


def main():
    """Run all examples"""
    print("=" * 60)
    print("Face Recognition API - Example Usage")
    print("=" * 60)

    # Check API health
    try:
        health = requests.get(f"{API_BASE}/api/health").json()
        if not health.get('success'):
            print("✗ API is not running or unhealthy")
            print(f"  Please start the server: python scripts/run_api.py")
            return
        print("✓ API is healthy")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API")
        print(f"  Please start the server: python scripts/run_api.py")
        return

    # Run examples
    example_register()
    example_recognize()
    example_detect_only()
    example_list_people()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

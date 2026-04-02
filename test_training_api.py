#!/usr/bin/env python3
"""
Test script for Training API endpoints
Tests all training-related functionality
"""
import requests
import json
import sys
from datetime import datetime

BASE_URL = "http://localhost:5001"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def print_result(test_name, success, data=None, error=None):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if data:
        print(f"   Response: {json.dumps(data, indent=2, ensure_ascii=False)[:200]}")
    if error:
        print(f"   Error: {error}")
    print()

def test_health():
    """Test health endpoint"""
    print_section("TEST 1: Health Check")
    try:
        res = requests.get(f"{BASE_URL}/api/health")
        data = res.json()
        print_result("Health Check", data.get('success'), data)
        return data.get('success')
    except Exception as e:
        print_result("Health Check", False, error=str(e))
        return False

def test_list_people():
    """Test listing people"""
    print_section("TEST 2: List People")
    try:
        res = requests.get(f"{BASE_URL}/api/people")
        data = res.json()
        print_result("List People", data.get('success'), data)
        return data.get('data', {}).get('people', [])
    except Exception as e:
        print_result("List People", False, error=str(e))
        return []

def test_train_single(people):
    """Test training single person"""
    print_section("TEST 3: Train Single Person")
    if not people:
        print_result("Train Single", False, error="No people available")
        return False
    
    person_name = people[0]
    try:
        res = requests.post(f"{BASE_URL}/api/train/{person_name}")
        data = res.json()
        print_result(f"Train '{person_name}'", data.get('success'), data)
        return data.get('success')
    except Exception as e:
        print_result(f"Train '{person_name}'", False, error=str(e))
        return False

def test_train_all():
    """Test training all people"""
    print_section("TEST 4: Train All People")
    try:
        res = requests.post(f"{BASE_URL}/api/train-all")
        data = res.json()
        print_result("Train All", data.get('success'), data)
        return data.get('success')
    except Exception as e:
        print_result("Train All", False, error=str(e))
        return False

def test_person_status(people):
    """Test getting person status"""
    print_section("TEST 5: Get Person Status")
    if not people:
        print_result("Get Status", False, error="No people available")
        return False
    
    person_name = people[0]
    try:
        res = requests.get(f"{BASE_URL}/api/person/{person_name}/status")
        data = res.json()
        print_result(f"Status '{person_name}'", data.get('success'), data)
        return data.get('success')
    except Exception as e:
        print_result(f"Status '{person_name}'", False, error=str(e))
        return False

def test_person_images(people):
    """Test getting person images"""
    print_section("TEST 6: Get Person Images")
    if not people:
        print_result("Get Images", False, error="No people available")
        return False
    
    person_name = people[0]
    try:
        res = requests.get(f"{BASE_URL}/api/person/{person_name}/images")
        data = res.json()
        success = data.get('success')
        image_count = data.get('data', {}).get('count', 0)
        print_result(
            f"Images '{person_name}'", 
            success, 
            {'count': image_count, 'has_images': image_count > 0}
        )
        return success
    except Exception as e:
        print_result(f"Images '{person_name}'", False, error=str(e))
        return False

def test_recognize():
    """Test recognition endpoint"""
    print_section("TEST 7: Recognition (No Image)")
    try:
        # Test without image - should return error
        res = requests.post(f"{BASE_URL}/api/recognize")
        data = res.json()
        # Expecting error since no image provided
        print_result("Recognize (no image)", not data.get('success'), data)
        return True
    except Exception as e:
        print_result("Recognize", False, error=str(e))
        return False

def main():
    print("\n" + "="*60)
    print("  FACE RECOGNITION API - TRAINING TEST SUITE")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = []
    
    # Test 1: Health
    results.append(("Health Check", test_health()))
    
    # Test 2: List People
    people = test_list_people()
    results.append(("List People", len(people) > 0))
    
    # Test 3: Train Single
    results.append(("Train Single", test_train_single(people)))
    
    # Test 4: Train All
    results.append(("Train All", test_train_all()))
    
    # Test 5: Person Status
    results.append(("Person Status", test_person_status(people)))
    
    # Test 6: Person Images
    results.append(("Person Images", test_person_images(people)))
    
    # Test 7: Recognition
    results.append(("Recognition", test_recognize()))
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n{'='*60}")
    print(f"  Total: {passed}/{total} tests passed")
    print(f"  Success Rate: {(passed/total*100):.1f}%")
    print(f"{'='*60}\n")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())

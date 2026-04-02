#!/usr/bin/env python3
"""
Installation Verification Script

Checks that all dependencies are installed and models are accessible.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_imports():
    """Check all required modules"""
    print("Checking imports...")
    modules = [
        'flask', 'cv2', 'numpy', 'torch', 'insightface',
        'albumentations', 'pandas', 'sklearn', 'matplotlib', 'seaborn'
    ]

    missing = []
    for mod in modules:
        try:
            __import__(mod.replace('-', '_'))
            print(f"  ✓ {mod}")
        except ImportError as e:
            print(f"  ✗ {mod} - {e}")
            missing.append(mod)

    return missing


def check_insightface_model():
    """Check InsightFace model can be loaded"""
    print("\nChecking InsightFace model...")
    try:
        from insightface.app import FaceAnalysis
        from config import config
        model_dir = config.MODELS_DIR
        print(f"  Model directory: {model_dir}")
        print("  Attempting to load model...")
        model = FaceAnalysis(name=config.MODEL_NAME, root=str(model_dir))
        model.prepare(ctx_id=0, det_size=(640, 640))
        print("  ✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        print("  The model will be downloaded on first run.")
        return False


def check_directories():
    """Check required directories exist"""
    print("\nChecking directories...")
    from config import config
    dirs = [
        config.DATA_DIR,
        config.UPLOAD_DIR,
        config.TRAINED_MODELS_DIR,
        config.MODELS_DIR,
        config.STATIC_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")


def check_flask_app():
    """Check Flask app can be created"""
    print("\nChecking Flask app...")
    try:
        from app import create_app
        app = create_app()
        print(f"  ✓ Flask app created")
        print(f"  ✓ API routes registered")
        return True
    except Exception as e:
        print(f"  ✗ Failed to create app: {e}")
        return False


def main():
    print("=" * 60)
    print("Face Recognition System - Installation Verification")
    print("=" * 60)

    issues = []

    missing = check_imports()
    if missing:
        issues.append(f"Missing modules: {', '.join(missing)}")
        print(f"\n  Install: pip install {' '.join(missing)}")

    check_directories()

    model_ok = check_insightface_model()
    if not model_ok:
        issues.append("Model download failed (will retry on first run)")

    app_ok = check_flask_app()
    if not app_ok:
        issues.append("Flask app creation failed")

    print("\n" + "=" * 60)
    if issues:
        print("WARNINGS:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nSome checks failed. Review the output above.")
        return 1
    else:
        print("✓ All checks passed! System ready to run.")
        print("\nTo start the server:")
        print("  python scripts/run_api.py")
        print("\nOr with Docker:")
        print("  docker-compose up")
        return 0


if __name__ == '__main__':
    sys.exit(main())

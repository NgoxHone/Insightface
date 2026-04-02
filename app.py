#!/usr/bin/env python3
"""
Main Flask Application for Face Recognition API

Run with: python app.py
"""
import sys
import logging
from pathlib import Path
import numpy as np
import cv2

from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS

from config import config
from api.routes import init_api, api_bp
from api.error_handlers import register_error_handlers

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE) if config.LOG_FILE.parent.exists() else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(
    __name__,
    static_folder=str(config.STATIC_DIR),
    template_folder=str(config.STATIC_DIR)
)

# CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = config.API_MAX_CONTENT_LENGTH
app.config['JSON_SORT_KEYS'] = False


# ============== Root Route ==============

@app.route('/')
def index():
    """Serve web interface or API documentation"""
    index_html = config.STATIC_DIR / 'index.html'
    if index_html.exists():
        return send_from_directory(config.STATIC_DIR, 'index.html')

    # Fallback to simple docs
    return send_from_directory(config.STATIC_DIR, 'index.html')
    # Alternative: return simple text if no index.html
    # return """
    # <h1>Face Recognition API</h1>
    # <p>Web UI not available. See <a href="/api/info">/api/info</a> for API documentation.</p>
    # """


# ============== Static Files ==============

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(config.STATIC_DIR, filename)


# ============== Error Handling ==============

register_error_handlers(app)


# ============== API Initialization ==============

# Register API blueprint
app.register_blueprint(api_bp)


# ============== Main ==============

def create_app():
    """Factory function for creating Flask app"""
    try:
        init_api()
        logger.info("Face Recognition API started successfully")
        return app
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


if __name__ == '__main__':
    try:
        # Initialize components
        init_api()

        # Run
        logger.info(f"Starting Face Recognition API on {config.API_HOST}:{config.API_PORT}")
        app.run(
            host=config.API_HOST,
            port=config.API_PORT,
            debug=config.API_DEBUG,
            threaded=config.API_THREADS > 1,
            processes=1
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        sys.exit(1)

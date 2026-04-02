#!/usr/bin/env python3
"""
Run Flask API Server

Usage:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 5001 --debug
"""
import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run Face Recognition API server'
    )
    parser.add_argument(
        '--host',
        type=str,
        help='Host address (default: from config)'
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Port number (default: from config)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of threads (default: 4)'
    )

    args = parser.parse_args()

    try:
        # Create app
        logger.info("Creating Flask app...")
        app = create_app()

        # Override config if arguments provided
        host = args.host or app.config.get('API_HOST', '0.0.0.0')
        port = args.port or app.config.get('API_PORT', 5000)
        debug = args.debug or app.config.get('API_DEBUG', False)

        logger.info("=" * 60)
        logger.info("Face Recognition API Server")
        logger.info("=" * 60)
        logger.info(f"Host: {host}")
        logger.info(f"Port: {port}")
        logger.info(f"Debug: {debug}")
        logger.info(f"Threads: {args.threads}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Endpoints:")
        logger.info("  GET  /                    - API documentation")
        logger.info("  GET  /api/health          - Health check")
        logger.info("  GET  /api/info            - System info")
        logger.info("  GET  /api/people          - List all people")
        logger.info("  POST /api/register        - Register new person")
        logger.info("  POST /api/recognize       - Recognize faces")
        logger.info("  POST /api/detect          - Detect faces")
        logger.info("  DELETE /api/person/<name> - Delete person")
        logger.info("")
        logger.info(f"Starting server at http://{host}:{port}")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)

        # Run
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=args.threads > 1,
            processes=1
        )

        return 0

    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        return 0
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

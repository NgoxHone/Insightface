"""
Error handlers forFlask API
Provides consistent JSON error responses
"""
import logging
import traceback
from typing import Tuple, Dict, Any, Optional
from flask import jsonify, Response

from .schemas import APIResponse

logger = logging.getLogger(__name__)


def register_error_handlers(app):
    """Register all error handlers with Flask app"""

    @app.errorhandler(400)
    def handle_bad_request(e):
        """Handle bad request"""
        logger.warning(f"400 Bad Request: {e}")
        return jsonify(APIResponse(
            success=False,
            error=f"Bad Request: {str(e)}"
        ).to_dict()), 400

    @app.errorhandler(404)
    def handle_not_found(e):
        """Handle not found"""
        logger.warning(f"404 Not Found: {e}")
        return jsonify(APIResponse(
            success=False,
            error="Resource not found"
        ).to_dict()), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(e):
        """Handle method not allowed"""
        logger.warning(f"405 Method Not Allowed: {e}")
        return jsonify(APIResponse(
            success=False,
            error="Method not allowed"
        ).to_dict()), 405

    @app.errorhandler(500)
    def handle_internal_error(e):
        """Handle internal server error"""
        logger.error(f"500 Internal Server Error: {e}", exc_info=True)
        return jsonify(APIResponse(
            success=False,
            error="Internal server error"
        ).to_dict()), 500

    @app.errorhandler(Exception)
    def handle_unhandled_exception(e):
        """Handle unhandled exceptions"""
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return jsonify(APIResponse(
            success=False,
            error="An unexpected error occurred"
        ).to_dict()), 500


def api_error(
    message: str,
    status_code: int = 400,
    details: Optional[Dict[str, Any]] = None
) -> Tuple[Response, int]:
    """
    Create an API error response

    Args:
        message: Error message
        status_code: HTTP status code
        details: Additional error details

    Returns:
        (Flask response, status code)
    """
    data = {
        'success': False,
        'error': message
    }
    if details:
        data['details'] = details

    return jsonify(data), status_code


def api_success(data: Optional[Dict[str, Any]] = None, message: Optional[str] = None) -> Tuple[Response, int]:
    """
    Create an API success response

    Args:
        data: Response data
        message: Optional success message

    Returns:
        (Flask response, status code)
    """
    response = {'success': True}
    if data:
        response['data'] = data
    if message:
        response['message'] = message
    return jsonify(response), 200

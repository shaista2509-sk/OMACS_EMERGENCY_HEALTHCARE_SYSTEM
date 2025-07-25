"""
Main application entry point for Emergency Healthcare System
"""

import os
from datetime import datetime
from app_init import create_app
from app.extensions import db, login_manager, socketio, migrate
from app.models import Patient, Emergency, NetworkSlice
from app.services.network_slicing_service import NetworkSlicingService
from ml_models.ml_service import HealthPredictionService
from flask import render_template

# Create Flask application
config_name = os.environ.get('FLASK_CONFIG', 'development')
app = create_app(config_name)


# Initialize core services
with app.app_context():
    # Database initialization
    try:
        db.create_all()
        print(f"Database initialized at {datetime.now().isoformat()}")
    except Exception as e:
        print(f"Database initialization failed: {str(e)}")
        raise

    # ML service initialization
    try:
        ml_service = HealthPredictionService()
        print(f"ML model loaded: {ml_service.model_path}")
    except Exception as e:
        print(f"ML service initialization failed: {e}")
        ml_service = None

    # Network slicing service
    try:
        network_slicer = NetworkSlicingService()
        print(f"Network slicing service initialized")
    except Exception as e:
        print(f"Network slicing service failed: {str(e)}")
        raise

    # Debug routes
    with app.app_context():
        print("\n=== Registered Routes ===")
        for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
            print(f"{'|'.join(rule.methods)} {rule.rule} -> {rule.endpoint}")


# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('errors/500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

    print(f"""
    ========================================================
    🚑 Emergency Healthcare System - Production v2.1 🚑
    ========================================================
    Timestamp: {datetime.now().isoformat()}
    API Base:   http://localhost:{port}/api
    Health:     http://localhost:{port}/health
    Patient Dashboard:  http://localhost:{port}/patient/dashboard
    Doctor Dashboard:  http://localhost:{port}/doctor/dashboard
    Authentication Dashboard: http://localhost:{port}/auth/login
    Debug Mode: {'Enabled' if debug_mode else 'Disabled'}
    Config:     {config_name.title()}
    Components:
      - Real-time Emergency Handling
      - Network Slicing (5G NR)
      - ML-Powered Diagnostics
      - Multi-User Dashboards
    ========================================================
    """)

    socketio.run(app,
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        use_reloader=True,
        log_output=True,
        allow_unsafe_werkzeug=True  # Development only
    )

# Updated app_init.py - Proper Flask-Login Configuration

from flask import Flask
from app.extensions import db, login_manager, socketio, migrate
from flask_cors import CORS
import os

def create_app(config_name='development'):
    """Create and configure Flask application"""
    app = Flask(__name__)

    # Load configuration
    try:
        from config import config
        app.config.from_object(config[config_name])
        print(f"Loaded {config_name} configuration")
    except ImportError as e:
        print(f"Configuration import error: {e}")
        # Fallback configuration
        app.config.update({
            'SQLALCHEMY_DATABASE_URI': 'mysql+pymysql://root:Karim%402510@localhost/emergency_health',
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'SECRET_KEY': 'Karim@2510',
            'ML_MODEL_PATH': os.path.join(os.path.dirname(__file__), 'ml_models', 'models')
        })

    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    socketio.init_app(app)
    migrate.init_app(app, db)
    CORS(app)

    # Configure login manager
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'

    # CRITICAL: User loader function with debugging
    @login_manager.user_loader
    def load_user(user_id):
        print(f"DEBUG: Loading user with ID: {user_id}")
        from app.models import Patient
        try:
            patient = Patient.query.get(int(user_id))
            print(f"DEBUG: Loaded patient: {patient}")
            return patient
        except Exception as e:
            print(f"DEBUG: Error loading user: {e}")
            return None

    # Register blueprints
    def register_blueprints():
        """Register all application blueprints"""
        try:
            # Import blueprints
            from app.routes.auth import auth_bp
            from app.routes.patient import patient_bp
            from app.routes.doctor import doctor_bp
            
            # Register blueprints
            app.register_blueprint(auth_bp, url_prefix='/auth')
            app.register_blueprint(patient_bp, url_prefix='/patient')
            app.register_blueprint(doctor_bp, url_prefix='/doctor')
            
            print("Blueprints registered successfully")
        except Exception as e:
            print(f"Blueprint registration failed: {str(e)}")

    register_blueprints()

    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {"status": "healthy", "service": "emergency_healthcare_system"}

    # API base endpoint
    @app.route('/api/')
    def api_base():
        return {"status": "API running", "version": "1.0.0"}

    return app

# Ensure models are imported for migrations
try:
    from app.models import Patient, Emergency, NetworkSlice
    print("Models imported successfully")
except ImportError as e:
    print(f"Model import warning: {str(e)}")

# Additional debugging function (optional)
def debug_session_info(app):
    """Debug function to check session configuration"""
    with app.app_context():
        print(f"DEBUG: SECRET_KEY set: {bool(app.config.get('SECRET_KEY'))}")
        print(f"DEBUG: Session cookie name: {app.config.get('SESSION_COOKIE_NAME', 'session')}")
        print(f"DEBUG: Login manager login_view: {login_manager.login_view}")
        print(f"DEBUG: Login manager configured: {login_manager._login_disabled is False}")
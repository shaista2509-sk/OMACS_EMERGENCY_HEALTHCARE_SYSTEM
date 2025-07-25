"""
Configuration settings for Emergency Healthcare System
"""

import os
from datetime import timedelta

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'emergency-healthcare-secret-key-2024'

    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql+pymysql://root:Karim%402510@localhost/emergency_health'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False

    # Flask-Login configuration
    REMEMBER_COOKIE_DURATION = timedelta(days=7)
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)

    # Flask-Mail configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER')
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')

    # ML Model configuration
    ML_MODEL_PATH = os.path.join(basedir, 'ml_models', 'models')
    PYTORCH_MODEL_PATH = os.path.join(ML_MODEL_PATH, 'groupnet.pth')
    ONNX_MODEL_PATH = os.path.join(ML_MODEL_PATH, 'groupnet.onnx')

    # Emergency system configuration
    EMERGENCY_RESPONSE_TIMEOUT = 300  # 5 minutes
    MAX_AMBULANCES_PER_EMERGENCY = 3

    # Network slicing configuration
    NETWORK_SLICE_PRIORITY = {
        'emergency': 1,
        'routine': 2,
        'maintenance': 3
    }

    # Rate limiting
    RATELIMIT_DEFAULT = "100 per hour"
    RATELIMIT_STORAGE_URL = "memory://"

    # O-RAN xApp configuration
    ORAN_RIC_URL = os.environ.get('ORAN_RIC_URL', 'http://localhost:8080')
    ORAN_E2_INTERFACE = os.environ.get('ORAN_E2_INTERFACE', 'localhost:36421')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql+pymysql://root:Karim%402510@localhost/emergency_health'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql+pymysql://root:Karim%402510@localhost/emergency_health'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False

class DockerConfig(Config):
    """Docker container configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql+pymysql://root:Karim%402510@localhost/emergency_health'


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'docker': DockerConfig,
    'default': DevelopmentConfig
}
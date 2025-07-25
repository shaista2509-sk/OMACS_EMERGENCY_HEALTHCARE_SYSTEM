import os
import sys
import warnings
from sqlalchemy import text

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Suppress SQLAlchemy warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from app_init import create_app, db
    from app.models import Patient, Emergency, Ambulance, Doctor
    print("Successfully imported Flask components")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure app_init.py exists and contains create_app() function")
    sys.exit(1)

def test_database_connection():
    """Test database connectivity and operations"""
    try:
        # Create Flask app with development config
        app = create_app('development')

        with app.app_context():
            print("Testing database connection...")

            # Test 1: Basic connectivity
            try:
                result = db.session.execute(text('SELECT 1'))
                print("Database connection successful!")
            except Exception as e:
                print(f"Database connection failed: {e}")
                return False

            # Test 2: Create tables
            try:
                db.create_all()
                print("Database tables created successfully!")
            except Exception as e:
                print(f"Table creation failed: {e}")
                return False

            # Test 3: Test model operations
            try:
                # Create test patient
                test_patient = Patient(
                    Name='Test Patient',
                    Age=30,
                    Gender='Male',
                    # Use the correct column name with space
                    Blood_Type='A+',
                    Medical_Condition='None',
                    Date_of_Admission=None,  # or a valid date if required
                    Doctor='Dr. Smith',
                    Hospital='City Hospital',
                    Insurance='ABC Insurance',
                    Billing_Amount=1000.00,
                    Room_Number='101A',
                    Admission_Type='Emergency',
                    Discharge_Date=None,  # or a valid date if required
                    medication='None',
                    Test_Result='Normal',
                    Date_of_Birth=None  # or a valid date if required
                )


                db.session.add(test_patient)
                db.session.commit()

                # Query the patient
                patient = Patient.query.filter_by(Name='Test Patient').first()
                if patient:
                    print("est patient created and retrieved successfully!")
                    # Clean up
                    db.session.delete(patient)
                    db.session.commit()
                else:
                    print("Failed to retrieve test patient")
                    return False

            except Exception as e:
                print(f"Model operation failed: {e}")
                db.session.rollback()
                return False

            # Test 4: Check ML model path
            try:
                ml_model_path = app.config.get('ML_MODEL_PATH')
                if ml_model_path and os.path.exists(ml_model_path):
                    print("ML model path configured correctly!")
                else:
                    print("ML model path not found (this is OK for now)")
            except Exception as e:
                print(f"ML model check warning: {e}")

            print("\nAll database tests passed!")
            return True

    except Exception as e:
        print(f"Critical error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Emergency Healthcare System - Database Test")
    print("=" * 50)

    success = test_database_connection()

    if success:
        print("\nDatabase setup is working correctly!")
        print("You can now proceed with Flask migrations and data population.")
    else:
        print("\nDatabase setup has issues. Please check the errors above.")
        sys.exit(1)
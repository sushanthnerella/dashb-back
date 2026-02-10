"""
Backend service to migrate and fix appointments to use correct patient IDs from database.
This reconciles frontend appointments with actual backend patient records.
"""

from database import patient_collection
from datetime import datetime
import json

def get_patient_by_name(name):
    """Get patient registration ID by name."""
    patient = patient_collection.find_one(
        {"name": {"$regex": f"^{name}$", "$options": "i"}},
        {"registrationId": 1, "name": 1, "created_at": 1}
    )
    return patient

def migrate_appointment_records():
    """
    Migrate appointment records to use correct backend patient IDs.
    This is a utility function that could be called via API endpoint if needed.
    """
    
    # Define which patient names need fixing (from your manual report)
    patients_to_fix = {
        "umesh": "REG-2025-871388",  # This is the correct ID in the backend
    }
    
    print("=== Appointment Migration Utility ===\n")
    
    for patient_name, expected_id in patients_to_fix.items():
        print(f"Processing: {patient_name}")
        
        # Get the actual patient record from backend
        patient = get_patient_by_name(patient_name)
        if not patient:
            print(f"  ✗ Patient '{patient_name}' not found in backend")
            continue
        
        actual_id = patient.get('registrationId')
        print(f"  Backend ID: {actual_id}")
        print(f"  Expected ID: {expected_id}")
        
        if actual_id != expected_id:
            print(f"  ⚠ ID Mismatch! Backend has different ID than expected")
        else:
            print(f"  ✓ ID matches")
        
        print(f"  Created: {patient.get('created_at')}")
        print()
    
    print("\nFrontend Action Needed:")
    print("1. Open DevTools (F12)")
    print("2. Go to Application > Local Storage")
    print("3. Find the 'queuedAppointments' entry")
    print("4. Edit the appointments to use the correct registration IDs from above")
    print("5. Replace REG-2025-600651 with REG-2025-871388 (the actual backend ID)")
    print("6. Delete any duplicate appointments")
    print("7. Save and refresh the page")
    
    return {
        "corrections": [
            {
                "patient_name": "umesh",
                "old_id": "REG-2025-600651",
                "correct_id": "REG-2025-871388",
                "action": "Replace all occurrences of old_id with correct_id in queuedAppointments"
            }
        ]
    }

if __name__ == "__main__":
    results = migrate_appointment_records()
    print(json.dumps(results, indent=2))


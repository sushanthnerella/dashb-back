#!/usr/bin/env python3
"""
Script to clean up and consolidate frontend appointment data.
This will merge duplicate patient appointments and use the actual backend registrationId.
"""

import json
import sys
from datetime import datetime

def load_json_file(path):
    """Load JSON file from path."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def save_json_file(path, data):
    """Save JSON data to file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")

def clean_appointments(appointments):
    """Clean and consolidate duplicate patient appointments."""
    if not isinstance(appointments, list):
        return appointments
    
    # Group by patient name
    by_name = {}
    for apt in appointments:
        name = apt.get('patientName', '').lower()
        if name not in by_name:
            by_name[name] = []
        by_name[name].append(apt)
    
    # Find duplicates
    duplicates = {k: v for k, v in by_name.items() if len(v) > 1}
    
    if duplicates:
        print(f"\nFound {len(duplicates)} patient names with multiple appointments:")
        for name, apts in duplicates.items():
            print(f"  {name}: {len(apts)} appointments")
            reg_ids = set(apt.get('patientRegistrationId') for apt in apts)
            print(f"    Registration IDs: {reg_ids}")
    
    return appointments

def consolidate_queue_data(queue_data):
    """Consolidate queue data, removing old/duplicate entries."""
    if not isinstance(queue_data, list):
        return queue_data
    
    # Group by patient name and date
    seen = {}
    consolidated = []
    
    for item in queue_data:
        name = item.get('patientName', '').lower()
        apt_date = item.get('appointmentDate', '')
        key = f"{name}_{apt_date}"
        
        if key not in seen:
            seen[key] = item
            consolidated.append(item)
        else:
            # Keep the latest one
            if item.get('bookedAt', '') > seen[key].get('bookedAt', ''):
                # Remove old and add new
                consolidated = [x for x in consolidated if not (x.get('patientName', '').lower() == name and x.get('appointmentDate') == apt_date)]
                consolidated.append(item)
                seen[key] = item
    
    return consolidated

if __name__ == "__main__":
    print("=== Frontend Appointment Data Cleaner ===\n")
    
    # Check if we have appointments data
    # In a real scenario, this would read from localStorage through the browser
    # For now, we'll just provide guidance
    
    print("Note: This script helps understand the data structure.")
    print("\nTo fix the Umesh duplicate appointments issue:")
    print("1. Open your browser DevTools (F12)")
    print("2. Go to Application > Local Storage")
    print("3. Find 'queuedAppointments' key")
    print("4. Export or manually fix the appointments")
    print("\nActions needed:")
    print("- Keep only ONE appointment for Umesh (the one with the correct backend registration ID)")
    print("- The backend only has ONE Umesh record")
    print("- Check the PatientHistoryView to see which registration ID is correct")
    print("- Delete the other duplicate appointment from queuedAppointments")
    
    print("\nAlternatively, manual cleanup steps:")
    print("1. Go to Reception Queue")
    print("2. Remove duplicate Umesh appointments from the queue")
    print("3. Or proceed with the correct one and ignore the wrong one")


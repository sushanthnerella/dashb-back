"""
Script to fix duplicate patient records (e.g., duplicate Umesh entries).
This consolidates multiple records of the same patient into one.
"""

from database import patient_collection
from datetime import datetime
import re

def find_duplicate_patients():
    """Find patients with the same name and/or email."""
    # Group by name
    pipeline = [
        {
            "$group": {
                "_id": {"$toLower": "$name"},
                "count": {"$sum": 1},
                "docs": {"$push": {"_id": "$_id", "registrationId": "$registrationId", "contactInfo": "$contactInfo", "created_at": "$created_at"}}
            }
        },
        {
            "$match": {"count": {"$gt": 1}}
        }
    ]
    
    duplicates = list(patient_collection.aggregate(pipeline))
    return duplicates

def merge_patient_records(primary_id, secondary_ids):
    """
    Merge multiple patient records, keeping the primary and deleting the rest.
    Args:
        primary_id: ObjectId of the primary record to keep
        secondary_ids: List of ObjectIds to merge into primary
    """
    # Get the primary document
    primary_doc = patient_collection.find_one({"_id": primary_id})
    if not primary_doc:
        print(f"Primary document not found: {primary_id}")
        return False
    
    # For each secondary document, merge its encounters/visits into primary
    for secondary_id in secondary_ids:
        secondary_doc = patient_collection.find_one({"_id": secondary_id})
        if not secondary_doc:
            continue
        
        # Merge encounters if they exist
        secondary_encounters = secondary_doc.get("encounters", [])
        if secondary_encounters:
            primary_encounters = primary_doc.get("encounters", [])
            primary_encounters.extend(secondary_encounters)
            primary_doc["encounters"] = primary_encounters
        
        # Update the merged document
        patient_collection.update_one(
            {"_id": primary_id},
            {
                "$set": {
                    "encounters": primary_doc.get("encounters", []),
                    "merged_from": secondary_ids,
                    "last_merged_at": datetime.now().isoformat()
                }
            }
        )
        
        # Delete the secondary document
        patient_collection.delete_one({"_id": secondary_id})
        print(f"Deleted secondary record: {secondary_id}")
    
    return True

def fix_duplicate_names(name_query):
    """Fix duplicate records for a specific patient name."""
    print(f"\nSearching for '{name_query}' records...")
    
    # Find all records with this name
    records = list(patient_collection.find(
        {"name": {"$regex": f"^{re.escape(name_query)}$", "$options": "i"}},
        {"_id": 1, "name": 1, "registrationId": 1, "contactInfo": 1, "created_at": 1, "encounters": 1}
    ))
    
    if len(records) < 2:
        print(f"Found {len(records)} record(s) for '{name_query}'. No merge needed.")
        return
    
    print(f"Found {len(records)} records for '{name_query}':")
    for i, rec in enumerate(records):
        encounters = rec.get("encounters", [])
        print(f"  [{i}] ID: {rec['_id']}")
        print(f"      Registration ID: {rec.get('registrationId')}")
        print(f"      Created: {rec.get('created_at')}")
        print(f"      Encounters: {len(encounters)}")
    
    # Keep the one with the most encounters (or earliest created_at if tied)
    primary = min(records, key=lambda x: (
        -len(x.get("encounters", [])),  # Negative to sort descending
        x.get("created_at", "")  # Earliest first
    ))
    
    secondary_records = [r for r in records if r["_id"] != primary["_id"]]
    
    print(f"\nMerging into primary record: {primary['registrationId']}")
    print(f"Deleting secondary records: {[r.get('registrationId') for r in secondary_records]}")
    
    # Perform the merge
    merge_patient_records(
        primary["_id"],
        [r["_id"] for r in secondary_records]
    )
    
    print(f"Merge for '{name_query}' complete!")

def fix_umesh_duplicates():
    """Specifically fix Umesh duplicate records."""
    fix_duplicate_names("umesh")

if __name__ == "__main__":
    print("=== Patient Duplicate Fixer ===\n")
    
    print("Step 1: Finding all duplicate patients...")
    duplicates = find_duplicate_patients()
    
    if duplicates:
        print(f"Found {len(duplicates)} names with duplicates:")
        for dup in duplicates:
            print(f"  - {dup['_id']}: {dup['count']} records")
            for doc in dup['docs']:
                print(f"      {doc['registrationId']}")
    else:
        print("No duplicates found.")
    
    print("\nStep 2: Fixing Umesh...")
    fix_duplicate_names("umesh")
    
    print("\nStep 3: Fixing Shashi...")
    fix_duplicate_names("shashi")
    
    print("\nDone!")

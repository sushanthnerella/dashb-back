#!/usr/bin/env python3
"""
Script to delete all pharmacy medicines from the database
WARNING: This will permanently remove all medicine records
"""

import sys
from datetime import datetime
from database import pharmacy_collection

def delete_all_medicines():
    """Delete all medicines from the database"""
    
    print("\n" + "="*60)
    print("PHARMACY MEDICINE DELETION TOOL")
    print("="*60)
    print("\n⚠️  WARNING: This will permanently delete ALL medicines!\n")
    
    # Get current count
    total_count = pharmacy_collection.count_documents({})
    print(f"Current medicines in database: {total_count}")
    
    if total_count == 0:
        print("\n✓ Database is already empty. Nothing to delete.\n")
        return
    
    # Confirm deletion
    response = input("\nAre you SURE you want to delete ALL medicines? Type 'YES' to confirm: ")
    
    if response.upper() != 'YES':
        print("\n✗ Deletion cancelled.")
        return
    
    try:
        # Delete all medicines
        result = pharmacy_collection.delete_many({})
        
        print("\n" + "="*60)
        print("DELETION SUMMARY")
        print("="*60)
        print(f"✓ Deleted: {result.deleted_count} medicines")
        print(f"✓ Timestamp: {datetime.now().isoformat()}")
        print("="*60 + "\n")
        
        # Verify deletion
        remaining = pharmacy_collection.count_documents({})
        print(f"Remaining medicines in database: {remaining}\n")
        
        if remaining == 0:
            print("✓ Database is now empty!\n")
        else:
            print(f"⚠️  {remaining} medicines still in database\n")
        
    except Exception as e:
        print(f"\n✗ Error during deletion: {str(e)}\n")
        import traceback
        traceback.print_exc()

def delete_recently_imported(minutes=5):
    """Delete medicines imported in the last N minutes"""
    
    from datetime import timedelta
    
    print("\n" + "="*60)
    print("PHARMACY MEDICINE DELETION TOOL - RECENT ONLY")
    print("="*60)
    
    # Calculate cutoff time
    cutoff_time = (datetime.utcnow() - timedelta(minutes=minutes)).isoformat()
    
    print(f"\nDeleting medicines imported after: {cutoff_time}\n")
    
    # Get count of recent medicines
    recent_count = pharmacy_collection.count_documents({
        "created_at": {"$gte": cutoff_time}
    })
    
    total_count = pharmacy_collection.count_documents({})
    
    print(f"Total medicines in database: {total_count}")
    print(f"Recently imported medicines: {recent_count}\n")
    
    if recent_count == 0:
        print("✓ No recently imported medicines found.\n")
        return
    
    # Confirm deletion
    response = input(f"Delete {recent_count} recently imported medicines? Type 'YES' to confirm: ")
    
    if response.upper() != 'YES':
        print("\n✗ Deletion cancelled.")
        return
    
    try:
        # Delete recent medicines
        result = pharmacy_collection.delete_many({
            "created_at": {"$gte": cutoff_time}
        })
        
        print("\n" + "="*60)
        print("DELETION SUMMARY")
        print("="*60)
        print(f"✓ Deleted: {result.deleted_count} medicines")
        print(f"✓ Timestamp: {datetime.now().isoformat()}")
        print("="*60 + "\n")
        
        # Verify deletion
        remaining = pharmacy_collection.count_documents({})
        print(f"Remaining medicines in database: {remaining}\n")
        
    except Exception as e:
        print(f"\n✗ Error during deletion: {str(e)}\n")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    
    if len(sys.argv) > 1 and sys.argv[1] == "--recent":
        # Delete only recently imported medicines
        minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        delete_recently_imported(minutes)
    else:
        # Delete all medicines
        delete_all_medicines()

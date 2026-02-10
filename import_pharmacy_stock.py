#!/usr/bin/env python3
"""
Script to import pharmacy medicines from Excel file into MongoDB
- Smart category detection from product name
- Uses environment variables for DB connection
- Handles duplicates by name + batch_number
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Excel file path
EXCEL_FILE_PATH = r"E:\New folder (7)\dashb (1)\dashb\PHAMACY STOCK.xls"

# Get DB connection from environment
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")

if not MONGO_URI or not DATABASE_NAME:
    print("❌ Error: MONGO_URI and DATABASE_NAME environment variables must be set.")
    print("   Please check your .env file in the backend folder.")
    sys.exit(1)


def get_database_connection():
    """Establish connection to MongoDB using environment variables"""
    try:
        # Use certifi for SSL certificates for Atlas connections
        client_kwargs = {}
        if "mongodb.net" in MONGO_URI or "mongodb+srv://" in MONGO_URI:
            client_kwargs["tlsCAFile"] = certifi.where()
        
        client = MongoClient(MONGO_URI, **client_kwargs)
        db = client[DATABASE_NAME]
        
        # Test connection
        client.admin.command('ping')
        print(f"✅ Connected to MongoDB: {DATABASE_NAME}")
        
        return db["pharmacy_medicines"]
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        sys.exit(1)


def detect_category(product_name, excel_type):
    """
    Smart category detection from product name
    Falls back to Excel 'Type' column if no keyword found
    """
    if not product_name:
        return excel_type if excel_type else "Uncategorized"
    
    name_upper = str(product_name).upper()
    
    # Detection rules (order matters - more specific first)
    if "EYE DROP" in name_upper or "DROPS" in name_upper or "DROP" in name_upper or " E/D" in name_upper or "E/D" in name_upper:
        return "Drops"
    
    if "TABLET" in name_upper or " TAB " in name_upper or " TAB" in name_upper or "TAB " in name_upper or name_upper.endswith(" TAB"):
        return "Tablet"
    
    if "CAPSULE" in name_upper or "CAPS" in name_upper:
        return "Capsules"
    
    if "OINTMENT" in name_upper:
        return "Ointment"
    
    if "INJECTION" in name_upper or " INJ " in name_upper or " INJ" in name_upper:
        return "Injection"
    
    if "GEL" in name_upper:
        return "Drops"  # Eye gels are typically in Drops category
    
    if "SYRUP" in name_upper:
        return "Syrup"
    
    if "STRIPS" in name_upper or "WIPES" in name_upper:
        return "Others"
    
    # Fallback to Excel Type column
    if excel_type and str(excel_type).strip():
        return str(excel_type).strip()
    
    return "Uncategorized"


def parse_expiry_date(exp_date_str):
    """
    Parse expiry date from Excel format (e.g., '9/2026', '12/2027')
    Returns in format 'YYYY-MM' or None if parsing fails
    """
    if not exp_date_str or pd.isna(exp_date_str) or str(exp_date_str).strip() == '':
        return None
    
    try:
        # Handle format like '9/2026' -> '2026-09'
        parts = str(exp_date_str).strip().split('/')
        if len(parts) == 2:
            month, year = parts
            return f"{year}-{int(month):02d}"
    except Exception as e:
        print(f"  ⚠ Warning: Could not parse date '{exp_date_str}': {e}")
    
    return str(exp_date_str)


def parse_price(price_val):
    """Parse price as float, handle empty/NaN values"""
    if price_val is None or pd.isna(price_val) or str(price_val).strip() == '':
        return 0.0
    try:
        return float(price_val)
    except:
        return 0.0


def parse_stock(stock_val):
    """Parse stock as integer, handle empty/NaN values"""
    if stock_val is None or pd.isna(stock_val) or str(stock_val).strip() == '':
        return 0
    try:
        return int(float(stock_val))
    except:
        return 0


def import_medicines():
    """Main import function"""
    print("\n" + "=" * 70)
    print("  PHARMACY STOCK EXCEL IMPORTER")
    print("=" * 70)
    print(f"\n📁 Excel File: {EXCEL_FILE_PATH}")
    print(f"🗄️  Database: {DATABASE_NAME}")
    print()
    
    # Check if file exists
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"❌ Error: Excel file not found at: {EXCEL_FILE_PATH}")
        sys.exit(1)
    
    # Connect to database
    pharmacy_collection = get_database_connection()
    
    # Read Excel file
    print(f"\n📖 Reading Excel file...")
    df = None
    
    try:
        # Try xlrd engine first for .xls files
        try:
            df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=0, engine='xlrd')
            print("✅ Successfully read as Excel (.xls)")
        except:
            # Try default engine for .xlsx
            try:
                df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=0)
                print("✅ Successfully read as Excel (.xlsx)")
            except:
                # Try reading as HTML (some .xls files are actually HTML)
                df_list = pd.read_html(EXCEL_FILE_PATH)
                if df_list:
                    df = df_list[0]
                    print("✅ Successfully read as HTML table")
    except Exception as e:
        print(f"❌ Failed to read Excel file: {e}")
        sys.exit(1)
    
    if df is None or df.empty:
        print("❌ No data found in Excel file")
        sys.exit(1)
    
    print(f"📊 Total rows in Excel: {len(df)}")
    
    # Display columns found
    print("\n📋 Columns detected in Excel:")
    for col in df.columns:
        print(f"   • {col}")
    
    # Counters
    imported = 0
    updated = 0
    skipped = 0
    errors = 0
    
    print("\n" + "-" * 70)
    print("  PROCESSING RECORDS")
    print("-" * 70 + "\n")
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            # Get values from row
            product_name = row.get('Product Name', '')
            excel_type = row.get('Type', '')
            exp_date = row.get('Exp Date', '')
            batch_number = row.get('Batch', '')
            stock = row.get('Stock', 0)
            vendor_name = row.get('Vendor Name', '')
            price = row.get('Price', 0)
            
            # Skip empty rows
            if pd.isna(product_name) or str(product_name).strip() == '':
                skipped += 1
                continue
            
            # Clean product name
            clean_name = str(product_name).strip()
            
            # Detect category from name
            detected_category = detect_category(clean_name, excel_type)
            
            # Prepare medicine document
            medicine = {
                "name": clean_name,
                "category": detected_category,
                "price": parse_price(price),
                "stock": parse_stock(stock),
                "description": "",
                "manufacturer": str(vendor_name).strip() if pd.notna(vendor_name) else "",
                "batch_number": str(batch_number).strip() if pd.notna(batch_number) else "",
                "expiry_date": parse_expiry_date(exp_date),
                "reorder_level": 10,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            
            # Check if medicine already exists by name and batch
            existing = pharmacy_collection.find_one({
                "name": medicine["name"],
                "batch_number": medicine["batch_number"]
            })
            
            if existing:
                # Update existing medicine
                pharmacy_collection.update_one(
                    {"_id": existing["_id"]},
                    {
                        "$set": {
                            "category": medicine["category"],
                            "price": medicine["price"],
                            "stock": medicine["stock"],
                            "expiry_date": medicine["expiry_date"],
                            "manufacturer": medicine["manufacturer"],
                            "updated_at": datetime.utcnow(),
                        }
                    }
                )
                updated += 1
                print(f"⟳  Row {idx + 2}: Updated '{clean_name}' | Batch: {batch_number} | Category: {detected_category}")
            else:
                # Insert new medicine
                pharmacy_collection.insert_one(medicine)
                imported += 1
                print(f"✅ Row {idx + 2}: Imported '{clean_name}' | Batch: {batch_number} | Category: {detected_category}")
            
        except Exception as e:
            errors += 1
            print(f"❌ Row {idx + 2}: Error - {str(e)}")
            continue
    
    # Get total count in database
    total_in_db = pharmacy_collection.count_documents({})
    
    # Print summary
    print("\n" + "=" * 70)
    print("  IMPORT SUMMARY")
    print("=" * 70)
    print(f"  ✅ Imported (New):     {imported} medicines")
    print(f"  ⟳  Updated (Existing): {updated} medicines")
    print(f"  ⊘  Skipped (Empty):    {skipped} rows")
    print(f"  ❌ Errors:             {errors} rows")
    print(f"  ─────────────────────────────────────")
    print(f"  📊 Total Processed:    {imported + updated} records")
    print(f"  🗄️  Total in Database:  {total_in_db} medicines")
    print("=" * 70 + "\n")
    
    return {
        "imported": imported,
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
        "total_in_db": total_in_db
    }


if __name__ == "__main__":
    try:
        result = import_medicines()
        if result["imported"] > 0 or result["updated"] > 0:
            print("🎉 Import completed successfully!\n")
        else:
            print("⚠️  No records were imported or updated.\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  Import cancelled by user.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

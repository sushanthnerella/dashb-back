#!/usr/bin/env python3
"""
Script to import pharmacy medicines from Excel file or HTML table into MongoDB
Maps Excel/HTML columns to database fields
"""

import pandas as pd
import sys
from datetime import datetime
from database import pharmacy_collection

def parse_expiry_date(exp_date_str):
    """
    Parse expiry date from Excel format (e.g., '9/2026', '12/2027')
    Returns in format 'YYYY-MM' or the original string if parsing fails
    """
    if not exp_date_str or exp_date_str == '':
        return None
    
    try:
        # Handle format like '9/2026' -> '2026-09'
        parts = str(exp_date_str).strip().split('/')
        if len(parts) == 2:
            month, year = parts
            return f"{year}-{int(month):02d}"
    except Exception as e:
        print(f"⚠ Warning: Could not parse date '{exp_date_str}': {e}")
    
    return str(exp_date_str)

def parse_price(price_str):
    """Parse price as float, handle empty strings"""
    if not price_str or price_str == '':
        return 0.0
    try:
        return float(price_str)
    except:
        return 0.0

def parse_stock(stock_str):
    """Parse stock as integer, handle empty strings"""
    if not stock_str or stock_str == '':
        return 0
    try:
        return int(float(stock_str))
    except:
        return 0

def import_medicines_from_excel(file_path):
    """
    Import medicines from Excel file or HTML table into MongoDB
    Supports: .xls, .xlsx, .html formats
    Auto-detects format by trying multiple readers
    
    Args:
        file_path: Path to the Excel or HTML file
    """
    print(f"Starting import from: {file_path}\n")
    
    df = None
    
    try:
        # Try to read as Excel first
        try:
            df = pd.read_excel(file_path, sheet_name=0, engine='xlrd')
            print("✓ Successfully read as Excel (.xls)")
        except:
            try:
                df = pd.read_excel(file_path, sheet_name=0)
                print("✓ Successfully read as Excel (.xlsx)")
            except:
                # Try HTML
                try:
                    print("Excel read failed, trying HTML...")
                    df_list = pd.read_html(file_path)
                    if df_list:
                        df = df_list[0]
                        print("✓ Successfully read as HTML table")
                except Exception as e:
                    print(f"✗ Could not read file in any format: {str(e)}")
                    return None
        
        if df is None:
            print("✗ Failed to read file")
            return None
        
        print(f"Total rows in file: {len(df)}\n")
        
        # Column mapping - handle variations in column names
        column_map = {
            'Product Name': 'name',
            'Type': 'category',
            'Exp Date': 'expiry_date',
            'Batch': 'batch_number',
            'Stock': 'stock',
            'Price': 'price',
            'Vendor Name': 'manufacturer',
        }
        
        # Check for columns
        print("Column names in file:")
        for col in df.columns:
            print(f"  - {col}")
        print()
        
        imported = 0
        skipped = 0
        updated = 0
        errors = 0
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Get values from row
                product_name = row.get('Product Name', '')
                medicine_type = row.get('Type', '')
                exp_date = row.get('Exp Date', '')
                batch_number = row.get('Batch', '')
                stock = row.get('Stock', 0)
                vendor_name = row.get('Vendor Name', '')
                price = row.get('Price', 0)
                
                # Skip empty rows
                if pd.isna(product_name) or str(product_name).strip() == '':
                    skipped += 1
                    continue
                
                # Prepare medicine document
                medicine = {
                    "name": str(product_name).strip(),
                    "category": str(medicine_type).strip() if pd.notna(medicine_type) else "Uncategorized",
                    "price": parse_price(price),
                    "stock": parse_stock(stock),
                    "description": "",  # Not in Excel
                    "manufacturer": str(vendor_name).strip() if pd.notna(vendor_name) else "",
                    "batch_number": str(batch_number).strip() if pd.notna(batch_number) else "",
                    "expiry_date": parse_expiry_date(exp_date),
                    "reorder_level": 10,  # Default reorder level
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
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
                                "price": medicine["price"],
                                "stock": medicine["stock"],
                                "expiry_date": medicine["expiry_date"],
                                "manufacturer": medicine["manufacturer"],
                                "updated_at": datetime.utcnow().isoformat(),
                            }
                        }
                    )
                    updated += 1
                    print(f"✓ Row {idx + 2}: Updated '{product_name}' (Batch: {batch_number}, Stock: {stock})")
                else:
                    # Insert new medicine
                    result = pharmacy_collection.insert_one(medicine)
                    imported += 1
                    print(f"✓ Row {idx + 2}: Imported '{product_name}' (Batch: {batch_number}, Stock: {stock})")
                
            except Exception as e:
                errors += 1
                print(f"✗ Row {idx + 2}: Error - {str(e)}")
                continue
        
        # Summary
        print("\n" + "="*60)
        print("IMPORT SUMMARY")
        print("="*60)
        print(f"✓ Imported (New):    {imported} medicines")
        print(f"⟳ Updated (Existing): {updated} medicines")
        print(f"⊘ Skipped (Empty):   {skipped} rows")
        print(f"✗ Errors:            {errors} rows")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Total Processed:     {imported + updated + errors} rows")
        print("="*60)
        
        # Verify total medicines in collection
        total_medicines = pharmacy_collection.count_documents({})
        print(f"\nTotal medicines in database: {total_medicines}\n")
        
        return {
            "imported": imported,
            "updated": updated,
            "skipped": skipped,
            "errors": errors,
            "total": total_medicines
        }
        
    except FileNotFoundError:
        print(f"✗ Error: File not found: {file_path}")
        print(f"  Make sure the file exists at: {file_path}")
        return None
    except Exception as e:
        print(f"✗ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Excel file path
    excel_file = "PHAMACY STOCK.xls"
    
    # You can also pass file path as command line argument
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    
    print("\n" + "="*60)
    print("PHARMACY MEDICINE EXCEL IMPORTER")
    print("="*60 + "\n")
    
    result = import_medicines_from_excel(excel_file)
    
    if result:
        print("\n✓ Import completed successfully!")
    else:
        print("\n✗ Import failed!")
        sys.exit(1)

# Pharmacy Medicine Excel Import Guide

## Overview
This script imports pharmacy medicine data from an Excel file into MongoDB database.

## File Mapping

| Excel Column | Database Field | Notes |
|--------------|----------------|-------|
| Product Name | `name` | Medicine product name |
| Type | `category` | Medicine type (Drops, Tablet, etc.) |
| Exp Date | `expiry_date` | Format: 9/2026 → stored as 2026-09 |
| Batch | `batch_number` | Batch identifier |
| Stock | `stock` | Current stock quantity |
| Price | `price` | Selling price (MRP used as reference) |
| Vendor Name | `manufacturer` | Supplier/vendor name |
| Purchase Date | - | Not imported |
| Invoice No | - | Not imported |
| Transfer Id | - | Not imported |
| MRP | - | Reference only |
| BranchName | - | Not imported |

## Database Field Defaults
- `description`: Empty string (can be updated manually)
- `reorder_level`: 10 (default minimum stock)
- `created_at`: Current timestamp (new medicines only)
- `updated_at`: Current timestamp

## Prerequisites

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install openpyxl if not already installed
pip install openpyxl
```

## Usage

### Method 1: Default filename
```bash
python import_pharmacy_excel.py
# Looks for: PHAMACY STOCK.xls in the backend directory
```

### Method 2: Custom file path
```bash
python import_pharmacy_excel.py "path/to/your/file.xls"
```

### Example:
```bash
cd backend
.\venv\Scripts\Activate.ps1
python import_pharmacy_excel.py
```

## What the Script Does

1. **Reads Excel file** from specified path
2. **Validates data** - skips empty rows
3. **Parses fields** - converts dates, prices, stock quantities
4. **Checks for duplicates** - matches by name + batch_number
5. **Updates existing** - if medicine with same name & batch exists
6. **Imports new** - if medicine doesn't exist
7. **Generates report** - shows imported, updated, skipped, errors

## Script Output

```
============================================================
PHARMACY MEDICINE EXCEL IMPORTER
============================================================

Starting import from: PHAMACY STOCK.xls

Sheet name: Sheet 1
Total rows: 99

✓ Row 2: Imported '4QUIN EYE DROPS' (Batch: A4094, Stock: 4)
✓ Row 3: Imported 'AMPLINAK' (Batch: 121793, Stock: 30)
...

============================================================
IMPORT SUMMARY
============================================================
✓ Imported (New):     95 medicines
⟳ Updated (Existing):  0 medicines
⊘ Skipped (Empty):    1 rows
✗ Errors:             0 rows
━━━━━━━━━━━━━━━━━━━━━━
Total Processed:      96 rows
============================================================

Total medicines in database: 95

✓ Import completed successfully!
```

## Important Notes

### Duplicate Handling
- If a medicine with the **same name AND batch number** exists:
  - **Updates**: price, stock, expiry_date, manufacturer
  - **Preserves**: _id, created_at (keeps original creation time)
  - Updates: updated_at (to current time)

- If only name matches but batch is different:
  - Treats as **NEW medicine** (different batches are separate records)

### Date Format
- Excel format: `9/2026` (Month/Year)
- Database format: `2026-09` (YYYY-MM)
- Empty dates: Stored as `None`

### Price/Stock Parsing
- Handles empty values (defaults to 0)
- Converts text to numbers
- Logs warnings for unparseable values

## Troubleshooting

### "File not found" error
```
✓ Check file is in backend directory
✓ Verify filename matches exactly
✓ Use full path if file is elsewhere
```

### "openpyxl not found" error
```bash
pip install openpyxl
```

### MongoDB connection error
```
✓ Ensure MongoDB is running
✓ Check MONGO_URI in .env file
✓ Verify DATABASE_NAME is set
```

## Verification

After import, verify in MongoDB:

```bash
# Connect to MongoDB
mongosh

# Check medicines count
use hospital_db
db.pharmacy_medicines.count()

# View imported medicines
db.pharmacy_medicines.find().limit(5)

# Check specific medicine
db.pharmacy_medicines.findOne({"name": "4QUIN EYE DROPS"})
```

## Rollback

If import needs to be reversed:

```python
# Connect to MongoDB and remove imported batch
from database import pharmacy_collection
from datetime import datetime

# Remove medicines imported on specific date
result = pharmacy_collection.delete_many({
    "created_at": {"$gte": "2026-01-16T00:00:00"}
})
print(f"Deleted: {result.deleted_count} medicines")
```

## Performance

- Typical import time: 1-2 seconds for 100 medicines
- Database operations: Optimized with single queries for duplicates
- Memory usage: Minimal (streams row by row)

---

**Last Updated:** 2026-01-16
**Script Version:** 1.0

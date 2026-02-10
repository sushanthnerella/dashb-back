# Quick Start: Import Pharmacy Medicines

## Step 1: Prepare Excel File
- File should be: `PHAMACY STOCK.xls` (or your custom name)
- Place it in: `backend/` directory
- Format: First row = headers, Data starts from row 2

## Step 2: Activate Virtual Environment
```powershell
cd backend
.\venv\Scripts\Activate.ps1
```

## Step 3: Run Import Script
```powershell
# Using default filename
python import_pharmacy_excel.py

# OR using custom file path
python import_pharmacy_excel.py "C:\path\to\file.xls"
```

## Step 4: Check Results
Monitor output for:
- ✓ Number of medicines imported
- ⟳ Number updated
- ⊘ Number skipped
- ✗ Any errors

## Data Mapping (Excel → Database)

```
Excel Column          →  Database Field
────────────────────────────────────
Product Name          →  name
Type                  →  category
Exp Date (9/2026)     →  expiry_date (2026-09)
Batch                 →  batch_number
Stock                 →  stock
Price                 →  price
Vendor Name           →  manufacturer
[Auto]                →  description (empty)
[Auto]                →  reorder_level (10)
[Auto]                →  created_at (timestamp)
[Auto]                →  updated_at (timestamp)
```

## Example Output

```
============================================================
PHARMACY MEDICINE EXCEL IMPORTER
============================================================

Starting import from: PHAMACY STOCK.xls

✓ Row 2: Imported '4QUIN EYE DROPS' (Batch: A4094, Stock: 4)
✓ Row 3: Imported 'AMPLINAK' (Batch: 121793, Stock: 30)
✓ Row 4: Imported 'ANAWIN' (Batch: SM107235, Stock: 14)
...

============================================================
IMPORT SUMMARY
============================================================
✓ Imported (New):     95 medicines
⟳ Updated (Existing):  4 medicines
⊘ Skipped (Empty):    0 rows
✗ Errors:             0 rows
============================================================

Total medicines in database: 99

✓ Import completed successfully!
```

## Key Features

✓ **Duplicate Detection**: Checks by name + batch_number
✓ **Smart Updates**: Updates existing medicines with new data
✓ **Error Handling**: Logs and continues on individual errors
✓ **Data Validation**: Parses dates, prices, stock quantities
✓ **Timestamps**: Auto-generates created_at and updated_at

## Verify Import

```powershell
# Connect to MongoDB and check
mongosh

# In mongosh shell:
use hospital_db
db.pharmacy_medicines.count()
db.pharmacy_medicines.findOne({"name": "4QUIN EYE DROPS"})
```

## Need Help?

See: `IMPORT_PHARMACY_README.md` for detailed documentation

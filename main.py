from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import random
import string
import os
import uuid
from datetime import datetime
import re
from passlib.context import CryptContext
from bson import ObjectId

# Fix for passlib/bcrypt compatibility issue
import bcrypt
if not hasattr(bcrypt, "__about__"):
    class About:
        __version__ = getattr(bcrypt, "__version__", "4.0.0")
    bcrypt.__about__ = About

from database import (
    patient_collection, 
    user_collection, 
    db, 
    pharmacy_collection, 
    pharmacy_billing_collection, 
    coupon_quota_collection,
    billing_cases_collection,
    surgery_packages_collection,
    billing_invoices_collection,
    initial_surgery_bills_collection,
    final_surgery_bills_collection,
    slit_lamp_collection
)
from models import (
    NewPatient,
    PatientInDB,
    ReceptionistEncounter,
    Demographics,
    ContactInfo,
    EmergencyContact,
    History,
    InitialComplaint,
    BillingCase,
    BillingStage,
    NewUser,
    UserInDB,
    ALLOWED_ROLES,
    InvestigationsModel,
    DoctorEncounterModel,
    NewSurgeryPackage,
    SurgeryPackageInDB,
    UpdateSurgeryPackage,
)
import database  # ensure database.py is loaded
from database import patient_collection  # re-import for clarity

app = FastAPI()

# Import SaaS endpoints
from saas_endpoints import router as saas_router
app.include_router(saas_router)


@app.post("/evaluate-reading")
async def evaluate_reading(payload: dict = Body(...)):
    """Evaluate a clinical reading and return a severity color.
    Expected payload: { field: str, value: str, unit?: str, registration_id?: str }
    Only evaluates explicitly numeric fields (BP, pulse, sugar, IOP, SpO2, etc).
    Returns no severity for free-text fields (complaint, medical condition, etc).
    """
    import json

    field = (payload.get("field") or "").strip().lower()
    value_raw = (payload.get("value") or "").strip()

    # Return no evaluation for empty values
    if not value_raw:
        return {"severity": None, "score": 0, "message": ""}

    # load env-configured thresholds and external API settings
    EVAL_API_URL = os.getenv("EVAL_API_URL")
    EVAL_API_KEY = os.getenv("EVAL_API_KEY")
    thresholds = {}
    try:
        t = os.getenv("EVAL_THRESHOLDS")
        if t:
            thresholds = json.loads(t)
    except Exception:
        thresholds = {}

    # Forward to external evaluator if configured (best-effort)
    if EVAL_API_URL:
        try:
            try:
                import httpx
                headers = {"Content-Type": "application/json"}
                if EVAL_API_KEY:
                    headers["Authorization"] = f"Bearer {EVAL_API_KEY}"
                r = httpx.post(EVAL_API_URL, json={"field": field, "value": value_raw}, headers=headers, timeout=5.0)
                if r.status_code == 200:
                    jr = r.json()
                    if jr.get("severity"):
                        return {"severity": jr.get("severity"), "score": jr.get("score", 0), "message": jr.get("message", "")}
            except Exception:
                try:
                    import requests
                    headers = {"Content-Type": "application/json"}
                    if EVAL_API_KEY:
                        headers["Authorization"] = f"Bearer {EVAL_API_KEY}"
                    r = requests.post(EVAL_API_URL, json={"field": field, "value": value_raw}, headers=headers, timeout=5.0)
                    if r.status_code == 200:
                        jr = r.json()
                        if jr.get("severity"):
                            return {"severity": jr.get("severity"), "score": jr.get("score", 0), "message": jr.get("message", "")}
                except Exception:
                    pass
        except Exception:
            pass

    # local fallback evaluation
    def parse_number(s: str):
        """Extract the first numeric value from a string (handles commas, spaces)."""
        # Remove commas and extra spaces
        cleaned = s.replace(",", "").replace(" ", "")
        m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        return float(m.group(0)) if m else None

    def apply_thresholds_for_key(key: str, numeric: float = None, systolic: int = None):
        cfg = thresholds.get(key) if isinstance(thresholds, dict) else None
        if not cfg:
            return None
        # BP-like structure
        if systolic is not None and isinstance(cfg.get("systolic"), dict):
            s_cfg = cfg.get("systolic")
            if systolic >= s_cfg.get("red", 9999):
                return ("red", f"High systolic {systolic}")
            if systolic >= s_cfg.get("yellow", 9999):
                return ("yellow", f"Elevated systolic {systolic}")
            return ("green", f"Systolic {systolic}")
        # generic numeric thresholds
        if numeric is not None and isinstance(cfg.get("red"), (int, float)):
            if numeric >= cfg.get("red"):
                return ("red", f"High value ({numeric})")
            if numeric >= cfg.get("yellow"):
                return ("yellow", f"Elevated value ({numeric})")
            return ("green", f"Value {numeric}")
        return None

    # Try to match a key in thresholds by first token
    key_token = field.split()[0] if field else ""

    severity = None
    message = ""

    # === NUMERIC FIELDS ONLY ===
    # BP special handling (systolic/diastolic)
    if "bp" in field or "blood" in field or "pressure" in field:
        m = re.search(r"(\d{2,3})(?:\s*/\s*(\d{2,3}))?", value_raw)
        if m:
            syst = int(m.group(1))
            thr = apply_thresholds_for_key(key_token, systolic=syst)
            if thr:
                return {"severity": thr[0], "score": 0, "message": thr[1]}
            # fallback local rules
            if syst >= 160:
                severity = "red"
                message = f"High systolic BP ({syst})"
            elif syst >= 140:
                severity = "yellow"
                message = f"Elevated systolic BP ({syst})"
            else:
                severity = "green"
                message = f"Systolic BP {syst}"
        else:
            # Cannot parse BP, no color
            return {"severity": None, "score": 0, "message": ""}

    elif "pulse" in field or "heart" in field or "bpm" in field:
        v = parse_number(value_raw)
        if v is None:
            return {"severity": None, "score": 0, "message": ""}
        thr = apply_thresholds_for_key(key_token, numeric=v)
        if thr:
            return {"severity": thr[0], "score": 0, "message": thr[1]}
        if v >= 120:
            severity = "red"; message = f"Tachycardia ({v} bpm)"
        elif v >= 100:
            severity = "yellow"; message = f"High pulse ({v} bpm)"
        else:
            severity = "green"; message = f"Pulse {v} bpm"

    elif any(x in field for x in ["sugar", "glucose", "bs", "blood sugar"]):
        v = parse_number(value_raw)
        if v is None:
            return {"severity": None, "score": 0, "message": ""}
        thr = apply_thresholds_for_key(key_token, numeric=v)
        if thr:
            return {"severity": thr[0], "score": 0, "message": thr[1]}
        if v >= 200:
            severity = "red"; message = f"High blood sugar ({v})"
        elif v >= 140:
            severity = "yellow"; message = f"Elevated blood sugar ({v})"
        else:
            severity = "green"; message = f"Blood sugar {v}"

    elif "iop" in field or "intra" in field:
        v = parse_number(value_raw)
        if v is None:
            return {"severity": None, "score": 0, "message": ""}
        thr = apply_thresholds_for_key(key_token, numeric=v)
        if thr:
            return {"severity": thr[0], "score": 0, "message": thr[1]}
        if v > 25:
            severity = "red"; message = f"High IOP ({v})"
        elif v >= 21:
            severity = "yellow"; message = f"Borderline IOP ({v})"
        else:
            severity = "green"; message = f"IOP {v}"

    elif "spo2" in field or "oxygen" in field or "spo" in field:
        v = parse_number(value_raw)
        if v is None:
            return {"severity": None, "score": 0, "message": ""}
        thr = apply_thresholds_for_key(key_token, numeric=v)
        if thr:
            return {"severity": thr[0], "score": 0, "message": thr[1]}
        if v < 90:
            severity = "red"; message = f"Low SpO2 ({v}%)"
        elif v < 95:
            severity = "yellow"; message = f"Mildly low SpO2 ({v}%)"
        else:
            severity = "green"; message = f"SpO2 {v}%"

    else:
        # For unknown fields, only evaluate if the value contains a number
        v = parse_number(value_raw)
        if v is None:
            # No number found, return no evaluation (no color) for free-text fields
            return {"severity": None, "score": 0, "message": ""}
        # If a number is found, apply thresholds or generic rules
        thr = apply_thresholds_for_key(key_token, numeric=v)
        if thr:
            return {"severity": thr[0], "score": 0, "message": thr[1]}
        # Generic numeric rules (used when no specific thresholds exist).
        # Default behavior: treat values >=200 as red, >=100 as yellow, else green.
        # These are conservative defaults and can be overridden by EVAL_THRESHOLDS env var.
        if v >= 200:
            severity = "red"; message = f"High value ({v})"
        elif v >= 100:
            severity = "yellow"; message = f"Elevated value ({v})"
        else:
            severity = "green"; message = f"Value {v}"

    return {"severity": severity, "score": 0, "message": message}


@app.get("/health")
def health_check():
    return {"status": "ok"}


def _parse_encounter_date(enc: dict):
    d = enc.get("date") or enc.get("encounterDate") or enc.get("timestamp")
    if d is None:
        return None
    try:
        if isinstance(d, str):
            # Try ISO parse
            from dateutil import parser
            return parser.parse(d)
        return d
    except Exception:
        return None


@app.get("/api/analytics/patient/{reg_id}/iop-trend")
def patient_iop_trend(reg_id: str, limit: int = 12):
    """Return patient's IOP trend from their encounters (most recent first).
    Response: [{date: 'YYYY-MM-DD', od: <num>|null, os: <num>|null}, ...]
    """
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        return []
    encs = p.get("encounters") or []
    rows = []
    for e in encs:
        dt = _parse_encounter_date(e)
        od = None
        os = None
        try:
            od = e.get("vitals", {}).get("iop", {}).get("od") if isinstance(e.get("vitals"), dict) else None
            os = e.get("vitals", {}).get("iop", {}).get("os") if isinstance(e.get("vitals"), dict) else None
        except Exception:
            od = None; os = None
        rows.append({"date": dt, "od": od, "os": os})
    # filter out entries with no date
    rows = [r for r in rows if r["date"] is not None]
    rows.sort(key=lambda x: x["date"], reverse=True)
    out = []
    for r in rows[:limit][::-1]:
        out.append({"date": r["date"].date().isoformat(), "od": r["od"], "os": r["os"]})
    return out


@app.get("/api/analytics/patient/{reg_id}/visual-acuity")
def patient_visual_acuity(reg_id: str, limit: int = 12):
    """Return patient's visual acuity entries.
    Response: [{date: 'YYYY-MM-DD', od: <num>|null, os: <num>|null, odText: '', osText: ''}, ...]
    """
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        return []
    encs = p.get("encounters") or []
    rows = []
    for e in encs:
        dt = _parse_encounter_date(e)
        va = e.get("visualAcuity") or {}
        od = None; os = None; odText = None; osText = None
        try:
            od = va.get("od") if isinstance(va, dict) else None
            os = va.get("os") if isinstance(va, dict) else None
            odText = va.get("odText") or (str(od) if od is not None else None)
            osText = va.get("osText") or (str(os) if os is not None else None)
        except Exception:
            pass
        rows.append({"date": dt, "od": od, "os": os, "odText": odText, "osText": osText})
    rows = [r for r in rows if r["date"] is not None]
    rows.sort(key=lambda x: x["date"], reverse=True)
    out = []
    for r in rows[:limit][::-1]:
        out.append({"date": r["date"].date().isoformat(), "od": r["od"], "os": r["os"], "odText": r.get("odText") or "--", "osText": r.get("osText") or "--"})
    return out


@app.get("/api/analytics/patient/{reg_id}/visits")
def patient_visits(reg_id: str):
    """Return timeline of the patient's visits (dates).
    Count unique dates only (one visit per day, regardless of how many encounters).
    Response: [{date: 'YYYY-MM-DD', visits: 1}, ...]
    """
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        return []
    encs = p.get("encounters") or []
    dates_set = set()
    for e in encs:
        dt = _parse_encounter_date(e)
        if dt:
            dates_set.add(dt.date().isoformat())
    # collapse by month for easy charting, count unique visit dates per month
    from collections import Counter
    months = Counter([d[:7] for d in dates_set])
    out = []
    for m, cnt in sorted(months.items()):
        out.append({"month": m, "visits": cnt})
    return out


@app.get("/api/analytics/patient/{reg_id}/iop-distribution")
def patient_iop_distribution(reg_id: str):
    """Return raw IOP readings for histogramming: {iops: [numbers]}
    """
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        return {"iops": []}
    encs = p.get("encounters") or []
    vals = []
    for e in encs:
        try:
            od = e.get("vitals", {}).get("iop", {}).get("od")
            os = e.get("vitals", {}).get("iop", {}).get("os")
            if od is not None:
                vals.append(float(od))
            if os is not None:
                vals.append(float(os))
        except Exception:
            continue
    return {"iops": vals}


@app.get("/api/analytics/patient/{reg_id}/procedures")
def patient_procedures_timeline(reg_id: str):
    """Return procedures/interventions timeline for patient.
    Response: [{date: 'YYYY-MM-DD', procedures: [<names>]}, ...]
    """
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        return []
    encs = p.get("encounters") or []
    out = []
    for e in encs:
        dt = _parse_encounter_date(e)
        if not dt:
            continue
        procs = e.get("procedures") or e.get("operations") or []
        # normalize to list of strings
        try:
            procs_list = [str(x) for x in procs] if isinstance(procs, (list, tuple)) else [str(procs)]
        except Exception:
            procs_list = []
        out.append({"date": dt.date().isoformat(), "procedures": procs_list})
    out.sort(key=lambda x: x["date"])
    return out

# --- CORS Middleware ---
# For local development allow all origins to avoid CORS blocking from various dev servers.
# Narrow this before deploying to production.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def generate_registration_id():
    """Generates a random registration ID."""
    prefix = "REG"
    year = "2025"
    random_part = ''.join(random.choices(string.digits, k=6))
    return f"{prefix}-{year}-{random_part}"

def sanitize(obj):
    """Convert MongoDB ObjectId and datetime objects to JSON-serializable types."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == '_id' and isinstance(v, ObjectId):
                out['id'] = str(v)
            elif isinstance(v, ObjectId):
                out[k] = str(v)
            elif isinstance(v, datetime):
                out[k] = v.isoformat()
            else:
                out[k] = sanitize(v)
        return out
    elif isinstance(obj, list):
        return [sanitize(x) for x in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

@app.get("/")
def read_root():
    return {"message": "Welcome to the EMR Dashboard API"}

# --- API Endpoints ---

@app.post("/patients/new", response_model=PatientInDB)
async def create_new_patient(patient_data: NewPatient = Body(...)):
    """
    Creates a new patient record. Used by receptionists.
    """
    details = patient_data.patientDetails

    # Validate required fields
    if not all([details.name, details.password, details.phone]):
        raise HTTPException(status_code=400, detail="Name, password, and phone are required fields.")

    # Separate emergency contact info
    emergency_parts = details.emergencyContact.split('-', 1)
    emergency_name = emergency_parts[0].strip() if emergency_parts else ""
    emergency_phone = emergency_parts[1].strip() if len(emergency_parts) > 1 else ""

    # --- Duplicate checks ---
    # If an email is provided, ensure it's not already registered (case-insensitive)
    email = (details.email or "").strip()
    if email:
        # Use a case-insensitive regex match on contactInfo.email
        escaped = re.escape(email)
        existing_by_email = patient_collection.find_one({"contactInfo.email": {"$regex": f"^{escaped}$", "$options": "i"}})
        if existing_by_email:
            raise HTTPException(status_code=409, detail="A patient with this email already exists.")

    # Construct the document to be saved in the database
    patient_dict = {
        "name": details.name,
        "demographics": Demographics(
            age=int(details.age) if details.age.isdigit() else 0,
            sex=details.sex,
            bloodType=details.bloodType
        ).model_dump(),
        "contactInfo": ContactInfo(
            phone=details.phone,
            email=details.email,
            address=details.address
        ).model_dump(),
        "emergencyContact": EmergencyContact(
            name=emergency_name,
            phone=emergency_phone
        ).model_dump(),
        "allergies": [s.strip() for s in details.allergies.split(',') if s.strip()],
        "history": History(
            **patient_data.presentingComplaints.history.model_dump(),
            medical=patient_data.medicalHistory.medical,
            surgical=patient_data.medicalHistory.surgical,
            family=patient_data.medicalHistory.familyHistory
        ).model_dump(),
        # Persist any drug/medication history supplied by the receptionist form.
        "drugHistory": patient_data.drugHistory if hasattr(patient_data, 'drugHistory') else {},
    }

    # Hash the password
    hashed_password = pwd_context.hash(details.password)
    patient_dict["hashed_password"] = hashed_password

    # Use provided registrationId or generate a new one
    if patient_data.registrationId:
        patient_dict["registrationId"] = patient_data.registrationId
    else:
        patient_dict["registrationId"] = generate_registration_id()
    # Creation timestamp
    patient_dict["created_at"] = datetime.utcnow().isoformat()

    # Create the initial encounter from the receptionist's data
    initial_encounter = ReceptionistEncounter(
        doctor="Reception",
        presentingComplaints=patient_data.presentingComplaints.complaints
    )
    patient_dict["encounters"] = [initial_encounter.model_dump()]

    # Insert into the database
    new_patient = patient_collection.insert_one(patient_dict)

    # Retrieve the newly created document to return it
    created_patient = patient_collection.find_one({"_id": new_patient.inserted_id})

    if created_patient:
        # Normalize history year fields so Pydantic validators that expect the
        # alias `year` will find the value. Some parts of the code/db store
        # the field as `diagnosedYear` or `procedureYear` which causes
        # validation errors when building `PatientInDB`.
        hist = created_patient.get("history") or {}
        # medical items: diagnosedYear -> year
        for m in hist.get("medical", []):
            if isinstance(m, dict) and "diagnosedYear" in m and "year" not in m:
                m["year"] = m.pop("diagnosedYear")
        # surgical items: procedureYear -> year
        for s in hist.get("surgical", []):
            if isinstance(s, dict) and "procedureYear" in s and "year" not in s:
                s["year"] = s.pop("procedureYear")

        created_patient["history"] = hist

        # Build a sanitized payload matching PatientInDB to avoid validation
        # errors caused by unexpected/missing keys in the raw DB document.
        sanitized = {
            "_id": created_patient.get("_id"),
            "registrationId": created_patient.get("registrationId"),
            "name": created_patient.get("name"),
            "hashed_password": created_patient.get("hashed_password") or created_patient.get("hashedPassword") or "",
            "demographics": created_patient.get("demographics") or {},
            "contactInfo": created_patient.get("contactInfo") or {},
            "emergencyContact": created_patient.get("emergencyContact") or {},
            "allergies": created_patient.get("allergies") or [],
            "documents_id": created_patient.get("documents_id") or [],
            "documents": created_patient.get("documents") or [],
            "drugHistory": created_patient.get("drugHistory"),
            "doctor": created_patient.get("doctor"),
            "history": created_patient.get("history") or {},
            "encounters": created_patient.get("encounters") or [],
        }

        return PatientInDB(**sanitized)
    
    raise HTTPException(status_code=500, detail="Failed to create patient.")


@app.post("/users/new", response_model=UserInDB)
async def create_new_user(user_data: NewUser = Body(...)):
    """
    Creates a new user account. Roles must be one of the ALLOWED_ROLES.
    """
    username = (user_data.username or "").strip()
    role = (user_data.role or "").strip().upper()

    if not username or not user_data.password:
        raise HTTPException(status_code=400, detail="username and password are required")

    if role not in ALLOWED_ROLES:
        raise HTTPException(status_code=400, detail=f"role must be one of: {', '.join(sorted(ALLOWED_ROLES))}")

    # Check uniqueness
    if user_collection.find_one({"username": username}):
        raise HTTPException(status_code=409, detail="username already exists")

    hashed = pwd_context.hash(user_data.password)

    user_doc = {
        "username": username,
        "full_name": user_data.full_name or "",
        "hashed_password": hashed,
        "role": role,
        "created_at": __import__('datetime').datetime.utcnow(),
    }

    res = user_collection.insert_one(user_doc)
    created = user_collection.find_one({"_id": res.inserted_id})
    if created:
        return UserInDB(**created)

    raise HTTPException(status_code=500, detail="Failed to create user")


@app.get("/users/all")
async def get_all_users(role: str | None = None):
    """Get all users from database. Optionally filter by role."""
    try:
        query = {}
        if role:
            query = {"role": role.upper()}
        
        users = list(user_collection.find(query, {"hashed_password": 0}))
        
        # Convert ObjectId to string for JSON serialization
        for user in users:
            if "_id" in user:
                user["_id"] = str(user["_id"])
        
        return {
            "users": users,
            "count": len(users)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch users: {str(e)}")


@app.post("/auth/login")
async def login(credentials: dict = Body(...)):
    """Simple login endpoint: accepts { username, password, role? } and returns basic user info on success.
    If a role is provided, the endpoint ensures the stored user role matches the requested role.
    This is minimal and does not issue JWTs yet — it returns 200 with username and role.
    """
    username = (credentials.get("username") or "").strip()
    password = credentials.get("password") or ""
    requested_role = (credentials.get("role") or "").strip().upper()

    if not username or not password:
        raise HTTPException(status_code=400, detail="username and password required")

    user = user_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=401, detail="invalid credentials")

    # Verify password
    if not pwd_context.verify(password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="invalid credentials")

    stored_role = (user.get("role") or "").strip().upper()
    if requested_role:
        # If user supplied a role during sign-in, ensure it matches the stored role
        if requested_role != stored_role:
            raise HTTPException(status_code=401, detail="role does not match credentials")

    # Return basic user info (no token yet)
    return {
        "username": user.get("username"),
        "full_name": user.get("full_name"),
        "role": stored_role
    }



@app.post("/patients/{registration_id}/documents")
async def upload_patient_documents(registration_id: str, files: list[UploadFile] = File(...), uploaded_by: str = Form(None)):
    """Upload one or more files and attach them to the patient document identified by registration_id.
    Files are saved under backend/uploads/{registration_id}/ and metadata is pushed into patient.documents.
    """
    # Find the patient
    patient = patient_collection.find_one({"registrationId": registration_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    upload_root = os.path.join(os.getcwd(), 'backend', 'uploads', registration_id)
    os.makedirs(upload_root, exist_ok=True)

    saved_docs = []
    for f in files:
        # safe filename generation
        file_id = str(uuid.uuid4())
        filename = f.filename
        _, ext = os.path.splitext(filename)
        safe_name = f"{file_id}{ext}"
        path = os.path.join(upload_root, safe_name)

        # write file to disk
        with open(path, 'wb') as out_file:
            content = await f.read()
            out_file.write(content)

        # build metadata
        metadata = {
            "id": file_id,
            "name": filename,
            "stored_name": safe_name,
            "path": path,
            "size": len(content),
            "type": ext.lower().lstrip('.'),
            "uploadedDate": datetime.utcnow().isoformat(),
            "uploadedBy": uploaded_by or "Reception",
        }

        # push into DB
        patient_collection.update_one({"registrationId": registration_id}, {"$push": {"documents": metadata}})
        saved_docs.append(metadata)

    return {"saved": saved_docs}


@app.get("/patients/{registration_id}/documents")
async def get_patient_documents(registration_id: str):
    """Return stored document metadata for a given patient registration id."""
    patient = patient_collection.find_one({"registrationId": registration_id}, {"documents": 1})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    docs = patient.get("documents") or []
    # sanitize documents for JSON
    def sanitize(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, ObjectId):
                    out[k] = str(v)
                else:
                    out[k] = sanitize(v)
            return out
        elif isinstance(obj, list):
            return [sanitize(x) for x in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    return {"documents": sanitize(docs)}


@app.get("/patients/{registration_id}/documents/{file_id}/download")
async def download_patient_document(registration_id: str, file_id: str, inline: bool | None = None):
    """Download or preview a previously uploaded patient document.
    - `inline=1` will attempt to return the file with a media-type so browsers can preview images/videos
    - otherwise the file is returned as an attachment for download
    """
    patient = patient_collection.find_one({"registrationId": registration_id}, {"documents": 1})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    docs = patient.get("documents") or []
    # find by id or stored_name starting with id
    match = None
    for d in docs:
        if str(d.get("id")) == str(file_id) or (d.get("stored_name") and d.get("stored_name").startswith(str(file_id))):
            match = d
            break

    if not match:
        raise HTTPException(status_code=404, detail="Document not found")

    stored_name = match.get("stored_name")
    upload_root = os.path.join(os.getcwd(), 'backend', 'uploads', registration_id)
    path = match.get("path") or os.path.join(upload_root, stored_name or "")

    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    # Set media type for common types to allow browser previews
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    media_type = "application/octet-stream"
    if ext in ('png', 'jpg', 'jpeg', 'gif'):
        media_type = f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"
    elif ext in ('mp4', 'webm', 'ogg'):
        media_type = f"video/{ext}"
    elif ext == 'pdf':
        media_type = 'application/pdf'

    # If inline flag is set (truthy), return with media_type so browser can preview
    if inline:
        return FileResponse(path, media_type=media_type, filename=match.get('name'))

    # Default: force download
    return FileResponse(path, media_type='application/octet-stream', filename=match.get('name'))


@app.get("/patients/recent")
async def get_most_recent_patient():
    """Return the most recently created patient record."""
    doc = patient_collection.find_one(sort=[('created_at', -1)])
    if not doc:
        raise HTTPException(status_code=404, detail="No patients found")

    def sanitize(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == '_id' and isinstance(v, ObjectId):
                    out['id'] = str(v)
                elif isinstance(v, ObjectId):
                    out[k] = str(v)
                elif isinstance(v, datetime):
                    out[k] = v.isoformat()
                else:
                    out[k] = sanitize(v)
            return out
        elif isinstance(obj, list):
            return [sanitize(x) for x in obj]
        else:
            return obj

    return sanitize(doc)

# Search patients by name, email, phone, or registration ID (case-insensitive). Returns a list of lightweight
# documents suitable for showing search results in the UI.
@app.get("/patients/search")
async def search_patients(q: str | None = None, limit: int = 20):
    """Search patients by name, email, phone, or registration ID. If q is None or empty returns an empty list.
    Uses case-insensitive regex to match name/email/phone/regid queries.
    """
    if not q or not q.strip():
        return {"results": []}

    # Build a case-insensitive regex that matches anywhere in the name, phone, or registration ID
    term = q.strip()
    # Escape user input for regex safety
    try:
        import re as _re
        escaped = _re.escape(term)
    except Exception:
        escaped = term

    # Search name (contains) OR patientDetails.email (exact/prefix) OR patientDetails.phone (contains) OR registrationId (contains)
    regex_name = {"$regex": escaped, "$options": "i"}
    regex_email = {"$regex": f"^{escaped}$", "$options": "i"}
    regex_phone = {"$regex": escaped, "$options": "i"}
    regex_regid = {"$regex": escaped, "$options": "i"}

    cursor = patient_collection.find(
        {"$or": [
            {"name": regex_name}, 
            {"patientDetails.email": regex_email},
            {"patientDetails.phone": regex_phone},
            {"registrationId": regex_regid}
        ]},
        {"name": 1, "registrationId": 1, "patientDetails.email": 1, "patientDetails.phone": 1, "created_at": 1}
    ).limit(limit)

    results = []
    for doc in cursor:
        # extract phone and email from patientDetails
        patient_details = doc.get("patientDetails") or {}
        phone = patient_details.get("phone")
        email = patient_details.get("email")
        profile_pic = doc.get("profilePic") or doc.get("profile_pic")

        # derive a last visit date from encounters if available
        last_visit = None
        encs = doc.get("encounters") or []
        if isinstance(encs, list) and len(encs) > 0:
            try:
                # Prefer the last encounter's date if present
                lv = encs[-1].get("date")
                if hasattr(lv, 'isoformat'):
                    last_visit = lv.isoformat()
                else:
                    last_visit = str(lv)
            except Exception:
                last_visit = None
        # If no last visit found, fall back to created_at if available
        if not last_visit:
            last_visit = doc.get("created_at")
        results.append({
            "name": doc.get("name"),
            "registrationId": doc.get("registrationId"),
            "email": email,
            "phone": phone,
            "profilePic": profile_pic,
            "created_at": doc.get("created_at"),
            "lastVisit": last_visit,
        })

    return {"results": results}


@app.get("/patients/all")
async def get_all_patients():
    """Retrieve all patients with their basic information."""
    try:
        cursor = patient_collection.find(
            {},
            {"name": 1, "registrationId": 1, "demographics": 1, "contactInfo": 1, "created_at": 1}
        ).sort("created_at", -1)
        
        patients = []
        for doc in cursor:
            # Convert ObjectId to string for JSON serialization
            doc_copy = {
                "_id": str(doc.get("_id")),
                "name": doc.get("name"),
                "registrationId": doc.get("registrationId"),
                "created_at": doc.get("created_at"),
                "demographics": doc.get("demographics", {}),
                "contactInfo": doc.get("contactInfo", {}),
            }
            patients.append(doc_copy)
        
        return {"patients": patients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving patients: {str(e)}")


@app.put("/patients/{registration_id}/investigations")
async def upsert_patient_investigations(registration_id: str, investigations: InvestigationsModel = Body(...), updated_by: str | None = None):
    """Insert or update investigation sections (optometry, ophthalmicInvestigations,
    iop, systemic) for the patient identified by registration_id.
    The frontend can send one or more of these sections; the endpoint will set
    the provided fields under patient.investigations and push an encounter entry.
    """
    patient = patient_collection.find_one({"registrationId": registration_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    set_updates = {}
    if investigations.optometry is not None:
        set_updates["investigations.optometry"] = investigations.optometry
    if investigations.ophthalmicInvestigations is not None:
        set_updates["investigations.ophthalmicInvestigations"] = investigations.ophthalmicInvestigations
    if investigations.iop is not None:
        set_updates["investigations.iop"] = investigations.iop
    if investigations.systemic is not None:
        set_updates["investigations.systemic"] = investigations.systemic

    if not set_updates:
        raise HTTPException(status_code=400, detail="No investigation data provided")

    # Prepare an encounter log for audit/history
    encounter_entry = {
        "date": datetime.utcnow(),
        "doctor": updated_by or "OPD",
        "type": "investigation_update",
        "details": {
            "fieldsUpdated": list(set_updates.keys())
        }
    }

    update_ops = {"$set": set_updates, "$push": {"encounters": encounter_entry}}
    patient_collection.update_one({"registrationId": registration_id}, update_ops)

    # Return the updated fields
    updated = patient_collection.find_one({"registrationId": registration_id}, {"investigations": 1, "registrationId":1})

    # sanitize before returning
    def sanitize(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, ObjectId):
                    out[k] = str(v)
                elif isinstance(v, datetime):
                    out[k] = v.isoformat()
                else:
                    out[k] = sanitize(v)
            return out
        elif isinstance(obj, list):
            return [sanitize(x) for x in obj]
        else:
            return obj

    return sanitize(updated)


@app.put("/patients/{registration_id}/doctor")
async def upsert_patient_doctor(registration_id: str, doctor_payload: DoctorEncounterModel = Body(...), updated_by: str | None = None):
    """Store doctor-entered examination/prescription data under patient.doctor and push an encounter.
    The payload is intentionally permissive (dicts) to match the investigations approach used for OPD.
    """
    patient = patient_collection.find_one({"registrationId": registration_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Build a list of provided top-level doctor keys so we can both persist the
    # full doctor payload and record which subfields were updated for the encounter.
    provided_keys = []
    # Expanded key list to include new fields from frontend
    for key in ("examination", "ophthalmologistExam", "prescription", "specialExamination", "investigationsSurgeries", "diagnosis", "followUp", "doctorName"):
        if getattr(doctor_payload, key, None) is not None:
            provided_keys.append(key)

    if not provided_keys:
        raise HTTPException(status_code=400, detail="No doctor data provided")

    # Persist the entire doctor payload under patient.doctor so the frontend's
    # permissive dictionary structure is kept intact. Use model_dump() to get
    # a plain dict that can be stored directly in MongoDB.
    set_updates = {"doctor": doctor_payload.model_dump()}

    encounter_entry = {
        "date": datetime.utcnow(),
        "doctor": updated_by or "Doctor",
        "type": "doctor_update",
        "details": {"fieldsUpdated": [f"doctor.{k}" for k in provided_keys]}
    }

    update_ops = {"$set": set_updates, "$push": {"encounters": encounter_entry}}
    patient_collection.update_one({"registrationId": registration_id}, update_ops)

    updated = patient_collection.find_one({"registrationId": registration_id}, {"doctor": 1, "registrationId":1})

    def sanitize(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if isinstance(v, ObjectId):
                    out[k] = str(v)
                elif isinstance(v, datetime):
                    out[k] = v.isoformat()
                else:
                    out[k] = sanitize(v)
            return out
        elif isinstance(obj, list):
            return [sanitize(x) for x in obj]
        else:
            return obj

    return sanitize(updated)


@app.get("/patients/{registration_id}")
async def get_patient_by_registration(registration_id: str):
    """Return the full patient document for a given registration id (sanitized for JSON)."""
    doc = patient_collection.find_one({"registrationId": registration_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Patient not found")

    def sanitize(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == '_id' and isinstance(v, ObjectId):
                    out['id'] = str(v)
                elif isinstance(v, ObjectId):
                    out[k] = str(v)
                elif isinstance(v, datetime):
                    out[k] = v.isoformat()
                else:
                    out[k] = sanitize(v)
            return out
        elif isinstance(obj, list):
            return [sanitize(x) for x in obj]
        else:
            return obj

    return sanitize(doc)


# --- Visit History Endpoints ---

@app.post("/patients/{registration_id}/visit")
async def save_visit_history(registration_id: str, visit_data: dict = Body(...)):
    """
    Save a complete visit record for a patient to MongoDB.
    A visit includes all stages (reception, OPD, doctor) completed.
    If patient doesn't exist, create a minimal record first.
    """
    try:
        print(f"DEBUG: Saving visit for patient {registration_id}")
        print(f"DEBUG: Visit data: {visit_data}")
        
        # Find the patient
        patient = patient_collection.find_one({"registrationId": registration_id})
        
        # If patient doesn't exist, create a minimal record
        if not patient:
            print(f"ℹ Patient {registration_id} not found in MongoDB, creating minimal record...")
            minimal_patient = {
                "registrationId": registration_id,
                "name": visit_data.get("stages", {}).get("reception", {}).get("data", {}).get("patientDetails", {}).get("name", f"Patient {registration_id}"),
                "contactInfo": {
                    "phone": visit_data.get("stages", {}).get("reception", {}).get("data", {}).get("patientDetails", {}).get("phone", ""),
                    "email": visit_data.get("stages", {}).get("reception", {}).get("data", {}).get("patientDetails", {}).get("email", "")
                },
                "visits": [],
                "createdAt": datetime.now().isoformat()
            }
            result = patient_collection.insert_one(minimal_patient)
            print(f"✓ Created minimal patient record for {registration_id} (ID: {result.inserted_id})")
        
        # Create visit record
        visit_record = {
            "visitId": visit_data.get("visitId", f"{registration_id}-{datetime.now().isoformat()}"),
            "visitDate": visit_data.get("visitDate", datetime.now().isoformat()),
            "stages": visit_data.get("stages", {}),
            "createdAt": datetime.now().isoformat()
        }
        
        print(f"DEBUG: Pushing visit record: {visit_record['visitId']}")
        
        # Add visit to patient's visits array in MongoDB (use $setOnInsert to ensure visits array exists)
        result = patient_collection.update_one(
            {"registrationId": registration_id},
            {
                "$push": {"visits": visit_record},
                "$setOnInsert": {"visits": []}  # Ensure visits array exists if creating
            },
            upsert=True  # Create if doesn't exist
        )
        
        print(f"DEBUG: Update result - matched: {result.matched_count}, modified: {result.modified_count}, upserted: {result.upserted_id}")
        
        if result.modified_count > 0 or result.upserted_id:
            print(f"✓ Visit saved to MongoDB for {registration_id}")
            return {
                "status": "success",
                "message": f"Visit saved for patient {registration_id}",
                "visitId": visit_record["visitId"]
            }
        else:
            print(f"✗ No update occurred for {registration_id}")
            raise HTTPException(status_code=400, detail="Failed to save visit")
    
    except Exception as e:
        import traceback
        print(f"✗ Error saving visit: {str(e)}")
        print(f"✗ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to save visit: {str(e)}")


@app.get("/patients/{registration_id}/visits")
async def get_patient_visits(registration_id: str):
    """
    Get all visit records for a patient from MongoDB.
    Returns complete examination history with all stages.
    """
    try:
        patient = patient_collection.find_one(
            {"registrationId": registration_id},
            {"visits": 1, "name": 1, "registrationId": 1, "contactInfo": 1}
        )
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        visits = patient.get("visits", [])
        
        return {
            "patientName": patient.get("name"),
            "registrationId": registration_id,
            "totalVisits": len(visits),
            "visits": visits,
            "contactInfo": patient.get("contactInfo", {})
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Appointment Management Endpoints ---

@app.post("/appointments")
async def create_appointment(appointment_data: dict = Body(...)):
    """
    Create or book a new appointment.
    Save to MongoDB appointments collection.
    Includes validation for past dates and doctor availability.
    """
    try:
        doctor_id = appointment_data.get("doctorId")
        doctor_name = appointment_data.get("doctorName")
        appt_date_str = appointment_data.get("appointmentDate")
        appt_time_str = appointment_data.get("appointmentTime")

        if not all([doctor_id, appt_date_str, appt_time_str]):
            raise HTTPException(status_code=400, detail="Missing required booking information")

        # 1. Check for past date/time
        try:
            # Parse date and time to compare with current time
            appt_datetime = datetime.strptime(f"{appt_date_str} {appt_time_str}", "%Y-%m-%d %H:%M")
            if appt_datetime < datetime.now():
                raise HTTPException(status_code=400, detail="Cannot book appointments in the past. Please select a future time.")
        except ValueError as ve:
            print(f"Date parsing error: {str(ve)}")
            raise HTTPException(status_code=400, detail="Invalid date or time format. Use YYYY-MM-DD and HH:MM")

        # Ensure appointments collection exists
        if "appointments" not in db.list_collection_names():
            db.create_collection("appointments")
        
        appointments_collection = db["appointments"]

        # 2. Check for slot exclusivity for the doctor
        # We check for the same doctor, same date, and same time slot
        existing_appointment = appointments_collection.find_one({
            "doctorId": doctor_id,
            "appointmentDate": appt_date_str,
            "appointmentTime": appt_time_str,
            "status": {"$ne": "cancelled"}
        })
        
        if existing_appointment:
            raise HTTPException(status_code=400, detail=f"The {appt_time_str} slot is already booked for {doctor_name or 'the selected doctor'}")

        # Extract appointment details
        appointment = {
            "appointmentId": appointment_data.get("_id", f"APT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"),
            "patientName": appointment_data.get("patientName"),
            "patientRegistrationId": appointment_data.get("patientRegistrationId"),
            "doctorName": doctor_name,
            "doctorId": doctor_id,
            "appointmentDate": appt_date_str,
            "appointmentTime": appt_time_str,
            "status": appointment_data.get("status", "booked"),
            "phone": appointment_data.get("phone"),
            "email": appointment_data.get("email"),
            "bookedAt": appointment_data.get("bookedAt", datetime.now().isoformat()),
            "notes": appointment_data.get("notes", ""),
            "createdAt": datetime.now().isoformat()
        }
        
        result = appointments_collection.insert_one(appointment)
        
        print(f"✓ Appointment created: {appointment['patientName']} with {doctor_name} at {appt_time_str} on {appt_date_str}")
        
        return {
            "status": "success",
            "message": f"Appointment successfully booked for {appointment['patientName']} with {doctor_name}",
            "appointmentId": str(result.inserted_id)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"✗ Error creating appointment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create appointment: {str(e)}")


@app.get("/appointments")
async def get_all_appointments():
    """
    Get all appointments from MongoDB.
    """
    try:
        # Ensure collection exists
        if "appointments" not in db.list_collection_names():
            return {
                "status": "success",
                "totalAppointments": 0,
                "appointments": []
            }
        
        appointments_collection = db["appointments"]
        appointments = list(appointments_collection.find())
        
        print(f"✓ Fetched {len(appointments)} appointments")
        
        return {
            "status": "success",
            "totalAppointments": len(appointments),
            "appointments": [sanitize(apt) for apt in appointments]
        }
    except Exception as e:
        print(f"✗ Error fetching appointments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch appointments: {str(e)}")


@app.get("/appointments/{appointment_id}")
async def get_appointment(appointment_id: str):
    """
    Get a specific appointment by ID.
    """
    try:
        appointments_collection = db["appointments"]
        
        # Try ObjectId first, then string match
        try:
            apt = appointments_collection.find_one({"_id": ObjectId(appointment_id)})
        except:
            apt = appointments_collection.find_one({"appointmentId": appointment_id})
        
        if not apt:
            raise HTTPException(status_code=404, detail="Appointment not found")
        
        return sanitize(apt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/appointments/{appointment_id}")
async def update_appointment(appointment_id: str, update_data: dict = Body(...)):
    """
    Update appointment status or details.
    Used to move appointments through queue states.
    Will create the appointment if it doesn't exist (upsert).
    """
    try:
        appointments_collection = db["appointments"]
        
        # Build update with timestamp
        update_doc = {
            **update_data,
            "updatedAt": datetime.now().isoformat()
        }
        
        # Try ObjectId first, then string match with upsert
        try:
            result = appointments_collection.update_one(
                {"_id": ObjectId(appointment_id)},
                {"$set": update_doc},
                upsert=True
            )
        except:
            result = appointments_collection.update_one(
                {"appointmentId": appointment_id},
                {"$set": update_doc},
                upsert=True
            )
        
        return {
            "status": "success",
            "message": f"Appointment updated",
            "appointmentId": appointment_id,
            "upserted": result.upserted_id is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/appointments")
async def get_queue_appointments(status: str = None):
    """
    Get appointments filtered by status.
    Used to load specific queue views (booked, reception_pending, etc.)
    """
    try:
        # Ensure collection exists
        if "appointments" not in db.list_collection_names():
            return {
                "status": "success",
                "queue": status or "all",
                "totalAppointments": 0,
                "appointments": []
            }
        
        appointments_collection = db["appointments"]
        
        query = {}
        if status:
            query["status"] = status
        
        appointments = list(appointments_collection.find(query))
        
        print(f"✓ Fetched {len(appointments)} appointments with status: {status or 'any'}")
        
        return {
            "status": "success",
            "queue": status or "all",
            "totalAppointments": len(appointments),
            "appointments": [sanitize(apt) for apt in appointments]
        }
    except Exception as e:
        print(f"✗ Error fetching queue appointments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch queue: {str(e)}")


@app.post("/queue/reception")
async def create_reception_queue(item: dict = Body(...)):
    """Create a reception queue item and mark the appointment as reception_pending.
    Prevents duplicate queue entries for the same appointment that is still waiting.
    """
    try:
        rc = db["reception_queue"]
        appt_id = item.get("appointmentId")
        
        # Check if this appointment already has a waiting reception queue entry
        existing = rc.find_one({
            "appointmentId": appt_id,
            "status": {"$in": ["waiting", "in_progress"]}
        })
        if existing:
            return {
                "status": "duplicate",
                "message": "Appointment already in reception queue",
                "queueId": str(existing.get("_id")),
                "item": sanitize(existing)
            }

        doc = {
            "appointmentId": appt_id,
            "registrationId": item.get("registrationId"),
            "patientName": item.get("patientName"),
            "appointmentDate": item.get("appointmentDate"),
            "status": item.get("status", "waiting"),
            "action": item.get("action", "reception_waiting"),
            "queueStatus": "Reception",  # Task 3: Canonical queue badge status
            "receptionData": item.get("receptionData", {}),
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat()
        }

        res = rc.insert_one(doc)

        # Sync appointment status and queueStatus
        appointments_collection = db["appointments"]
        try:
            appointments_collection.update_one({"_id": ObjectId(doc["appointmentId"])}, {"$set": {"status": "reception_pending", "queueStatus": "Reception", "updatedAt": datetime.utcnow().isoformat()}}, upsert=False)
        except Exception:
            # fallback to appointmentId string match
            appointments_collection.update_one({"appointmentId": doc["appointmentId"]}, {"$set": {"status": "reception_pending", "queueStatus": "Reception", "updatedAt": datetime.utcnow().isoformat()}}, upsert=False)

        created = rc.find_one({"_id": res.inserted_id})
        return {"status": "success", "queueId": str(res.inserted_id), "item": sanitize(created)}
    except Exception as e:
        print(f"✗ Error creating reception queue item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/reception")
async def list_reception_queue(status: str | None = None):
    """List reception queue items, optional filter by status."""
    try:
        rc = db["reception_queue"]
        query = {}
        if status:
            query["status"] = status
        items = list(rc.find(query))
        return {"status": "success", "total": len(items), "items": [sanitize(i) for i in items]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/queue/reception/{queue_id}")
async def update_reception_queue(queue_id: str, payload: dict = Body(...)):
    """Update a reception queue item. If marked done, create an OPD queue item and update patient/appointment."""
    try:
        rc = db["reception_queue"]
        # try ObjectId then string id
        try:
            qobj = rc.find_one({"_id": ObjectId(queue_id)})
        except Exception:
            qobj = rc.find_one({"_id": queue_id})

        if not qobj:
            raise HTTPException(status_code=404, detail="Reception queue item not found")

        updates = {**payload, "updatedAt": datetime.utcnow().isoformat()}
        rc.update_one({"_id": qobj.get("_id")}, {"$set": updates})

        # If reception completed, sync appointment, upsert patient partial data, and push to OPD queue
        if payload.get("status") in ("done", "opd_pending") or payload.get("action") == "reception_done":
            # update appointment status and queueStatus
            appointments_collection = db["appointments"]
            appt_id = qobj.get("appointmentId")
            update_op = {"$set": {"status": "opd_pending", "queueStatus": "OPD", "updatedAt": datetime.utcnow().isoformat()}}
            
            matched = False
            try:
                # Try ObjectId first
                result = appointments_collection.update_one({"_id": ObjectId(appt_id)}, update_op, upsert=False)
                if result.matched_count > 0:
                    matched = True
            except Exception:
                pass
                
            if not matched:
                # Try string ID or appointmentId field (ensure appt_id is string for these checks)
                str_appt_id = str(appt_id)
                appointments_collection.update_one(
                    {"$or": [{"_id": str_appt_id}, {"appointmentId": str_appt_id}]}, 
                    update_op, 
                    upsert=False
                )

            # upsert patient with reception data
            reg = qobj.get("registrationId")
            reception_data = payload.get("receptionData") or qobj.get("receptionData") or {}
            if reg:
                pat_col = db["patients"]
                encounter = {"date": datetime.utcnow(), "doctor": "Reception", "type": "reception", "details": reception_data}
                pat_col.update_one({"registrationId": reg}, {"$set": {"lastReception": reception_data, "lastUpdated": datetime.utcnow().isoformat()}, "$push": {"encounters": encounter}}, upsert=True)

            # create OPD queue item with queueStatus
            oq = db["opd_queue"]
            opd_doc = {
                "appointmentId": appt_id,
                "registrationId": reg,
                "patientName": qobj.get("patientName"),
                "appointmentDate": qobj.get("appointmentDate"),
                "status": "waiting",
                "queueStatus": "OPD",  # Task 3: Canonical queue badge status
                "opdData": {},
                "receptionData": reception_data,
                "createdAt": datetime.utcnow().isoformat(),
                "updatedAt": datetime.utcnow().isoformat()
            }
            res = oq.insert_one(opd_doc)
            created_opd = oq.find_one({"_id": res.inserted_id})

            return {"status": "success", "message": "Reception completed, OPD queued", "opdQueueId": str(res.inserted_id), "opdItem": sanitize(created_opd)}

        updated = rc.find_one({"_id": qobj.get("_id")})
        return {"status": "success", "item": sanitize(updated)}
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error updating reception queue item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/queue/opd")
async def create_opd_queue(item: dict = Body(...)):
    """Create an OPD queue item and optionally update appointment status.
    Prevents duplicate queue entries for the same appointment that is still waiting.
    """
    try:
        oq = db["opd_queue"]
        appt_id = item.get("appointmentId")
        
        # Check if this appointment already has a waiting OPD queue entry
        existing = oq.find_one({
            "appointmentId": appt_id,
            "status": {"$in": ["waiting", "in_progress"]}
        })
        if existing:
            return {
                "status": "duplicate",
                "message": "Appointment already in OPD queue",
                "queueId": str(existing.get("_id")),
                "item": sanitize(existing)
            }
        
        doc = {
            "appointmentId": appt_id,
            "registrationId": item.get("registrationId"),
            "patientName": item.get("patientName"),
            "status": item.get("status", "waiting"),
            "queueStatus": "OPD",
            "opdData": item.get("opdData", {}),
            "receptionData": item.get("receptionData", {}),
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat()
        }
        res = oq.insert_one(doc)

        # sync appointment
        appointments_collection = db["appointments"]
        update_op = {"$set": {"status": "opd_pending", "queueStatus": "OPD", "updatedAt": datetime.utcnow().isoformat()}}
        
        matched = False
        try:
            # Try ObjectId first
            result = appointments_collection.update_one({"_id": ObjectId(doc["appointmentId"])}, update_op, upsert=False)
            if result.matched_count > 0:
                matched = True
        except Exception:
            pass
            
        if not matched:
            # Try string ID or appointmentId field
                str_appt_id = str(doc["appointmentId"])
                appointments_collection.update_one(
                    {"$or": [{"_id": str_appt_id}, {"appointmentId": str_appt_id}]}, 
            )

        created = oq.find_one({"_id": res.inserted_id})
        return {"status": "success", "queueId": str(res.inserted_id), "item": sanitize(created)}
    except Exception as e:
        print(f"✗ Error creating OPD queue item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/opd")
async def list_opd_queue(status: str | None = None):
    try:
        oq = db["opd_queue"]
        query = {}
        if status:
            query["status"] = status
        items = list(oq.find(query))
        return {"status": "success", "total": len(items), "items": [sanitize(i) for i in items]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/queue/opd/{queue_id}")
async def update_opd_queue(queue_id: str, payload: dict = Body(...)):
    try:
        oq = db["opd_queue"]
        try:
            qobj = oq.find_one({"_id": ObjectId(queue_id)})
        except Exception:
            qobj = oq.find_one({"_id": queue_id})

        if not qobj:
            raise HTTPException(status_code=404, detail="OPD queue item not found")

        updates = {**payload, "updatedAt": datetime.utcnow().isoformat()}
        oq.update_one({"_id": qobj.get("_id")}, {"$set": updates})

        # If OPD completed, update appointment, patient, and create doctor queue item
        if payload.get("status") in ("done", "doctor_pending") or payload.get("action") == "opd_done":
            appointments_collection = db["appointments"]
            appt_id = qobj.get("appointmentId")
            update_op = {"$set": {"status": "doctor_pending", "queueStatus": "Doctor", "updatedAt": datetime.utcnow().isoformat()}}
            
            matched = False
            try:
                result = appointments_collection.update_one({"_id": ObjectId(appt_id)}, update_op, upsert=False)
                if result.matched_count > 0: matched = True
            except Exception:
                pass
            
            if not matched:
                str_appt_id = str(appt_id)
                appointments_collection.update_one(
                    {"$or": [{"_id": str_appt_id}, {"appointmentId": str_appt_id}]}, 
                    update_op, 
                    upsert=False
                )

            # update patient with opd data
            reg = qobj.get("registrationId")
            opd_data = payload.get("opdData") or qobj.get("opdData") or {}
            reception_data = qobj.get("receptionData") or {}
            
            if reg:
                pat_col = db["patients"]
                encounter = {"date": datetime.utcnow(), "doctor": "OPD", "type": "opd", "details": opd_data}
                # Merge OPD data into patient record
                update_fields = {
                    "lastOpd": opd_data,
                    "lastUpdated": datetime.utcnow().isoformat()
                }
                # Add any specific OPD fields (optometry, iop, etc.) if present
                for key in ["optometry", "iop", "ophthalmicInvestigations", "systemicInvestigations", "specialExaminations"]:
                    if key in opd_data:
                        update_fields[key] = opd_data[key]
                
                pat_col.update_one({"registrationId": reg}, {"$set": update_fields, "$push": {"encounters": encounter}}, upsert=True)

            # Create doctor queue item
            dq = db["doctor_queue"]
            doctor_doc = {
                "appointmentId": appt_id,
                "registrationId": reg,
                "patientName": qobj.get("patientName"),
                "status": "waiting",
                "queueStatus": "Doctor",
                "doctorData": {},
                "opdData": opd_data,
                "receptionData": reception_data,
                "createdAt": datetime.utcnow().isoformat(),
                "updatedAt": datetime.utcnow().isoformat()
            }
            res = dq.insert_one(doctor_doc)
            
            # Update appointment with queueStatus for Operations Hub badge
            try:
                appointments_collection.update_one({"_id": ObjectId(appt_id)}, {"$set": {"queueStatus": "Doctor"}})
            except Exception:
                appointments_collection.update_one({"appointmentId": appt_id}, {"$set": {"queueStatus": "Doctor"}})
            created_doctor = dq.find_one({"_id": res.inserted_id})

            return {"status": "success", "message": "OPD completed, Doctor queued", "doctorQueueId": str(res.inserted_id), "doctorItem": sanitize(created_doctor)}

        updated = oq.find_one({"_id": qobj.get("_id")})
        return {"status": "success", "item": sanitize(updated)}
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error updating OPD queue item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== DOCTOR QUEUE ENDPOINTS ====================

@app.post("/queue/doctor")
async def create_doctor_queue(item: dict = Body(...)):
    """Create a doctor queue item.
    Prevents duplicate queue entries for the same appointment that is still waiting.
    """
    try:
        dq = db["doctor_queue"]
        appt_id = item.get("appointmentId")
        
        # Check if this appointment already has a waiting doctor queue entry
        existing = dq.find_one({
            "appointmentId": appt_id,
            "status": {"$in": ["waiting", "in_progress"]}
        })
        if existing:
            return {
                "status": "duplicate",
                "message": "Appointment already in doctor queue",
                "queueId": str(existing.get("_id")),
                "item": sanitize(existing)
            }
        
        doc = {
            "appointmentId": appt_id,
            "registrationId": item.get("registrationId"),
            "patientName": item.get("patientName"),
            "status": item.get("status", "waiting"),
            "queueStatus": "Doctor",
            "doctorData": item.get("doctorData", {}),
            "opdData": item.get("opdData", {}),
            "receptionData": item.get("receptionData", {}),
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat()
        }
        res = dq.insert_one(doc)

        # sync appointment with queueStatus
        appointments_collection = db["appointments"]
        update_op = {"$set": {"status": "doctor_pending", "queueStatus": "Doctor", "updatedAt": datetime.utcnow().isoformat()}}
        
        matched = False
        try:
            result = appointments_collection.update_one({"_id": ObjectId(doc["appointmentId"])}, update_op, upsert=False)
            if result.matched_count > 0: matched = True
        except Exception:
            pass
            
        if not matched:
            str_appt_id = str(doc["appointmentId"])
            appointments_collection.update_one(
                {"$or": [{"_id": str_appt_id}, {"appointmentId": str_appt_id}]}, 
                update_op, 
                upsert=False
            )

        created = dq.find_one({"_id": res.inserted_id})
        return {"status": "success", "queueId": str(res.inserted_id), "item": sanitize(created)}
    except Exception as e:
        print(f"✗ Error creating doctor queue item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/doctor")
async def list_doctor_queue(status: str | None = None):
    """List doctor queue items."""
    try:
        dq = db["doctor_queue"]
        query = {}
        if status:
            query["status"] = status
        items = list(dq.find(query))
        return {"status": "success", "total": len(items), "items": [sanitize(i) for i in items]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/queue/doctor/{queue_id}")
async def update_doctor_queue(queue_id: str, payload: dict = Body(...)):
    """Update a doctor queue item. If marked done, CREATE A NEW VISIT RECORD and mark discharged.
    This preserves patient history - each visit is a separate record, not an overwrite.
    """
    try:
        dq = db["doctor_queue"]
        try:
            qobj = dq.find_one({"_id": ObjectId(queue_id)})
        except Exception:
            qobj = dq.find_one({"_id": queue_id})

        if not qobj:
            raise HTTPException(status_code=404, detail="Doctor queue item not found")

        updates = {**payload, "updatedAt": datetime.utcnow().isoformat()}
        dq.update_one({"_id": qobj.get("_id")}, {"$set": updates})

        # If doctor completed, update appointment to discharged and CREATE A NEW VISIT RECORD
        if payload.get("status") in ("done", "discharged") or payload.get("action") == "doctor_done":
            appointments_collection = db["appointments"]
            appt_id = qobj.get("appointmentId")
            update_op = {"$set": {"status": "discharged", "queueStatus": "Discharged", "updatedAt": datetime.utcnow().isoformat()}}
            
            matched = False
            try:
                result = appointments_collection.update_one({"_id": ObjectId(appt_id)}, update_op, upsert=False)
                if result.matched_count > 0: matched = True
            except Exception:
                pass
            
            if not matched:
                str_appt_id = str(appt_id)
                appointments_collection.update_one(
                    {"$or": [{"_id": str_appt_id}, {"appointmentId": str_appt_id}]}, 
                    update_op, 
                    upsert=False
                )

            # CREATE A NEW VISIT RECORD (not overwrite)
            reg = qobj.get("registrationId")
            doctor_data = payload.get("doctorData") or qobj.get("doctorData") or {}
            opd_data = qobj.get("opdData") or {}
            reception_data = qobj.get("receptionData") or {}
            
            if reg:
                pat_col = db["patients"]
                now = datetime.utcnow()
                
                # Create a complete visit record with all stages
                visit_record = {
                    "visitId": f"{reg}-{now.strftime('%Y%m%d%H%M%S')}",
                    "visitDate": now.isoformat(),
                    "appointmentId": appt_id,
                    "stages": {
                        "reception": {
                            "stageCompletedAt": reception_data.get("completedAt", now.isoformat()),
                            "data": reception_data
                        },
                        "opd": {
                            "stageCompletedAt": opd_data.get("completedAt", now.isoformat()),
                            "data": opd_data
                        },
                        "doctor": {
                            "stageCompletedAt": now.isoformat(),
                            "data": doctor_data
                        }
                    },
                    "dischargedAt": now.isoformat()
                }
                
                # Push the visit record to the visits array (preserving history)
                # Only update lastVisit timestamp, don't overwrite clinical data
                pat_col.update_one(
                    {"registrationId": reg},
                    {
                        "$push": {"visits": visit_record},
                        "$set": {
                            "lastVisit": now.isoformat(),
                            "lastUpdated": now.isoformat()
                        }
                    },
                    upsert=True
                )
                
                print(f"✓ Visit record saved for patient {reg} - Visit ID: {visit_record['visitId']}")

            return {"status": "success", "message": "Doctor consultation completed, visit record saved, patient discharged"}

        updated = dq.find_one({"_id": qobj.get("_id")})
        return {"status": "success", "item": sanitize(updated)}
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error updating doctor queue item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/patients/{registration_id}")
async def save_patient_examination(registration_id: str, patient_data: dict = Body(...)):
    """
    Simple endpoint to save complete patient examination record.
    Upserts patient record with all examination data in one call.
    """
    try:
        print(f"\n📝 === SAVE PATIENT EXAMINATION ===")
        print(f"   Registration ID: {registration_id}")
        print(f"   Patient name: {patient_data.get('name', 'Unknown')}")
        print(f"   Data keys: {list(patient_data.keys())}")
        
        patients_collection = db["patients"]
        
        # Add metadata
        patient_data["lastUpdated"] = datetime.utcnow().isoformat()
        patient_data["registrationId"] = registration_id
        
        # Upsert - update if exists, insert if new
        result = patients_collection.update_one(
            {"registrationId": registration_id},
            {"$set": patient_data},
            upsert=True
        )
        
        print(f"✓ MongoDB upsert result:")
        print(f"   - Matched: {result.matched_count}")
        print(f"   - Modified: {result.modified_count}")
        print(f"   - Upserted ID: {result.upserted_id}")
        
        return {
            "status": "success",
            "message": f"Patient {registration_id} examination data saved",
            "registrationId": registration_id,
            "upserted": result.upserted_id is not None,
            "modified": result.modified_count > 0
        }
    except Exception as e:
        print(f"✗ Error saving patient examination: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save patient: {str(e)}")


# ==================== VISIT EDIT ENDPOINTS ====================

@app.put("/patients/{registration_id}/visits/{visit_id}")
async def update_visit_record(registration_id: str, visit_id: str, visit_data: dict = Body(...)):
    """
    Update a specific visit record for a patient.
    This allows corrections to be made after a visit has been completed.
    Maintains audit trail by adding edit history.
    """
    try:
        print(f"\n📝 === UPDATE VISIT RECORD ===")
        print(f"   Registration ID: {registration_id}")
        print(f"   Visit ID: {visit_id}")
        
        patients_collection = db["patients"]
        patient = patients_collection.find_one({"registrationId": registration_id})
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        visits = patient.get("visits", [])
        visit_index = None
        
        # Find the visit by visitId
        for idx, visit in enumerate(visits):
            if visit.get("visitId") == visit_id:
                visit_index = idx
                break
        
        if visit_index is None:
            raise HTTPException(status_code=404, detail="Visit not found")
        
        # Get original visit for audit
        original_visit = visits[visit_index]
        
        # Create edit history entry
        edit_entry = {
            "editedAt": datetime.utcnow().isoformat(),
            "editedBy": visit_data.get("editedBy", "Unknown"),
            "reason": visit_data.get("editReason", "Correction"),
            "previousData": original_visit.get("stages", {})
        }
        
        # Update the visit with new data
        updated_visit = {
            **original_visit,
            "stages": visit_data.get("stages", original_visit.get("stages", {})),
            "lastEditedAt": datetime.utcnow().isoformat(),
            "editHistory": original_visit.get("editHistory", []) + [edit_entry]
        }
        
        # Update the visit in the array
        visits[visit_index] = updated_visit
        
        # Save back to database
        result = patients_collection.update_one(
            {"registrationId": registration_id},
            {"$set": {"visits": visits, "lastUpdated": datetime.utcnow().isoformat()}}
        )
        
        print(f"✓ Visit updated successfully")
        
        return {
            "status": "success",
            "message": f"Visit {visit_id} updated for patient {registration_id}",
            "visitId": visit_id
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error updating visit: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update visit: {str(e)}")


# ==================== RECALL PATIENT ENDPOINTS ====================

@app.post("/queue/recall/to-reception")
async def recall_to_reception(payload: dict = Body(...)):
    """
    Recall a patient from OPD queue back to reception queue for corrections.
    Removes from OPD queue and re-adds to reception queue.
    """
    try:
        queue_id = payload.get("queueId")
        reason = payload.get("reason", "Correction needed")
        
        oq = db["opd_queue"]
        rc = db["reception_queue"]
        
        # Find the OPD queue item
        try:
            qobj = oq.find_one({"_id": ObjectId(queue_id)})
        except:
            qobj = oq.find_one({"_id": queue_id})
        
        if not qobj:
            raise HTTPException(status_code=404, detail="OPD queue item not found")
        
        # Create new reception queue item with recall flag
        recall_doc = {
            "appointmentId": qobj.get("appointmentId"),
            "registrationId": qobj.get("registrationId"),
            "patientName": qobj.get("patientName"),
            "status": "waiting",
            "action": "recalled_for_correction",
            "receptionData": qobj.get("receptionData", {}),
            "recallReason": reason,
            "recalledFrom": "opd",
            "recalledAt": datetime.utcnow().isoformat(),
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat()
        }
        
        # Insert into reception queue
        res = rc.insert_one(recall_doc)
        
        # Remove from OPD queue (or mark as recalled)
        oq.update_one({"_id": qobj.get("_id")}, {"$set": {"status": "recalled", "recalledAt": datetime.utcnow().isoformat()}})
        
        # Update appointment status
        appointments_collection = db["appointments"]
        appt_id = qobj.get("appointmentId")
        try:
            appointments_collection.update_one({"_id": ObjectId(appt_id)}, {"$set": {"status": "reception_pending", "recalled": True}})
        except:
            appointments_collection.update_one({"appointmentId": appt_id}, {"$set": {"status": "reception_pending", "recalled": True}})
        
        return {
            "status": "success",
            "message": "Patient recalled to reception",
            "newQueueId": str(res.inserted_id)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error recalling to reception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/queue/recall/to-opd")
async def recall_to_opd(payload: dict = Body(...)):
    """
    Recall a patient from doctor queue back to OPD queue for corrections.
    Removes from doctor queue and re-adds to OPD queue.
    """
    try:
        queue_id = payload.get("queueId")
        reason = payload.get("reason", "Correction needed")
        
        dq = db["doctor_queue"]
        oq = db["opd_queue"]
        
        # Find the doctor queue item
        try:
            qobj = dq.find_one({"_id": ObjectId(queue_id)})
        except:
            qobj = dq.find_one({"_id": queue_id})
        
        if not qobj:
            raise HTTPException(status_code=404, detail="Doctor queue item not found")
        
        # Create new OPD queue item with recall flag
        recall_doc = {
            "appointmentId": qobj.get("appointmentId"),
            "registrationId": qobj.get("registrationId"),
            "patientName": qobj.get("patientName"),
            "status": "waiting",
            "action": "recalled_for_correction",
            "opdData": qobj.get("opdData", {}),
            "receptionData": qobj.get("receptionData", {}),
            "recallReason": reason,
            "recalledFrom": "doctor",
            "recalledAt": datetime.utcnow().isoformat(),
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat()
        }
        
        # Insert into OPD queue
        res = oq.insert_one(recall_doc)
        
        # Remove from doctor queue (or mark as recalled)
        dq.update_one({"_id": qobj.get("_id")}, {"$set": {"status": "recalled", "recalledAt": datetime.utcnow().isoformat()}})
        
        # Update appointment status
        appointments_collection = db["appointments"]
        appt_id = qobj.get("appointmentId")
        try:
            appointments_collection.update_one({"_id": ObjectId(appt_id)}, {"$set": {"status": "opd_pending", "recalled": True}})
        except:
            appointments_collection.update_one({"appointmentId": appt_id}, {"$set": {"status": "opd_pending", "recalled": True}})
        
        return {
            "status": "success",
            "message": "Patient recalled to OPD",
            "newQueueId": str(res.inserted_id)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error recalling to OPD: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ BILLING & INSURANCE ENDPOINTS ============

@app.get("/api/billing/patient/{reg_id}/summary")
def get_billing_summary(reg_id: str):
    """Get billing summary for a patient.
    Returns: {totalOutstanding, totalPaid, pendingClaims, deductibleMet, activeCases}
    """
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        return {"totalOutstanding": 0, "totalPaid": 0, "pendingClaims": 0, "deductibleMet": 0, "activeCases": 0}
    
    invoices = p.get("billing", {}).get("invoices", [])
    
    total_outstanding = sum(inv["patientResponsibility"] for inv in invoices if inv.get("status") != "paid")
    total_paid = sum(inv["patientResponsibility"] for inv in invoices if inv.get("status") == "paid")
    pending_claims = sum(1 for inv in invoices if inv.get("status") == "pending")
    deductible_met = p.get("billing", {}).get("insurance", {}).get("deductibleMet", 0)

    # New: Check for active billing cases (insurance cycles)
    active_cases = billing_cases_collection.count_documents({"registrationId": reg_id, "status": "open"})
    
    return {
        "totalOutstanding": total_outstanding,
        "totalPaid": total_paid,
        "pendingClaims": pending_claims,
        "deductibleMet": deductible_met,
        "activeCases": active_cases
    }


@app.get("/api/billing/patient/{reg_id}/insurance")
def get_patient_insurance(reg_id: str):
    """Get insurance information for a patient."""
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        return None
    
    insurance = p.get("billing", {}).get("insurance", {})
    return {
        "provider": insurance.get("provider", ""),
        "policyNumber": insurance.get("policyNumber", ""),
        "groupNumber": insurance.get("groupNumber", ""),
        "coverageType": insurance.get("coverageType", ""),
        "copay": insurance.get("copay", ""),
        "deductible": insurance.get("deductible", ""),
        "deductibleMet": insurance.get("deductibleMet", 0),
        "outOfPocketMax": insurance.get("outOfPocketMax", ""),
        "outOfPocketMet": insurance.get("outOfPocketMet", 0),
        "effectiveDate": insurance.get("effectiveDate", ""),
        "expirationDate": insurance.get("expirationDate", ""),
        "coverageVerified": insurance.get("coverageVerified", False),
        "lastVerified": insurance.get("lastVerified", "")
    }


@app.put("/api/billing/patient/{reg_id}/insurance")
def update_patient_insurance(reg_id: str, insurance_data: dict = Body(...)):
    """Update insurance information for a patient."""
    try:
        patient_collection.update_one(
            {"registrationId": reg_id},
            {"$set": {
                "billing.insurance": insurance_data,
                "updatedAt": datetime.utcnow().isoformat()
            }},
            upsert=True
        )
        return {"status": "success", "message": "Insurance information updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/billing/patient/{reg_id}/invoices")
def get_patient_invoices(reg_id: str):
    """Get all invoices for a patient."""
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        return []
    
    invoices = p.get("billing", {}).get("invoices", [])
    # Sort by date descending
    return sorted(invoices, key=lambda x: x.get("date", ""), reverse=True)


@app.post("/api/billing/patient/{reg_id}/invoices")
def create_invoice(reg_id: str, invoice_data: dict = Body(...)):
    """Create a new invoice for a patient with insurance and coupon support."""
    try:
        # Handle Coupon Logic
        coupon_code = invoice_data.get("couponCode")
        worker_id = invoice_data.get("appliedBy")
        discount_amount = float(invoice_data.get("discountAmount", 0))

        if coupon_code and worker_id:
            quota = coupon_quota_collection.find_one({"worker_id": worker_id})
            if not quota or quota.get("remaining", 0) <= 0:
                raise HTTPException(status_code=400, detail="No coupons remaining for this worker")
            
            # Deduct from quota
            coupon_quota_collection.update_one(
                {"worker_id": worker_id},
                {"$inc": {"remaining": -1, "used": 1}}
            )

        # Get service items with surgery breakdown if provided
        service_items = invoice_data.get("serviceItems", [])

        invoice = {
            "id": f"INV-{datetime.now().strftime('%Y-%m')}-{str(uuid.uuid4())[:8].upper()}",
            "date": invoice_data.get("date", datetime.now().strftime("%Y-%m-%d")),
            "service": invoice_data.get("service", ""),
            "serviceItems": service_items,  # Store full items including surgery breakdown
            "amount": float(invoice_data.get("amount", 0)),
            "status": invoice_data.get("status", "pending"), # 'paid', 'partially_paid', 'pending'
            "insuranceCovered": float(invoice_data.get("insuranceCovered", 0)),
            "insuranceStatus": invoice_data.get("insuranceStatus", "none"), # 'none', 'pending', 'claimed', 'received'
            "patientResponsibility": float(invoice_data.get("patientResponsibility", 0)),
            "patientPaidAmount": float(invoice_data.get("patientPaidAmount", 0)),
            "couponCode": coupon_code,
            "appliedBy": worker_id,
            "discountAmount": discount_amount,
            "notes": invoice_data.get("notes", ""),
            "createdAt": datetime.utcnow().isoformat(),
            # Doctor/appointment tracking for billing dashboard
            "appointmentId": invoice_data.get("appointmentId", ""),
            "doctorName": invoice_data.get("doctorName", ""),
            # New fields for Multi-stage tracking directly from the UI
            "isSurgeryCase": invoice_data.get("isSurgeryCase", False),
            "expectedFromInsurance": float(invoice_data.get("expectedFromInsurance", 0)),
            "upfrontPaid": float(invoice_data.get("upfrontPaid", 0))
        }

        # Auto-create a Billing Case if this is a surgery/insurance package
        if invoice["isSurgeryCase"] or invoice["expectedFromInsurance"] > 0:
            case_id = f"CASE-{str(uuid.uuid4())[:8].upper()}"
            new_case = {
                "caseId": case_id,
                "registrationId": reg_id,
                "procedureName": invoice["service"],
                "totalEstimatedAmount": invoice["amount"],
                "insuranceApprovedAmount": invoice["expectedFromInsurance"],
                "preSurgeryPaidAmount": invoice["upfrontPaid"],
                "status": "open",
                "createdAt": datetime.utcnow().isoformat(),
                "updatedAt": datetime.utcnow().isoformat(),
                "stages": [
                    {
                        "name": "insurance_approval", 
                        "status": "approved" if invoice["expectedFromInsurance"] > 0 else "none", 
                        "amount": invoice["expectedFromInsurance"], 
                        "date": datetime.utcnow().isoformat()
                    },
                    {
                        "name": "pre_surgery", 
                        "status": "paid" if invoice["upfrontPaid"] > 0 else "pending", 
                        "amount": invoice["upfrontPaid"], 
                        "date": datetime.utcnow().isoformat()
                    },
                    {
                        "name": "final_settlement", 
                        "status": "pending", 
                        "amount": invoice["patientResponsibility"] - invoice["upfrontPaid"], 
                        "date": datetime.utcnow().isoformat()
                    }
                ]
            }
            billing_cases_collection.insert_one(new_case)
            invoice["linkedCaseId"] = case_id
        
        patient_collection.update_one(
            {"registrationId": reg_id},
            {"$push": {"billing.invoices": invoice}},
            upsert=True
        )
        
        return {"status": "success", "invoiceId": invoice["id"]}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ SURGERY BILLING (TWO-BILL SYSTEM) ============

@app.post("/api/billing/patient/{reg_id}/surgery-bills/initial")
def create_initial_surgery_bill(reg_id: str, bill_data: dict = Body(...)):
    """
    Create an Initial/Provisional Surgery Bill with Security Deposit.
    This is the first bill generated before surgery when insurance is involved.
    Stores the bill in MongoDB under billing.surgeryBills array.
    """
    try:
        bill_id = f"SB-INIT-{datetime.now().strftime('%Y%m')}-{str(uuid.uuid4())[:8].upper()}"
        
        initial_bill = {
            "billId": bill_id,
            "billType": "initial",  # 'initial' or 'final'
            "status": "pending_approval",  # 'pending_approval', 'approved', 'settled'
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat(),
            
            # Patient Details
            "registrationId": reg_id,
            "patientName": bill_data.get("patientName", ""),
            
            # Surgery Details
            "surgeryName": bill_data.get("surgeryName", ""),
            "surgeryBreakdown": bill_data.get("surgeryBreakdown", []),
            "totalSurgeryCost": float(bill_data.get("totalSurgeryCost", 0)),
            
            # Insurance Details
            "hasInsurance": bill_data.get("hasInsurance", False),
            "insuranceType": bill_data.get("insuranceType", ""),  # CGHS, SGHS, PRIVATE
            "insuranceCompany": bill_data.get("insuranceCompany", ""),
            "insuranceTPA": bill_data.get("insuranceTPA", ""),
            "estimatedInsuranceCoverage": float(bill_data.get("estimatedInsuranceCoverage", 0)),
            
            # Security Deposit (Upfront Amount)
            "securityDeposit": float(bill_data.get("securityDeposit", 0)),
            "securityDepositPaid": bill_data.get("securityDepositPaid", False),
            "securityDepositPaymentMethod": bill_data.get("securityDepositPaymentMethod", ""),
            "securityDepositDate": bill_data.get("securityDepositDate", ""),
            
            # Amounts Summary (at initial stage)
            "estimatedPatientShare": float(bill_data.get("estimatedPatientShare", 0)),
            
            # Notes
            "notes": bill_data.get("notes", "Insurance Approval Pending"),
            "createdBy": bill_data.get("createdBy", ""),
        }
        
        # Store in patient's billing.surgeryBills array
        patient_collection.update_one(
            {"registrationId": reg_id},
            {"$push": {"billing.surgeryBills": initial_bill}},
            upsert=True
        )
        
        return {
            "status": "success",
            "billId": bill_id,
            "billType": "initial",
            "message": "Initial Surgery Bill created successfully. Insurance approval pending."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/billing/patient/{reg_id}/surgery-bills/final")
def create_final_surgery_bill(reg_id: str, bill_data: dict = Body(...)):
    """
    Create a Final Settlement Surgery Bill after insurance approval.
    This is the second bill generated after insurance approval comes.
    Auto-calculates patient balance or refund based on:
    - Total surgery cost
    - Insurance approved amount
    - Security deposit already paid
    """
    try:
        # Get the initial bill to reference
        initial_bill_id = bill_data.get("initialBillId", "")
        
        bill_id = f"SB-FINAL-{datetime.now().strftime('%Y%m')}-{str(uuid.uuid4())[:8].upper()}"
        
        # Extract amounts
        total_surgery_cost = float(bill_data.get("totalSurgeryCost", 0))
        insurance_approved_amount = float(bill_data.get("insuranceApprovedAmount", 0))
        security_deposit_paid = float(bill_data.get("securityDepositPaid", 0))
        
        # Auto-calculate final amounts
        patient_total_share = total_surgery_cost - insurance_approved_amount
        balance_or_refund = patient_total_share - security_deposit_paid
        
        # Determine if patient needs to pay more or gets a refund
        payment_status = "balance_due" if balance_or_refund > 0 else ("refund_due" if balance_or_refund < 0 else "settled")
        
        final_bill = {
            "billId": bill_id,
            "billType": "final",
            "linkedInitialBillId": initial_bill_id,
            "status": payment_status,
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat(),
            
            # Patient Details
            "registrationId": reg_id,
            "patientName": bill_data.get("patientName", ""),
            
            # Surgery Details
            "surgeryName": bill_data.get("surgeryName", ""),
            "surgeryBreakdown": bill_data.get("surgeryBreakdown", []),
            "totalSurgeryCost": total_surgery_cost,
            
            # Insurance Details
            "hasInsurance": bill_data.get("hasInsurance", True),
            "insuranceType": bill_data.get("insuranceType", ""),
            "insuranceCompany": bill_data.get("insuranceCompany", ""),
            "insuranceTPA": bill_data.get("insuranceTPA", ""),
            "insuranceApprovedAmount": insurance_approved_amount,
            "insuranceClaimReference": bill_data.get("insuranceClaimReference", ""),
            "insuranceApprovalDate": bill_data.get("insuranceApprovalDate", ""),
            
            # Payment Summary
            "securityDepositPaid": security_deposit_paid,
            "patientTotalShare": patient_total_share,
            "balancePayable": max(0, balance_or_refund),  # If positive, patient pays this
            "refundAmount": abs(min(0, balance_or_refund)),  # If negative, patient gets this back
            
            # Final Payment Details (to be updated when paid)
            "finalPaymentAmount": float(bill_data.get("finalPaymentAmount", 0)),
            "finalPaymentMethod": bill_data.get("finalPaymentMethod", ""),
            "finalPaymentDate": bill_data.get("finalPaymentDate", ""),
            
            # Notes
            "notes": bill_data.get("notes", ""),
            "createdBy": bill_data.get("createdBy", ""),
        }
        
        # Store in patient's billing.surgeryBills array
        patient_collection.update_one(
            {"registrationId": reg_id},
            {"$push": {"billing.surgeryBills": final_bill}},
            upsert=True
        )
        
        # Update the initial bill status if linked
        if initial_bill_id:
            patient_collection.update_one(
                {"registrationId": reg_id, "billing.surgeryBills.billId": initial_bill_id},
                {"$set": {
                    "billing.surgeryBills.$.status": "settled",
                    "billing.surgeryBills.$.linkedFinalBillId": bill_id,
                    "billing.surgeryBills.$.updatedAt": datetime.utcnow().isoformat()
                }}
            )
        
        return {
            "status": "success",
            "billId": bill_id,
            "billType": "final",
            "calculation": {
                "totalSurgeryCost": total_surgery_cost,
                "insuranceApprovedAmount": insurance_approved_amount,
                "patientTotalShare": patient_total_share,
                "securityDepositPaid": security_deposit_paid,
                "balancePayable": max(0, balance_or_refund),
                "refundAmount": abs(min(0, balance_or_refund))
            },
            "message": "Final Settlement Bill created successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/billing/patient/{reg_id}/surgery-bills")
def get_patient_surgery_bills(reg_id: str):
    """Get all surgery bills (initial and final) for a patient."""
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        return []
    
    surgery_bills = p.get("billing", {}).get("surgeryBills", [])
    # Sort by createdAt descending
    return sorted(surgery_bills, key=lambda x: x.get("createdAt", ""), reverse=True)


@app.get("/api/billing/patient/{reg_id}/surgery-bills/{bill_id}")
def get_surgery_bill_by_id(reg_id: str, bill_id: str):
    """Get a specific surgery bill by ID."""
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    surgery_bills = p.get("billing", {}).get("surgeryBills", [])
    for bill in surgery_bills:
        if bill.get("billId") == bill_id:
            return bill
    
    raise HTTPException(status_code=404, detail="Surgery bill not found")


@app.put("/api/billing/patient/{reg_id}/surgery-bills/{bill_id}")
def update_surgery_bill(reg_id: str, bill_id: str, update_data: dict = Body(...)):
    """Update a surgery bill (e.g., mark security deposit as paid, update status)."""
    try:
        # Build update fields
        update_fields = {}
        for key, value in update_data.items():
            update_fields[f"billing.surgeryBills.$.{key}"] = value
        
        update_fields["billing.surgeryBills.$.updatedAt"] = datetime.utcnow().isoformat()
        
        result = patient_collection.update_one(
            {"registrationId": reg_id, "billing.surgeryBills.billId": bill_id},
            {"$set": update_fields}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Surgery bill not found")
        
        return {"status": "success", "message": "Surgery bill updated successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ COUPON QUOTA ENDPOINTS ============

@app.get("/api/coupons/quota/{worker_id}")
def get_worker_coupon_quota(worker_id: str):
    """Get remaining coupons for a worker."""
    quota = coupon_quota_collection.find_one({"worker_id": worker_id})
    if not quota:
        # Initialize if not exists
        default_quota = {"worker_id": worker_id, "limit": 10, "remaining": 10, "used": 0}
        result = coupon_quota_collection.insert_one(default_quota)
        default_quota["id"] = str(result.inserted_id)
        return default_quota
    
    # Convert MongoDB ObjectId to string for JSON serialization
    if "_id" in quota:
        quota["id"] = str(quota["_id"])
        del quota["_id"]
    return quota

@app.post("/api/coupons/refresh")
def refresh_worker_quota(data: dict = Body(...)):
    """Admin endpoint to refresh worker coupon quotas."""
    worker_id = data.get("workerId")
    new_limit = data.get("limit", 10)
    refreshed_by = data.get("refreshedBy") # CEO or Main Doctor

    if not worker_id or not refreshed_by:
        raise HTTPException(status_code=400, detail="Worker ID and Refreshed By are required")

    coupon_quota_collection.update_one(
        {"worker_id": worker_id},
        {"$set": {
            "limit": new_limit,
            "remaining": new_limit,
            "last_refreshed_by": refreshed_by,
            "last_refreshed_at": datetime.utcnow().isoformat()
        }},
        upsert=True
    )
    return {"status": "success", "message": f"Quota refreshed for {worker_id}"}


@app.get("/api/billing/patient/{reg_id}/payments")
def get_patient_payments(reg_id: str):
    """Get all payments for a patient."""
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        return []
    
    payments = p.get("billing", {}).get("payments", [])
    # Sort by date descending
    return sorted(payments, key=lambda x: x.get("date", ""), reverse=True)


@app.post("/api/billing/patient/{reg_id}/payments")
def record_payment(reg_id: str, payment_data: dict = Body(...)):
    """Record a payment for a patient."""
    try:
        payment = {
            "id": f"PAY-{str(uuid.uuid4())[:8].upper()}",
            "date": payment_data.get("date", datetime.now().strftime("%Y-%m-%d")),
            "amount": float(payment_data.get("amount", 0)),
            "method": payment_data.get("method", ""),
            "invoiceId": payment_data.get("invoiceId", ""),
            "caseId": payment_data.get("caseId", ""), # Link to a multi-stage case
            "stageName": payment_data.get("stageName", ""), # e.g. 'pre_surgery'
            "notes": payment_data.get("notes", ""),
            "createdAt": datetime.utcnow().isoformat()
        }
        
        patient_collection.update_one(
            {"registrationId": reg_id},
            {"$push": {"billing.payments": payment}},
            upsert=True
        )

        # Update Billing Case stage if linked
        if payment["caseId"] and payment["stageName"]:
            case = billing_cases_collection.find_one({"caseId": payment["caseId"]})
            if case:
                stages = case.get("stages", [])
                for stage in stages:
                    if stage["name"] == payment["stageName"]:
                        stage["status"] = "paid"
                        stage["amount"] = payment["amount"]
                        stage["date"] = datetime.utcnow().isoformat()
                        break
                
                update_fields = {"stages": stages, "updatedAt": datetime.utcnow().isoformat()}
                if payment["stageName"] == "pre_surgery":
                    update_fields["preSurgeryPaidAmount"] = payment["amount"]
                
                billing_cases_collection.update_one(
                    {"caseId": payment["caseId"]},
                    {"$set": update_fields}
                )
        
        # Update invoice status if all paid
        if payment_data.get("invoiceId"):
            p = patient_collection.find_one({"registrationId": reg_id})
            invoices = p.get("billing", {}).get("invoices", [])
            payments = p.get("billing", {}).get("payments", [])
            
            # Find invoice and update if fully paid
            for inv in invoices:
                if inv.get("id") == payment_data.get("invoiceId"):
                    inv_total = inv.get("patientResponsibility", 0)
                    paid = sum(pay["amount"] for pay in payments if pay.get("invoiceId") == inv.get("id"))
                    if paid >= inv_total:
                        inv["status"] = "paid"
                    break
            
            patient_collection.update_one(
                {"registrationId": reg_id},
                {"$set": {"billing.invoices": invoices}}
            )
        
        return {"status": "success", "paymentId": payment["id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ MULTI-STAGE BILLING (INSURANCE & SURGERY) ============

@app.post("/api/billing/cases")
def create_billing_case(case_data: BillingCase):
    """Create a new multi-stage billing case (e.g. for surgery)."""
    try:
        case_dict = case_data.model_dump()
        case_dict["createdAt"] = datetime.utcnow().isoformat()
        case_dict["updatedAt"] = datetime.utcnow().isoformat()
        
        # Ensure stages are initialized if empty
        if not case_dict.get("stages"):
            case_dict["stages"] = [
                {"name": "insurance_approval", "status": "pending", "amount": 0, "date": datetime.utcnow().isoformat()},
                {"name": "pre_surgery", "status": "pending", "amount": 0, "date": datetime.utcnow().isoformat()},
                {"name": "final_settlement", "status": "pending", "amount": 0, "date": datetime.utcnow().isoformat()}
            ]
            
        result = billing_cases_collection.insert_one(case_dict)
        return {"status": "success", "caseId": case_dict["caseId"], "id": str(result.inserted_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/billing/cases/patient/{reg_id}")
def get_patient_billing_cases(reg_id: str):
    """Get all billing cases for a patient."""
    cases = list(billing_cases_collection.find({"registrationId": reg_id}))
    for case in cases:
        case["_id"] = str(case["_id"])
    return cases

@app.put("/api/billing/cases/{case_id}/stage")
def update_billing_stage(case_id: str, stage_update: dict = Body(...)):
    """Update a specific stage in a billing case."""
    # stage_update should have: stageName, status, amount, updatedBy, notes
    try:
        case = billing_cases_collection.find_one({"caseId": case_id})
        if not case:
            raise HTTPException(status_code=404, detail="Billing case not found")
        
        stage_name = stage_update.get("stageName")
        stages = case.get("stages", [])
        
        updated = False
        for stage in stages:
            if stage["name"] == stage_name:
                stage["status"] = stage_update.get("status", stage["status"])
                stage["amount"] = float(stage_update.get("amount", stage["amount"]))
                stage["notes"] = stage_update.get("notes", stage.get("notes"))
                stage["updatedBy"] = stage_update.get("updatedBy")
                stage["date"] = datetime.utcnow().isoformat()
                updated = True
                
                # Update high-level case fields based on stage
                if stage_name == "insurance_approval" and stage["status"] == "approved":
                    case["insuranceApprovedAmount"] = stage["amount"]
                elif stage_name == "pre_surgery" and stage["status"] == "paid":
                    case["preSurgeryPaidAmount"] = stage["amount"]
                break
        
        if not updated:
            # If stage doesn't exist, add it
            new_stage = {
                "name": stage_name,
                "status": stage_update.get("status", "pending"),
                "amount": float(stage_update.get("amount", 0)),
                "notes": stage_update.get("notes"),
                "updatedBy": stage_update.get("updatedBy"),
                "date": datetime.utcnow().isoformat()
            }
            stages.append(new_stage)
            
        billing_cases_collection.update_one(
            {"caseId": case_id},
            {"$set": {
                "stages": stages,
                "insuranceApprovedAmount": case.get("insuranceApprovedAmount", 0),
                "preSurgeryPaidAmount": case.get("preSurgeryPaidAmount", 0),
                "updatedAt": datetime.utcnow().isoformat()
            }}
        )
        
        return {"status": "success", "message": f"Stage {stage_name} updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/billing/patient/{reg_id}/claims")
def get_insurance_claims(reg_id: str):
    """Get insurance claims for a patient."""
    p = patient_collection.find_one({"registrationId": reg_id})
    if not p:
        return []
    
    claims = p.get("billing", {}).get("claims", [])
    return sorted(claims, key=lambda x: x.get("dateFiled", ""), reverse=True)


@app.post("/api/billing/patient/{reg_id}/claims")
def create_claim(reg_id: str, claim_data: dict = Body(...)):
    """Create an insurance claim for a patient."""
    try:
        claim = {
            "id": f"CLM-{datetime.now().strftime('%Y-%m')}-{str(uuid.uuid4())[:8].upper()}",
            "dateFiled": claim_data.get("dateFiled", datetime.now().strftime("%Y-%m-%d")),
            "service": claim_data.get("service", ""),
            "billed": float(claim_data.get("billed", 0)),
            "approved": float(claim_data.get("approved", 0)),
            "status": claim_data.get("status", "pending"),
            "invoiceId": claim_data.get("invoiceId", ""),
            "notes": claim_data.get("notes", ""),
            "createdAt": datetime.utcnow().isoformat()
        }
        
        patient_collection.update_one(
            {"registrationId": reg_id},
            {"$push": {"billing.claims": claim}},
            upsert=True
        )
        
        return {"status": "success", "claimId": claim["id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== PHARMACY ENDPOINTS ====================

@app.get("/pharmacy/medicines")
async def get_pharmacy_medicines(category: Optional[str] = None):
    """
    Get all pharmacy medicines from inventory.
    Optional category filter: 'Eye Drops', 'Tablets', 'Ointments', 'Contact Lens', 'Surgical'
    """
    try:
        query = {}
        if category:
            # Case-insensitive category match
            query = {"category": {"$regex": f"^{category}$", "$options": "i"}}
        
        medicines = list(pharmacy_collection.find(query))
        
        # Sanitize ObjectId and format response
        result = []
        for med in medicines:
            result.append({
                "id": str(med.get("_id")),
                "name": med.get("name"),
                "category": med.get("category"),
                "price": float(med.get("price", 0)),
                "stock": int(med.get("stock", 0)),
                "description": med.get("description"),
                "manufacturer": med.get("manufacturer"),
                "batch_number": med.get("batch_number"),
                "expiry_date": med.get("expiry_date"),
                "reorder_level": int(med.get("reorder_level", 10))
            })
        
        print(f"✓ Fetched {len(result)} medicines" + (f" (category: {category})" if category else ""))
        return {"status": "success", "total": len(result), "medicines": result}
    
    except Exception as e:
        print(f"✗ Error fetching medicines: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch medicines: {str(e)}")


@app.get("/pharmacy/medicines/{medicine_id}")
async def get_medicine(medicine_id: str):
    """
    Get a specific medicine by ID.
    """
    try:
        med = None
        try:
            med = pharmacy_collection.find_one({"_id": ObjectId(medicine_id)})
        except:
            med = pharmacy_collection.find_one({"id": medicine_id})
        
        if not med:
            raise HTTPException(status_code=404, detail="Medicine not found")
        
        return {
            "id": str(med.get("_id")),
            "name": med.get("name"),
            "category": med.get("category"),
            "price": float(med.get("price", 0)),
            "stock": int(med.get("stock", 0)),
            "description": med.get("description"),
            "manufacturer": med.get("manufacturer"),
            "batch_number": med.get("batch_number"),
            "expiry_date": med.get("expiry_date"),
            "reorder_level": int(med.get("reorder_level", 10))
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pharmacy/medicines")
async def create_medicine(medicine_data: dict = Body(...)):
    """
    Create a new medicine in the pharmacy inventory.
    """
    try:
        med = {
            "name": medicine_data.get("name"),
            "category": medicine_data.get("category"),
            "price": float(medicine_data.get("price", 0)),
            "stock": int(medicine_data.get("stock", 0)),
            "description": medicine_data.get("description", ""),
            "manufacturer": medicine_data.get("manufacturer", ""),
            "batch_number": medicine_data.get("batch_number", ""),
            "expiry_date": medicine_data.get("expiry_date", ""),
            "reorder_level": int(medicine_data.get("reorder_level", 10)),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        result = pharmacy_collection.insert_one(med)
        print(f"✓ Medicine created: {med['name']} (ID: {result.inserted_id})")
        
        return {
            "status": "success",
            "message": f"Medicine {med['name']} added to inventory",
            "medicineId": str(result.inserted_id)
        }
    except Exception as e:
        print(f"✗ Error creating medicine: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/pharmacy/medicines/{medicine_id}")
async def update_medicine(medicine_id: str, medicine_data: dict = Body(...)):
    """
    Update medicine details in the pharmacy inventory.
    """
    try:
        update_data = {**medicine_data, "updated_at": datetime.utcnow().isoformat()}
        
        try:
            result = pharmacy_collection.update_one({"_id": ObjectId(medicine_id)}, {"$set": update_data})
        except:
            result = pharmacy_collection.update_one({"id": medicine_id}, {"$set": update_data})
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Medicine not found")
        
        print(f"✓ Medicine updated: {medicine_id}")
        return {"status": "success", "message": "Medicine updated"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/pharmacy/medicines/{medicine_id}/stock")
async def update_medicine_stock(medicine_id: str, payload: dict = Body(...)):
    """
    Update medicine stock after a sale/billing.
    Expects: { quantity_sold: int }
    """
    try:
        quantity_sold = int(payload.get("quantity_sold", 0))
        
        # Get current medicine
        med = None
        try:
            med = pharmacy_collection.find_one({"_id": ObjectId(medicine_id)})
        except:
            med = pharmacy_collection.find_one({"id": medicine_id})
        
        if not med:
            raise HTTPException(status_code=404, detail="Medicine not found")
        
        current_stock = int(med.get("stock", 0))
        
        # Check if enough stock available
        if current_stock < quantity_sold:
            return {
                "status": "error",
                "message": f"Insufficient stock. Available: {current_stock}, Requested: {quantity_sold}"
            }
        
        # Update stock (decrement)
        new_stock = current_stock - quantity_sold
        try:
            pharmacy_collection.update_one(
                {"_id": ObjectId(medicine_id)},
                {"$set": {"stock": new_stock, "updated_at": datetime.utcnow().isoformat()}}
            )
        except:
            pharmacy_collection.update_one(
                {"id": medicine_id},
                {"$set": {"stock": new_stock, "updated_at": datetime.utcnow().isoformat()}}
            )
        
        print(f"✓ Medicine stock updated: {medicine_id} | Old: {current_stock}, New: {new_stock}")
        
        return {
            "status": "success",
            "message": f"Stock updated from {current_stock} to {new_stock}",
            "previousStock": current_stock,
            "newStock": new_stock,
            "quantitySold": quantity_sold
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error updating stock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pharmacy/billing")
async def create_pharmacy_bill(billing_data: dict = Body(...)):
    """
    Create and save a pharmacy billing record.
    Records medicine sales, updates inventory, and tracks patient purchases.
    
    Expected payload:
    {
        "registrationId": "REG-2025-123456",
        "patientName": "John Doe",
        "items": [
            {"medicineId": "...", "name": "...", "quantity": 2, "price": 100, "total": 200}
        ],
        "totalAmount": 500,
        "paymentMethod": "cash|card|insurance"
    }
    """
    try:
        registration_id = billing_data.get("registrationId")
        patient_name = billing_data.get("patientName", "Unknown")
        items = billing_data.get("items", [])
        total_amount = float(billing_data.get("totalAmount", 0))
        payment_method = billing_data.get("paymentMethod", "cash")
        
        if not registration_id:
            raise HTTPException(status_code=400, detail="registrationId is required")
        
        if not items or len(items) == 0:
            raise HTTPException(status_code=400, detail="At least one medicine item is required")
        
        # Validate all items have sufficient stock before committing any updates
        for item in items:
            medicine_id = item.get("medicineId")
            quantity = int(item.get("quantity", 0))
            
            med = None
            try:
                med = pharmacy_collection.find_one({"_id": ObjectId(medicine_id)})
            except:
                med = pharmacy_collection.find_one({"id": medicine_id})
            
            if not med:
                raise HTTPException(status_code=404, detail=f"Medicine {medicine_id} not found")
            
            available_stock = int(med.get("stock", 0))
            if available_stock < quantity:
                return {
                    "status": "error",
                    "message": f"Insufficient stock for {med.get('name', 'Unknown medicine')}. Available: {available_stock}, Requested: {quantity}"
                }
        
        # Create billing record
        bill_id = f"BILL-{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:6].upper()}"
        
        bill_record = {
            "billId": bill_id,
            "registrationId": registration_id,
            "patientName": patient_name,
            "items": items,
            "totalAmount": total_amount,
            "paymentMethod": payment_method,
            "status": "completed",
            "billDate": datetime.utcnow().isoformat(),
            "createdAt": datetime.utcnow().isoformat()
        }
        
        # Save billing record
        result = pharmacy_billing_collection.insert_one(bill_record)
        
        print(f"✓ Pharmacy bill created: {bill_id}")
        
        # Update medicine stock for each item
        for item in items:
            medicine_id = item.get("medicineId")
            quantity_sold = int(item.get("quantity", 0))
            
            try:
                pharmacy_collection.update_one(
                    {"_id": ObjectId(medicine_id)},
                    {"$inc": {"stock": -quantity_sold}, "$set": {"updated_at": datetime.utcnow().isoformat()}}
                )
            except:
                pharmacy_collection.update_one(
                    {"id": medicine_id},
                    {"$inc": {"stock": -quantity_sold}, "$set": {"updated_at": datetime.utcnow().isoformat()}}
                )
            
            print(f"  - Stock updated for medicine {medicine_id}: -{quantity_sold}")
        
        # Update patient's pharmacy billing history
        patient_collection.update_one(
            {"registrationId": registration_id},
            {
                "$push": {"pharmacyBills": bill_record},
                "$set": {"lastPharmacyPurchase": datetime.utcnow().isoformat()}
            },
            upsert=True
        )
        
        print(f"✓ Patient {registration_id} pharmacy history updated")
        
        return {
            "status": "success",
            "message": f"Pharmacy bill created successfully",
            "billId": bill_id,
            "totalAmount": total_amount,
            "itemsCount": len(items)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error creating pharmacy bill: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create pharmacy bill: {str(e)}")


@app.get("/pharmacy/billing/{bill_id}")
async def get_pharmacy_bill(bill_id: str):
    """
    Get a specific pharmacy billing record.
    """
    try:
        bill = pharmacy_billing_collection.find_one({"billId": bill_id})
        
        if not bill:
            raise HTTPException(status_code=404, detail="Bill not found")
        
        return {
            "billId": bill.get("billId"),
            "registrationId": bill.get("registrationId"),
            "patientName": bill.get("patientName"),
            "items": bill.get("items", []),
            "totalAmount": float(bill.get("totalAmount", 0)),
            "paymentMethod": bill.get("paymentMethod"),
            "status": bill.get("status"),
            "billDate": bill.get("billDate"),
            "createdAt": bill.get("createdAt")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pharmacy/billing/patient/{registration_id}")
async def get_patient_pharmacy_bills(registration_id: str):
    """
    Get all pharmacy billing records for a specific patient.
    """
    try:
        bills = list(pharmacy_billing_collection.find({"registrationId": registration_id}).sort("billDate", -1))
        
        result = []
        for bill in bills:
            result.append({
                "billId": bill.get("billId"),
                "registrationId": bill.get("registrationId"),
                "patientName": bill.get("patientName"),
                "items": bill.get("items", []),
                "totalAmount": float(bill.get("totalAmount", 0)),
                "paymentMethod": bill.get("paymentMethod"),
                "status": bill.get("status"),
                "billDate": bill.get("billDate"),
                "createdAt": bill.get("createdAt")
            })
        
        print(f"✓ Fetched {len(result)} pharmacy bills for patient {registration_id}")
        
        return {
            "status": "success",
            "registrationId": registration_id,
            "totalBills": len(result),
            "bills": result
        }
    
    except Exception as e:
        print(f"✗ Error fetching pharmacy bills: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ AGGREGATED BILLING DASHBOARD ENDPOINT ============

@app.get("/api/billing/dashboard/stats")
def get_billing_dashboard_stats():
    """Get aggregated billing statistics for the dashboard.
    Returns aggregated data across ALL patients for KPI cards and billing records table.
    """
    try:
        from datetime import datetime, timedelta
        
        # Get all patients with billing info
        all_patients = list(patient_collection.find({"billing.invoices": {"$exists": True}}))
        
        # Build appointment lookup cache for doctor names
        appointments_collection = db["appointments"]
        all_appointments = list(appointments_collection.find({}))
        appt_lookup = {}
        for appt in all_appointments:
            appt_id = str(appt.get("_id", ""))
            appt_lookup[appt_id] = appt.get("doctorName", "")
            # Also index by registrationId for fallback
            reg = appt.get("registrationId", "")
            if reg and reg not in appt_lookup:
                appt_lookup[f"reg_{reg}"] = appt.get("doctorName", "")
        
        today = datetime.utcnow().date()
        total_revenue = 0
        pending_bills_count = 0
        completed_today_count = 0
        refunds_total = 0
        all_billing_records = []
        
        # Process each patient
        for patient in all_patients:
            invoices = patient.get("billing", {}).get("invoices", [])
            
            for invoice in invoices:
                # Build billing record for table
                invoice_date_str = invoice.get("date", "")
                try:
                    invoice_date = datetime.strptime(invoice_date_str, "%Y-%m-%d").date()
                except:
                    invoice_date = today
                
                # KPI calculations
                status = invoice.get("status", "pending")
                patient_responsibility = float(invoice.get("patientResponsibility", 0))
                
                if status == "paid":
                    total_revenue += patient_responsibility
                    # Check if paid today
                    created_at_str = invoice.get("createdAt", "")
                    try:
                        created_at = datetime.fromisoformat(created_at_str).date()
                        if created_at == today:
                            completed_today_count += 1
                    except:
                        pass
                else:
                    pending_bills_count += 1
                
                # Track refunds (if status is 'refunded' or amount is negative)
                if status == "refunded" or patient_responsibility < 0:
                    refunds_total += abs(patient_responsibility)
                
                # Build record for table
                # Lookup doctor name from appointment
                appt_id = invoice.get("appointmentId", "")
                reg_id = patient.get("registrationId", "")
                doctor_name = appt_lookup.get(appt_id, "") or appt_lookup.get(f"reg_{reg_id}", "") or invoice.get("doctorName", "N/A")
                
                record = {
                    "id": invoice.get("id", ""),
                    "type": invoice.get("service", "OPD"),
                    "checkInTime": invoice_date_str,
                    "patientName": patient.get("name", "Unknown"),
                    "registrationId": patient.get("registrationId", "N/A"),
                    "age": str(patient.get("demographics", {}).get("age", "N/A")),
                    "sex": patient.get("demographics", {}).get("sex", "N/A"),
                    "phone": patient.get("contactInfo", {}).get("phone", "N/A"),
                    "refDoctor": "Self",
                    "visitType": "New",
                    "visitReason": invoice.get("service", "General Checkup"),
                    "doctorName": doctor_name,
                    "optomName": "Optom. Jane",
                    "followUpDate": "",
                    "waitingTime": "0 min",
                    "status": "Completed" if status == "paid" else "Waiting",
                    "paymentStatus": status,
                    "insuranceStatus": invoice.get("insuranceStatus", "none"),
                    "notes": invoice.get("notes", ""),
                    "dilationStatus": "Not Started",
                    "amount": patient_responsibility,
                    "insuranceCovered": float(invoice.get("insuranceCovered", 0))
                }
                all_billing_records.append(record)
        
        # Sort records by date descending
        all_billing_records.sort(key=lambda x: x["checkInTime"], reverse=True)
        
        return {
            "status": "success",
            "totalRevenue": round(total_revenue, 2),
            "pendingBills": pending_bills_count,
            "completedToday": completed_today_count,
            "refunds": round(refunds_total, 2),
            "records": all_billing_records,
            "totalRecords": len(all_billing_records)
        }
    except Exception as e:
        print(f"Error fetching billing dashboard stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pharmacy/stock-report")
async def get_stock_report():
    """
    Get overall pharmacy inventory status report.
    Shows medicines below reorder level, expiring soon, etc.
    """
    try:
        medicines = list(pharmacy_collection.find())
        
        low_stock = []
        total_value = 0
        
        for med in medicines:
            stock = int(med.get("stock", 0))
            reorder_level = int(med.get("reorder_level", 10))
            price = float(med.get("price", 0))
            
            total_value += (stock * price)
            
            if stock <= reorder_level:
                low_stock.append({
                    "id": str(med.get("_id")),
                    "name": med.get("name"),
                    "stock": stock,
                    "reorderLevel": reorder_level,
                    "needed": max(0, reorder_level - stock)
                })
        
        return {
            "status": "success",
            "totalMedicines": len(medicines),
            "totalInventoryValue": round(total_value, 2),
            "lowStockMedicines": low_stock,
            "lowStockCount": len(low_stock)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pharmacy/initialize-sample-data")
async def initialize_sample_pharmacy_data():
    """
    Initialize pharmacy collection with sample medicines.
    Used for demo/testing purposes.
    """
    try:
        # Check if pharmacy collection already has data
        count = pharmacy_collection.count_documents({})
        if count > 0:
            return {
                "status": "info",
                "message": f"Pharmacy collection already has {count} medicines. Skipping initialization."
            }
        
        sample_medicines = [
            {
                "name": "Refresh Tears",
                "category": "Eye Drops",
                "price": 150.00,
                "stock": 50,
                "description": "Artificial tears for dry eyes",
                "manufacturer": "Allergan",
                "batch_number": "AB-2025-001",
                "expiry_date": "2026-12-31",
                "reorder_level": 10
            },
            {
                "name": "Chloramphenicol Eye Drop",
                "category": "Eye Drops",
                "price": 80.00,
                "stock": 75,
                "description": "Antibiotic eye drops",
                "manufacturer": "Cipla",
                "batch_number": "CB-2025-045",
                "expiry_date": "2026-06-30",
                "reorder_level": 15
            },
            {
                "name": "Paracetamol 500mg",
                "category": "Tablets",
                "price": 25.00,
                "stock": 200,
                "description": "Pain reliever and fever reducer",
                "manufacturer": "Medley",
                "batch_number": "TB-2025-234",
                "expiry_date": "2026-10-15",
                "reorder_level": 50
            },
            {
                "name": "Cefixime 200mg",
                "category": "Tablets",
                "price": 120.00,
                "stock": 40,
                "description": "Antibiotic for bacterial infections",
                "manufacturer": "Lupin",
                "batch_number": "TB-2025-567",
                "expiry_date": "2026-08-20",
                "reorder_level": 20
            },
            {
                "name": "Tetracycline Ointment",
                "category": "Ointments",
                "price": 95.00,
                "stock": 60,
                "description": "Antibiotic ointment for eye infections",
                "manufacturer": "Pfizer",
                "batch_number": "OI-2025-123",
                "expiry_date": "2026-05-12",
                "reorder_level": 12
            },
            {
                "name": "Fluorometholone Ointment",
                "category": "Ointments",
                "price": 250.00,
                "stock": 30,
                "description": "Steroid ointment for inflammation",
                "manufacturer": "Sun Pharma",
                "batch_number": "OI-2025-456",
                "expiry_date": "2026-09-30",
                "reorder_level": 8
            },
            {
                "name": "Monthly Contact Lens",
                "category": "Contact Lens",
                "price": 800.00,
                "stock": 25,
                "description": "Premium monthly disposable contact lens",
                "manufacturer": "Bausch & Lomb",
                "batch_number": "CL-2025-001",
                "expiry_date": "2025-12-31",
                "reorder_level": 5
            },
            {
                "name": "Daily Contact Lens Pack",
                "category": "Contact Lens",
                "price": 500.00,
                "stock": 40,
                "description": "Daily disposable contact lens (30 pack)",
                "manufacturer": "Alcon",
                "batch_number": "CL-2025-789",
                "expiry_date": "2026-03-15",
                "reorder_level": 10
            },
            {
                "name": "IOL (Intraocular Lens)",
                "category": "Surgical",
                "price": 15000.00,
                "stock": 8,
                "description": "Premium monofocal intraocular lens",
                "manufacturer": "Alcon",
                "batch_number": "IOL-2025-001",
                "expiry_date": "2027-06-30",
                "reorder_level": 3
            },
            {
                "name": "Surgical Drapes Set",
                "category": "Surgical",
                "price": 450.00,
                "stock": 15,
                "description": "Sterile surgical drapes for eye surgery",
                "manufacturer": "3M",
                "batch_number": "SD-2025-567",
                "expiry_date": "2026-11-30",
                "reorder_level": 5
            }
        ]
        
        # Insert all sample medicines
        result = pharmacy_collection.insert_many(sample_medicines)
        
        print(f"✓ Initialized pharmacy collection with {len(result.inserted_ids)} sample medicines")
        
        return {
            "status": "success",
            "message": f"Sample pharmacy data initialized with {len(result.inserted_ids)} medicines",
            "count": len(result.inserted_ids)
        }
    
    except Exception as e:
        print(f"✗ Error initializing sample data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ SURGERY PACKAGES ENDPOINTS (Reusable Templates) ============

@app.post("/api/surgery-packages")
async def create_surgery_package(package_data: NewSurgeryPackage = Body(...)):
    """Create a new reusable surgery package"""
    try:
        # Calculate total amount
        total_amount = sum(item.amount for item in package_data.items)
        
        package_doc = {
            "packageName": package_data.packageName,
            "description": package_data.description or "",
            "items": [item.model_dump() for item in package_data.items],
            "totalAmount": total_amount,
            "createdBy": package_data.createdBy or "System",
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat(),
        }
        
        result = surgery_packages_collection.insert_one(package_doc)
        
        # Retrieve and return the created package
        created_package = surgery_packages_collection.find_one({"_id": result.inserted_id})
        
        return sanitize(created_package)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/surgery-packages")
async def get_all_surgery_packages():
    """Get all surgery packages"""
    try:
        packages = list(surgery_packages_collection.find())
        return [sanitize(pkg) for pkg in packages]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/surgery-packages/{package_id}")
async def get_surgery_package(package_id: str):
    """Get a specific surgery package by ID"""
    try:
        if not ObjectId.is_valid(package_id):
            raise HTTPException(status_code=400, detail="Invalid package ID")
        
        package = surgery_packages_collection.find_one({"_id": ObjectId(package_id)})
        
        if not package:
            raise HTTPException(status_code=404, detail="Package not found")
        
        return sanitize(package)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/surgery-packages/{package_id}")
async def update_surgery_package(package_id: str, update_data: UpdateSurgeryPackage = Body(...)):
    """Update an existing surgery package"""
    try:
        if not ObjectId.is_valid(package_id):
            raise HTTPException(status_code=400, detail="Invalid package ID")
        
        package = surgery_packages_collection.find_one({"_id": ObjectId(package_id)})
        if not package:
            raise HTTPException(status_code=404, detail="Package not found")
        
        # Prepare update data
        update_dict = {}
        
        if update_data.packageName is not None:
            update_dict["packageName"] = update_data.packageName
        
        if update_data.description is not None:
            update_dict["description"] = update_data.description
        
        if update_data.items is not None:
            update_dict["items"] = [item.model_dump() for item in update_data.items]
            # Recalculate total
            update_dict["totalAmount"] = sum(item.amount for item in update_data.items)
        
        update_dict["updatedAt"] = datetime.utcnow().isoformat()
        
        surgery_packages_collection.update_one(
            {"_id": ObjectId(package_id)},
            {"$set": update_dict}
        )
        
        # Return updated package
        updated_package = surgery_packages_collection.find_one({"_id": ObjectId(package_id)})
        return sanitize(updated_package)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/surgery-packages/{package_id}")
async def delete_surgery_package(package_id: str):
    """Delete a surgery package"""
    try:
        if not ObjectId.is_valid(package_id):
            raise HTTPException(status_code=400, detail="Invalid package ID")
        
        result = surgery_packages_collection.delete_one({"_id": ObjectId(package_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Package not found")
        
        return {"message": "Package deleted successfully", "deleted_id": package_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============ NEW SURGERY PACKAGE ENDPOINTS FOR PHASE 2 ============

@app.post("/api/save-surgery-package")
async def save_surgery_package(bill_data: dict = Body(...)):
    """Save current surgery bill items as a reusable package"""
    try:
        package_name = bill_data.get("packageName")
        items = bill_data.get("items", [])
        
        if not package_name or not items:
            raise HTTPException(status_code=400, detail="Package name and items required")
        
        # Calculate total amount
        total_amount = sum(item.get("amount", 0) for item in items)
        
        # Check for duplicate package (by name and total amount)
        existing = surgery_packages_collection.find_one({
            "packageName": package_name,
            "totalAmount": total_amount
        })
        
        if existing:
            raise HTTPException(status_code=409, detail="Package with this name and amount already exists")
        
        # Create package
        package_doc = {
            "packageName": package_name,
            "description": bill_data.get("description", ""),
            "items": items,
            "totalAmount": total_amount,
            "lastUsedDate": datetime.utcnow().isoformat(),
            "usageCount": 1,
            "createdBy": bill_data.get("createdBy", "System"),
            "createdAt": datetime.utcnow().isoformat(),
            "updatedAt": datetime.utcnow().isoformat(),
        }
        
        result = surgery_packages_collection.insert_one(package_doc)
        
        # Retrieve and return the created package
        created_package = surgery_packages_collection.find_one({"_id": result.inserted_id})
        
        return sanitize(created_package)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/surgery-packages/recent")
async def get_recent_surgery_packages(limit: int = 10):
    """Get recently used surgery packages (sorted by lastUsedDate)"""
    try:
        # Get packages sorted by lastUsedDate in descending order, limited to recent ones
        packages = list(
            surgery_packages_collection.find()
            .sort("lastUsedDate", -1)
            .limit(limit)
        )
        
        return [sanitize(pkg) for pkg in packages]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/surgery-packages/search")
async def search_surgery_packages(term: str = ""):
    """Search surgery packages by name"""
    try:
        if not term.strip():
            # Return all packages sorted alphabetically if no search term
            packages = list(
                surgery_packages_collection.find()
                .sort("packageName", 1)
            )
        else:
            # Search for packages matching the term (case-insensitive)
            packages = list(
                surgery_packages_collection.find({
                    "packageName": {
                        "$regex": term,
                        "$options": "i"
                    }
                }).sort("packageName", 1)
            )
        
        return [sanitize(pkg) for pkg in packages]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/billing/invoices/{registration_id}")
async def update_bill_with_dates(registration_id: str, bill_data: dict = Body(...)):
    """Update billing invoice with surgery and discharge dates"""
    try:
        # Verify patient exists
        patient = patient_collection.find_one({"registrationId": registration_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Update or create invoice with dates
        invoice_data = {
            "registrationId": registration_id,
            "dateOfSurgery": bill_data.get("dateOfSurgery"),
            "dateOfDischarge": bill_data.get("dateOfDischarge"),
            "items": bill_data.get("items", []),
            "totalAmount": bill_data.get("totalAmount", 0),
            "updatedAt": datetime.utcnow().isoformat(),
        }
        
        # Add other fields if provided
        if "claimNumber" in bill_data:
            invoice_data["claimNumber"] = bill_data["claimNumber"]
        if "insuranceAmount" in bill_data:
            invoice_data["insuranceAmount"] = bill_data["insuranceAmount"]
        if "patientAmount" in bill_data:
            invoice_data["patientAmount"] = bill_data["patientAmount"]
        
        # Upsert invoice
        result = billing_invoices_collection.update_one(
            {"registrationId": registration_id},
            {
                "$set": invoice_data,
                "$setOnInsert": {
                    "createdAt": datetime.utcnow().isoformat(),
                    "_id": ObjectId()
                }
            },
            upsert=True
        )
        
        # Return updated invoice
        updated_invoice = billing_invoices_collection.find_one(
            {"registrationId": registration_id}
        )
        
        return sanitize(updated_invoice)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/billing/patient/{registration_id}/surgery-bills/initial")
async def update_initial_bill_with_dates(registration_id: str, bill_data: dict = Body(...)):
    """Update initial surgery bill with dates (remove UI coverage display)"""
    try:
        patient = patient_collection.find_one({"registrationId": registration_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Extract dates and other relevant data
        date_of_surgery = bill_data.get("dateOfSurgery")
        date_of_discharge = bill_data.get("dateOfDischarge")
        
        # Process items - calculate coverage amount (shown only in alert, not in UI)
        items = bill_data.get("items", [])
        total_amount = sum(item.get("amount", 0) for item in items)
        
        # Estimated coverage (for backend calculation, not displayed in UI)
        estimated_coverage = bill_data.get("estimatedCoverage", 0)
        
        # Create initial bill document
        initial_bill = {
            "registrationId": registration_id,
            "billType": "initial",
            "dateOfSurgery": date_of_surgery,
            "dateOfDischarge": date_of_discharge,
            "items": items,
            "totalAmount": total_amount,
            "estimatedCoverage": estimated_coverage,  # Stored but not displayed in UI
            "claimNumber": bill_data.get("claimNumber", ""),
            "insuranceProvider": bill_data.get("insuranceProvider", ""),
            "createdAt": datetime.utcnow().isoformat(),
        }
        
        result = initial_surgery_bills_collection.insert_one(initial_bill)
        
        # Return created bill (without coverage amount for UI - it gets shown in alert only)
        created_bill = initial_surgery_bills_collection.find_one({"_id": result.inserted_id})
        bill_response = sanitize(created_bill)
        
        # Remove coverage amount from response for UI (will be shown in alert only)
        if "estimatedCoverage" in bill_response:
            bill_response["_uiHidden_estimatedCoverage"] = bill_response.pop("estimatedCoverage")
        
        return bill_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/billing/patient/{registration_id}/surgery-bills/final")
async def update_final_bill_with_dates(registration_id: str, bill_data: dict = Body(...)):
    """Update final surgery bill with dates (remove UI refund display)"""
    try:
        patient = patient_collection.find_one({"registrationId": registration_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Extract dates and other relevant data
        date_of_surgery = bill_data.get("dateOfSurgery")
        date_of_discharge = bill_data.get("dateOfDischarge")
        
        # Process items - calculate refund amount (shown only in alert, not in UI)
        items = bill_data.get("items", [])
        total_amount = sum(item.get("amount", 0) for item in items)
        
        # Actual refund (for backend calculation, not displayed in UI)
        refund_amount = bill_data.get("refundAmount", 0)
        
        # Create final bill document
        final_bill = {
            "registrationId": registration_id,
            "billType": "final",
            "dateOfSurgery": date_of_surgery,
            "dateOfDischarge": date_of_discharge,
            "items": items,
            "totalAmount": total_amount,
            "refundAmount": refund_amount,  # Stored but not displayed in UI
            "claimNumber": bill_data.get("claimNumber", ""),
            "insuranceProvider": bill_data.get("insuranceProvider", ""),
            "createdAt": datetime.utcnow().isoformat(),
        }
        
        result = final_surgery_bills_collection.insert_one(final_bill)
        
        # Return created bill (without refund amount for UI - it gets shown in alert only)
        created_bill = final_surgery_bills_collection.find_one({"_id": result.inserted_id})
        bill_response = sanitize(created_bill)
        
        # Remove refund amount from response for UI (will be shown in alert only)
        if "refundAmount" in bill_response:
            bill_response["_uiHidden_refundAmount"] = bill_response.pop("refundAmount")
        
        return bill_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SlitLampImage(BaseModel):
    patientId: str
    patientName: str
    doctorName: Optional[str] = "Unknown"
    image: str  # Base64 string
    notes: Optional[str] = ""
    timestamp: Optional[str] = None
    eyeSide: Optional[str] = "Both" # Left, Right, Both

# --- Slit Lamp Endpoints ---
@app.post("/slit-lamp/upload")
async def upload_slit_lamp_image(data: SlitLampImage):
    try:
        if not data.timestamp:
            data.timestamp = datetime.now().isoformat()
            
        doc = data.dict()
        # Synchronous insert (pymongo)
        result = slit_lamp_collection.insert_one(doc)
        
        return {
            "status": "success",
            "message": "Image saved successfully",
            "id": str(result.inserted_id)
        }
    except Exception as e:
        print(f"Slit lamp upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/slit-lamp/patient/{patient_id}")
async def get_patient_slit_lamp_images(patient_id: str):
    # Synchronous find
    cursor = slit_lamp_collection.find({"patientId": patient_id}).sort("timestamp", -1).limit(100)
    images = list(cursor)
    for img in images:
        img["_id"] = str(img["_id"])
    return {"images": images}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
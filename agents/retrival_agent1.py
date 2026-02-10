# retrieval_agent_1.py

import os
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

# Offline-first: always use a standard local MongoDB URI.
# This avoids mongodb+srv DNS/SRV lookups (Atlas) which fail without internet.
MONGO_URI = os.getenv("MONGO_URI_LOCAL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DATABASE_NAME", "hospital-emr")

# Short timeout so offline runs fail fast instead of hanging.
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client[DB_NAME]
patients = db["patients"]


def _coerce_object_id(value):
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    return None

def extract_summary_from_visit(visit):
    visit_summary = {
        "visitDate": visit.get("visitDate"),
        "iop": {},
        "vision": {},
        "refraction": {},
        "diagnosis": "",
        "prescription": [],
        "notes": ""
    }

    stages = visit.get("stages", {})

    # Optometry data
    opd = stages.get("opd", {}).get("data", {}).get("optometry", {})
    visit_summary["vision"] = opd.get("vision", {})
    visit_summary["iop"] = opd.get("iop", {})
    visit_summary["refraction"] = opd.get("autoRefraction", {})

    # Doctor data
    doc_data = stages.get("doctor", {}).get("data", {})
    visit_summary["diagnosis"] = doc_data.get("diagnosis", "")
    visit_summary["prescription"] = doc_data.get("prescription", {}).get("items", [])
    visit_summary["notes"] = doc_data.get("notes", "") or doc_data.get("followUp", "")

    return visit_summary

def get_patient_summary(patient_id, max_visits=3):
    oid = _coerce_object_id(patient_id)
    if oid is None:
        return {
            "error": "Invalid patient_id (expected a 24-character hex MongoDB ObjectId)",
            "patient_id": str(patient_id),
        }

    patient = patients.find_one({ "_id": oid })

    if not patient:
        return { "error": "Patient not found" }

    summary = {
        "name": patient.get("name"),
        "registrationId": patient.get("registrationId"),
        "sex": patient.get("demographics", {}).get("sex", ""),
        "age": patient.get("demographics", {}).get("age", ""),
        "visits": []
    }

    visits = patient.get("visits", [])[-max_visits:]
    summary["visits"] = [extract_summary_from_visit(v) for v in visits]

    return summary

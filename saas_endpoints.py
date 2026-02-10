"""
Multi-Tenant Database Provisioning for SaaS
Handles organization signup, payment verification, and database creation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pymongo
import certifi
from typing import List, Optional
import json
import os
from dotenv import load_dotenv
from mongodb_atlas_manager import MongoDBAtlasManager

load_dotenv()

router = APIRouter()

# Initialize Atlas Manager with credentials from .env
atlas_manager = MongoDBAtlasManager()

# Connect to master database (MongoDB Atlas)
master_uri = os.getenv("MASTER_DATABASE_URI")

if not master_uri:
    # Fallback to local if Atlas URI not configured
    print("[INIT] WARNING: MASTER_DATABASE_URI not set, using local MongoDB")
    master_client = pymongo.MongoClient("mongodb://localhost:27017")
else:
    print("[INIT] Connecting to MongoDB Atlas for master database...")
    try:
        # Use certifi for SSL certificates only for Atlas connections to avoid handshake errors on Windows
        client_kwargs = {"serverSelectionTimeoutMS": 5000}
        if "mongodb.net" in master_uri or "mongodb+srv://" in master_uri:
            client_kwargs["tlsCAFile"] = certifi.where()
            
        master_client = pymongo.MongoClient(master_uri, **client_kwargs)
        # Test connection
        master_client.admin.command('ping')
        print("[INIT] Successfully connected to MongoDB Atlas")
    except Exception as e:
        print(f"[INIT] ERROR connecting to Atlas: {e}")
        print("[INIT] Falling back to local MongoDB")
        master_client = pymongo.MongoClient("mongodb://localhost:27017")

master_db = master_client["chakravue_master"]
organizations = master_db["organizations"]
org_users = master_db["organization_users"]

# ============= DATA MODELS =============

class PlanDetails(BaseModel):
    plan_id: str
    name: str
    price: int
    max_users: int

class OrganizationSignup(BaseModel):
    organization_name: str
    organization_email: str
    organization_phone: str
    plan: PlanDetails

class PaymentDetails(BaseModel):
    organization_id: str
    card_number: str
    amount: float
    currency: str = "USD"

class UserSetup(BaseModel):
    organization_id: str
    email: str
    role: str  # receptionist, opd, doctor
    password: str = "default_password_123"

class OrganizationResponse(BaseModel):
    organization_id: str
    organization_name: str
    database_name: str
    plan: str
    status: str

# ============= ORGANIZATION SETUP ENDPOINTS =============

@router.post("/signup")
async def create_organization(signup_data: OrganizationSignup):
    """
    Step 1: Register new organization
    """
    try:
        org_id = f"org_{int(datetime.now().timestamp() * 1000)}"
        db_name = signup_data.organization_name.lower().replace(" ", "_")
        
        organization_doc = {
            "organization_id": org_id,
            "organization_name": signup_data.organization_name,
            "organization_email": signup_data.organization_email,
            "organization_phone": signup_data.organization_phone,
            "database_name": db_name,
            "plan": signup_data.plan.plan_id,
            "plan_name": signup_data.plan.name,
            "plan_price": signup_data.plan.price,
            "max_users": signup_data.plan.max_users,
            "status": "pending_payment",
            "created_at": datetime.now().isoformat(),
            "payment_date": None,
            "subscription_id": None
        }
        
        # Insert into master database
        result = organizations.insert_one(organization_doc)
        
        return {
            "status": "success",
            "organization_id": org_id,
            "message": f"Organization {signup_data.organization_name} registered. Proceed to payment."
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-payment")
async def process_payment(payment: PaymentDetails):
    """
    Step 2: Process payment and create MongoDB Atlas database
    """
    try:
        print(f"[PAYMENT] Starting payment for org: {payment.organization_id}")
        
        # Verify organization exists
        org = organizations.find_one({"organization_id": payment.organization_id})
        if not org:
            print(f"[PAYMENT] ERROR: Organization not found")
            raise HTTPException(status_code=404, detail="Organization not found")
        
        print(f"[PAYMENT] Organization found: {org['organization_name']}")
        
        # Dummy payment verification
        if not payment.card_number.startswith("4111"):
            print(f"[PAYMENT] ERROR: Invalid card number")
            raise HTTPException(status_code=400, detail="Payment failed")
        
        print(f"[PAYMENT] Payment verified, creating Atlas cluster...")
        
        # Create MongoDB Atlas cluster for this hospital
        cluster_name = f"hospital-{payment.organization_id[:8]}"
        db_name = org["database_name"]
        db_username = f"user_{payment.organization_id[:8]}"
        
        print(f"[PAYMENT] Cluster name: {cluster_name}")
        
        # Create cluster
        try:
            cluster_response = atlas_manager.create_cluster(
                org["organization_name"], 
                org["organization_email"],
                org["plan"]  # Pass the plan ID to determine cluster tier
            )
            print(f"[PAYMENT] Cluster response: {cluster_response}")
            
            if not cluster_response or "cluster_id" not in cluster_response:
                print(f"[PAYMENT] ERROR: Invalid cluster response")
                raise HTTPException(status_code=500, detail="Failed to create MongoDB Atlas cluster")
            
            cluster_id = cluster_response["cluster_id"]
            connection_string = cluster_response.get("connection_string")
            cluster_tier = cluster_response.get("cluster_tier", "M0")
            
            print(f"✓ Cluster created: {cluster_id}")
            
        except Exception as e:
            print(f"[PAYMENT] ERROR creating cluster: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Atlas error: {str(e)}")
        
        print(f"[PAYMENT] Updating organization in master database...")
        
        # Update organization in master database with Atlas connection details
        organizations.update_one(
            {"organization_id": payment.organization_id},
            {
                "$set": {
                    "status": "active",
                    "payment_date": datetime.now().isoformat(),
                    "subscription_id": f"sub_{int(datetime.now().timestamp())}",
                    "database_created": True,
                    "mongodb_cluster_id": cluster_id,
                    "mongodb_cluster_name": cluster_name,
                    "mongodb_cluster_tier": cluster_tier,  # NEW: Store cluster tier
                    "mongodb_connection_string": connection_string,
                    "mongodb_database_name": db_name,
                    "mongodb_username": db_username
                }
            }
        )
        
        print(f"✓ Organization updated in master database")
        
        return {
            "status": "success",
            "message": "Payment processed and MongoDB Atlas database created",
            "organization_id": payment.organization_id,
            "database_name": db_name,
            "cluster_id": cluster_id,
            "cluster_tier": cluster_tier,  # NEW: Return cluster tier
            "subscription_id": f"sub_{int(datetime.now().timestamp())}",
            "mongodb_connection_string": connection_string
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PAYMENT] FATAL ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-user")
async def add_user_to_organization(user: UserSetup):
    """
    Step 3: Add users (receptionist, opd, doctor) to organization database
    """
    try:
        print(f"[ADD-USER] Starting for org: {user.organization_id}")
        
        # Get organization details
        org = organizations.find_one({"organization_id": user.organization_id})
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        print(f"[ADD-USER] Organization found: {org['organization_name']}")
        
        if org["status"] != "active":
            raise HTTPException(status_code=400, detail="Organization is not active")
        
        # Use MongoDB Atlas connection string instead of local
        connection_string = org.get("mongodb_connection_string")
        print(f"[ADD-USER] Connection string exists: {bool(connection_string)}")
        
        # Determine if this is a real Atlas connection or a mock/local one
        # Real Atlas: mongodb+srv://user:pass@clusterX.xxxx.mongodb.net/database
        # Mock: mongodb+srv://user:pass@hospitalname.mongodb.net/database (fake)
        
        is_real_atlas = (
            connection_string and 
            "mongodb+srv://" in connection_string and 
            ".mongodb.net" in connection_string and
            "localhost" not in connection_string
        )
        
        # Check if it's actually a mock by looking for Atlas project structure
        is_mock = is_real_atlas and not any(x in connection_string for x in ["cluster", "a1b2c", "l6uewin", "2f1"])
        
        if not connection_string or is_mock:
            # Use local MongoDB for mock/testing or if no connection string
            print("[ADD-USER] Using local MongoDB (mock response or testing)")
            org_client = pymongo.MongoClient("mongodb://localhost:27017")
            org_db = org_client[org["database_name"]]
        else:
            print(f"[ADD-USER] Connecting to real MongoDB Atlas...")
            try:
                # Use certifi for SSL certificates only for Atlas connections to avoid handshake errors on Windows
                client_kwargs = {"serverSelectionTimeoutMS": 5000}
                if "mongodb.net" in connection_string or "mongodb+srv://" in connection_string:
                    client_kwargs["tlsCAFile"] = certifi.where()
                
                org_client = pymongo.MongoClient(connection_string, **client_kwargs)
                org_db = org_client[org["mongodb_database_name"]]
                print(f"[ADD-USER] Connected to Atlas database: {org['mongodb_database_name']}")
            except Exception as e:
                print(f"[ADD-USER] Atlas connection error: {e}, falling back to local MongoDB")
                org_client = pymongo.MongoClient("mongodb://localhost:27017")
                org_db = org_client[org["database_name"]]
        
        # Check if user already exists
        existing_user = org_db.users.find_one({"email": user.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        print(f"[ADD-USER] Creating user: {user.email}")
        
        # Create user document
        user_doc = {
            "user_id": f"user_{int(datetime.now().timestamp() * 1000)}",
            "email": user.email,
            "role": user.role.upper(),
            "password": user.password,
            "organization_id": user.organization_id,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
        
        # Insert user into organization database
        result = org_db.users.insert_one(user_doc)
        
        print(f"[ADD-USER] User created successfully: {user_doc['user_id']}")
        
        return {
            "status": "success",
            "message": f"User {user.email} added with role {user.role}",
            "user_id": user_doc["user_id"],
            "email": user.email,
            "role": user.role
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ADD-USER] ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organization/{organization_id}")
async def get_organization_details(organization_id: str):
    """
    Get organization details and database info
    """
    try:
        org = organizations.find_one({"organization_id": organization_id})
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Remove MongoDB internal ID
        org.pop("_id", None)
        
        return {
            "status": "success",
            "data": org
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organization/{organization_id}/users")
async def list_organization_users(organization_id: str):
    """
    List all users in an organization
    """
    try:
        org = organizations.find_one({"organization_id": organization_id})
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        
        # Connect to organization database
        connection_string = org.get("mongodb_connection_string")
        if connection_string and ("mongodb+srv://" in connection_string or "mongodb.net" in connection_string):
            org_client = pymongo.MongoClient(connection_string, tlsCAFile=certifi.where())
            org_db = org_client[org.get("mongodb_database_name", org["database_name"])]
        else:
            org_client = pymongo.MongoClient("mongodb://localhost:27017")
            org_db = org_client[org["database_name"]]
        
        users = list(org_db.users.find({}, {"password": 0, "_id": 0}))
        
        return {
            "status": "success",
            "organization_id": organization_id,
            "user_count": len(users),
            "users": users
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/organization-login")
async def organization_login(email: str, password: str, organization_id: str):
    """
    Login endpoint for organization users
    Returns JWT token with organization context
    """
    try:
        org = organizations.find_one({"organization_id": organization_id})
        if not org or org["status"] != "active":
            raise HTTPException(status_code=401, detail="Invalid organization")
        
        # Connect to organization database
        connection_string = org.get("mongodb_connection_string")
        if connection_string and ("mongodb+srv://" in connection_string or "mongodb.net" in connection_string):
            org_client = pymongo.MongoClient(connection_string, tlsCAFile=certifi.where())
            org_db = org_client[org.get("mongodb_database_name", org["database_name"])]
        else:
            org_client = pymongo.MongoClient("mongodb://localhost:27017")
            org_db = org_client[org["database_name"]]
        
        # Find user
        user = org_db.users.find_one({"email": email})
        if not user or user["password"] != password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        if user["status"] != "active":
            raise HTTPException(status_code=401, detail="User is inactive")
        
        # Update last login
        org_db.users.update_one(
            {"user_id": user["user_id"]},
            {"$set": {"last_login": datetime.now().isoformat()}}
        )
        
        return {
            "status": "success",
            "message": "Login successful",
            "token": f"token_{user['user_id']}_{organization_id}",
            "user": {
                "user_id": user["user_id"],
                "email": user["email"],
                "role": user["role"],
                "organization_id": organization_id,
                "organization_name": org["organization_name"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/master/all-organizations")
async def get_all_organizations():
    """
    Get all organizations (admin endpoint)
    """
    try:
        orgs = list(organizations.find({}, {"_id": 0}))
        return {
            "status": "success",
            "total_organizations": len(orgs),
            "organizations": orgs
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check for multi-tenant system
    """
    return {
        "status": "healthy",
        "service": "multi-tenant-saas",
        "timestamp": datetime.now().isoformat()
    }

# ============= ADMIN ENDPOINTS - VIEW ALL HOSPITAL DATABASES =============

@router.get("/admin/hospitals")
async def admin_get_all_hospitals():
    """Admin: Get list of all hospitals with stats"""
    try:
        hospitals_list = []
        for org in organizations.find({}):
            hospitals_list.append({
                "organization_id": org.get("organization_id"),
                "hospital_name": org.get("organization_name"),
                "email": org.get("organization_email"),
                "plan": org.get("plan"),
                "status": org.get("status"),
                "created_date": org.get("created_at"),
            })
        return {
            "status": "success",
            "total_hospitals": len(hospitals_list),
            "hospitals": hospitals_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/hospital/{hospital_id}/patients")
async def admin_get_hospital_patients(hospital_id: str):
    """Admin: Get all patients of a specific hospital"""
    try:
        org = organizations.find_one({"organization_id": hospital_id})
        if not org:
            raise HTTPException(status_code=404, detail="Hospital not found")
        
        hospital_connection = org.get("mongodb_connection_string")
        hospital_db_name = org.get("mongodb_database_name")
        if not hospital_connection:
            raise HTTPException(status_code=400, detail="Hospital database not configured")
        
        # Use certifi for SSL certificates only for Atlas connections to avoid handshake errors on Windows
        client_kwargs = {}
        if "mongodb.net" in hospital_connection or "mongodb+srv://" in hospital_connection:
            client_kwargs["tlsCAFile"] = certifi.where()
            
        hospital_client = pymongo.MongoClient(hospital_connection, **client_kwargs)
        hospital_db = hospital_client[hospital_db_name]
        patients = list(hospital_db["patients"].find({}, {"_id": 0}))
        
        return {
            "status": "success",
            "hospital_name": org.get("organization_name"),
            "patients": patients,
            "total_count": len(patients)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/hospital/{hospital_id}/stats")
async def admin_get_hospital_stats(hospital_id: str):
    """Admin: Get database statistics for a hospital"""
    try:
        org = organizations.find_one({"organization_id": hospital_id})
        if not org:
            raise HTTPException(status_code=404, detail="Hospital not found")
        
        hospital_connection = org.get("mongodb_connection_string")
        hospital_db_name = org.get("mongodb_database_name")
        if not hospital_connection:
            raise HTTPException(status_code=400, detail="Hospital database not configured")
        
        # Use certifi for SSL certificates only for Atlas connections to avoid handshake errors on Windows
        client_kwargs = {}
        if "mongodb.net" in hospital_connection or "mongodb+srv://" in hospital_connection:
            client_kwargs["tlsCAFile"] = certifi.where()
            
        hospital_client = pymongo.MongoClient(hospital_connection, **client_kwargs)
        hospital_db = hospital_client[hospital_db_name]
        
        patients_count = hospital_db["patients"].count_documents({})
        appointments_count = hospital_db["appointments"].count_documents({})
        billing_count = hospital_db["billing"].count_documents({})
        
        return {
            "status": "success",
            "hospital_name": org.get("organization_name"),
            "stats": {
                "patients": patients_count,
                "appointments": appointments_count,
                "billing": billing_count,
                "total_records": patients_count + appointments_count + billing_count
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/admin/hospital/{hospital_id}/patient/{patient_id}")
async def admin_edit_patient(hospital_id: str, patient_id: str, patient_data: dict):
    """Admin: Edit patient data"""
    try:
        org = organizations.find_one({"organization_id": hospital_id})
        if not org:
            raise HTTPException(status_code=404, detail="Hospital not found")
        
        hospital_connection = org.get("mongodb_connection_string")
        hospital_db_name = org.get("mongodb_database_name")
        if not hospital_connection:
            raise HTTPException(status_code=400, detail="Hospital database not configured")
        
        # Use certifi for SSL certificates only for Atlas connections to avoid handshake errors on Windows
        client_kwargs = {}
        if "mongodb.net" in hospital_connection or "mongodb+srv://" in hospital_connection:
            client_kwargs["tlsCAFile"] = certifi.where()
            
        hospital_client = pymongo.MongoClient(hospital_connection, **client_kwargs)
        hospital_db = hospital_client[hospital_db_name]
        result = hospital_db["patients"].update_one(
            {"patient_id": patient_id},
            {"$set": patient_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        return {
            "status": "success",
            "message": f"Patient {patient_id} updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/admin/hospital/{hospital_id}/patient/{patient_id}")
async def admin_delete_patient(hospital_id: str, patient_id: str):
    """Admin: Delete patient data"""
    try:
        org = organizations.find_one({"organization_id": hospital_id})
        if not org:
            raise HTTPException(status_code=404, detail="Hospital not found")
        
        hospital_connection = org.get("mongodb_connection_string")
        hospital_db_name = org.get("mongodb_database_name")
        if not hospital_connection:
            raise HTTPException(status_code=400, detail="Hospital database not configured")
        
        # Use certifi for SSL certificates only for Atlas connections to avoid handshake errors on Windows
        client_kwargs = {}
        if "mongodb.net" in hospital_connection or "mongodb+srv://" in hospital_connection:
            client_kwargs["tlsCAFile"] = certifi.where()
            
        hospital_client = pymongo.MongoClient(hospital_connection, **client_kwargs)
        hospital_db = hospital_client[hospital_db_name]
        result = hospital_db["patients"].delete_one({"patient_id": patient_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        return {"status": "success", "message": f"Patient {patient_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

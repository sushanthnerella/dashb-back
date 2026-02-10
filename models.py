from pydantic import BaseModel, Field
from typing import List, Optional, Union, Any
from datetime import datetime
from bson import ObjectId
from pydantic_core import core_schema

# Helper for MongoDB's _id field
class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        # Validator for when the input is already an ObjectId
        def validate_from_object_id(v: ObjectId) -> ObjectId:
            return v

        # Validator for when the input is a string
        def validate_from_str(v: str) -> ObjectId:
            if not ObjectId.is_valid(v):
                raise ValueError('Invalid ObjectId')
            return ObjectId(v)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_plain_validator_function(validate_from_str),
            python_schema=core_schema.union_schema(
                [core_schema.is_instance_schema(ObjectId), core_schema.no_info_plain_validator_function(validate_from_str)]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )

# --- Sub-models ---

# --- Sub-models for Receptionist Data ---
class Demographics(BaseModel):
    age: int
    sex: str
    bloodType: str

class ContactInfo(BaseModel):
    phone: str = Field(...)
    email: Optional[str] = None
    address: str = Field(...)

class PatientDetailModel(BaseModel):
    name: str
    password: str
    age: str
    sex: str
    phone: str
    email: Optional[str] = None
    address: str
    bloodType: str
    allergies: str
    emergencyContact: str

class PresentingComplaintHistory(BaseModel):
    severity: Optional[str] = ""
    onset: Optional[str] = ""
    aggravating: Optional[str] = ""
    relieving: Optional[str] = ""
    associated: Optional[str] = ""

class EmergencyContact(BaseModel):
    name: str
    phone: str

class MedicalHistoryItem(BaseModel):
    id: str
    condition: str
    diagnosedYear: str = Field(..., validation_alias="year", serialization_alias="year")
    status: str

class SurgicalHistoryItem(BaseModel):
    id: str
    procedure: str
    procedureYear: str = Field(..., validation_alias="year", serialization_alias="year")
    type: str

class History(BaseModel):
    severity: Optional[str] = ""
    onset: Optional[str] = ""
    aggravating: Optional[str] = ""
    relieving: Optional[str] = ""
    associated: Optional[str] = ""
    medical: List[MedicalHistoryItem] = []
    surgical: List[SurgicalHistoryItem] = []
    family: Optional[str] = ""

class InitialComplaint(BaseModel):
    id: str
    complaint: Optional[str] = ""
    duration: Optional[str] = ""

class PresentingComplaints(BaseModel):
    complaints: List[InitialComplaint] = []
    history: PresentingComplaintHistory = Field(default_factory=PresentingComplaintHistory)

class MedicalHistory(BaseModel):
    medical: List[MedicalHistoryItem] = []
    surgical: List[SurgicalHistoryItem] = []
    familyHistory: Optional[str] = ""

# --- Role-Specific Models ---

# Model for creating a NEW patient (from Receptionist)
class NewPatient(BaseModel):
    registrationId: Optional[str] = None  # If provided, use this; otherwise backend will auto-generate
    patientDetails: PatientDetailModel
    presentingComplaints: PresentingComplaints
    medicalHistory: MedicalHistory
    # Accept drugHistory as a flexible shape from the receptionist form.
    # Typing as Optional[dict] keeps the model permissive while allowing
    # the backend to persist medication-related receptionist input.
    drugHistory: Optional[dict] = None

# --- Encounter Models ---

# Base model for all encounters
class BaseEncounter(BaseModel):
    date: datetime = Field(default_factory=datetime.utcnow)
    doctor: str # Can be "Reception", "Nurse X", "Dr. Y"

# Receptionist's initial encounter data
class ReceptionistEncounter(BaseEncounter):
    presentingComplaints: List[InitialComplaint] = []

# --- Full Patient Model in Database ---

class PatientInDB(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    registrationId: str
    name: str
    hashed_password: str
    demographics: Demographics
    contactInfo: ContactInfo
    emergencyContact: EmergencyContact
    allergies: List[str] = []
    documents_id: List[str] = []
    documents: List[dict] = []
    # Persisted drug/medication history if present
    drugHistory: Optional[dict] = None
    # Doctor-entered free-form data (store the four doctor cards as a permissive dict)
    # This allows the frontend to send whatever nested structure it uses for the
    # Ophthalmologist / Special / Medication / Investigations cards and have it
    # persisted without strict Pydantic schema errors.
    doctor: Optional[dict] = None
    history: History
    encounters: List[Union[ReceptionistEncounter, dict]] = [] # Using dict for future nurse/doc models

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# --- Models for Updating Encounters (to be expanded) ---

# Example for Nurse data
class NurseEncounterModel(BaseModel):
    optometry: dict # Replace with a detailed OptometryData model
    iop: dict       # Replace with a detailed IOPData model

# Example for Doctor data
class DoctorEncounterModel(BaseModel):
    # Keep the model permissive to accept the frontend's dictionary payload
    # for the doctor's save action.
    examination: Optional[dict] = None
    ophthalmologistExam: Optional[dict] = None
    prescription: Optional[dict] = None
    specialExamination: Optional[dict] = None
    investigationsSurgeries: Optional[dict] = None
    doctorName: Optional[str] = None
    diagnosis: Optional[str] = None
    followUp: Optional[str] = None
    lastStage: Optional[str] = None
    lastUpdated: Optional[str] = None


# --- Investigations Models ---
class InvestigationsModel(BaseModel):
    """Flexible model to accept investigation data from the frontend cards.
    Each field is intentionally typed as dict to allow the frontend to send
    the exact structure used in the UI. These will be stored under the
    `investigations` key in the patient document.
    """
    optometry: Optional[dict] = None
    ophthalmicInvestigations: Optional[dict] = None
    iop: Optional[dict] = None
    systemic: Optional[dict] = None


# --- User / Auth Models ---
from datetime import datetime

# Allowed roles for the application. These map to the front-end roles.
# Normalization in the API uppercases the incoming role string before validation,
# so we list canonical UPPERCASE role names here.
ALLOWED_ROLES = {"RECEPTIONIST", "OPD", "DOCTOR", "PATIENT"}

class NewUser(BaseModel):
    username: str
    full_name: Optional[str] = None
    password: str
    role: str


class UserInDB(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    username: str
    full_name: Optional[str] = None
    hashed_password: str
    role: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# --- Pharmacy Models ---
class PharmacyMedicine(BaseModel):
    id: Optional[str] = None
    name: str
    category: str  # Eye Drops, Tablets, Ointments, Contact Lens, Surgical
    price: float
    stock: int
    description: str
    manufacturer: Optional[str] = None
    batch_number: Optional[str] = None
    expiry_date: Optional[str] = None
    reorder_level: int = 10
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PharmacyMedicineInDB(PharmacyMedicine):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
# --- Insurance & Multi-stage Billing Models ---
class BillingStage(BaseModel):
    name: str # 'insurance_approval', 'pre_surgery', 'final_settlement'
    status: str # 'pending', 'approved', 'paid', 'cancelled'
    amount: float = 0.0
    notes: Optional[str] = None
    date: datetime = Field(default_factory=datetime.utcnow)
    updatedBy: Optional[str] = None

class BillingCase(BaseModel):
    caseId: str # f"CASE-{uuid.uuid4()[:8]}"
    registrationId: str
    procedureName: str
    totalEstimatedAmount: float = 0.0
    insuranceApprovedAmount: float = 0.0
    preSurgeryPaidAmount: float = 0.0
    stages: List[BillingStage] = []
    insuranceProvider: Optional[str] = None
    policyNumber: Optional[str] = None
    status: str = "open" # open, completed, closed
    dateOfSurgery: Optional[str] = None  # DD/MM/YYYY format
    dateOfDischarge: Optional[str] = None  # DD/MM/YYYY format
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)

class BillingCaseInDB(BillingCase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class CartItem(BaseModel):
    medicineId: str
    name: str
    quantity: int
    price: float
    total: float


class PharmacyBilling(BaseModel):
    registrationId: str
    patientName: str
    items: List[CartItem]
    totalAmount: float
    billDate: datetime = Field(default_factory=datetime.utcnow)
    status: str = "completed"  # completed, pending, cancelled
    paymentMethod: Optional[str] = None
    notes: Optional[str] = None


class PharmacyBillingInDB(PharmacyBilling):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# --- Surgery Packages Models (Reusable Templates) ---
class SurgeryParticularsBreakdown(BaseModel):
    """Detailed breakdown of a surgery particular"""
    sNo: int
    particular: str
    cost: float
    qty: int
    netAmt: float
    grossAmt: float


class SurgeryPackageItem(BaseModel):
    """Individual item/charge in a surgery package"""
    description: str
    amount: float
    breakdown: Optional[SurgeryParticularsBreakdown] = None  # Store full breakdown if available


class SurgeryPackage(BaseModel):
    """Reusable surgery package template"""
    packageName: str
    description: Optional[str] = ""
    items: List[SurgeryPackageItem]
    totalAmount: float
    createdBy: Optional[str] = None
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)


class SurgeryPackageInDB(SurgeryPackage):
    """Surgery package stored in database with MongoDB ID"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    hospitalId: Optional[str] = None  # Hospital-specific packages
    lastUsedDate: Optional[datetime] = None  # Track last usage
    usageCount: int = 0  # Track how many times used

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class NewSurgeryPackage(BaseModel):
    """Request model for creating new surgery package"""
    packageName: str
    description: Optional[str] = ""
    items: List[SurgeryPackageItem]
    createdBy: Optional[str] = None


class UpdateSurgeryPackage(BaseModel):
    """Request model for updating surgery package"""
    packageName: Optional[str] = None
    description: Optional[str] = None
    items: Optional[List[SurgeryPackageItem]] = None
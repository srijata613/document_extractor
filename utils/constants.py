"""
Application-wide constants used across the document extraction pipeline.
"""

from __future__ import annotations

# Supported File Types
SUPPORTED_PDF_EXTENSIONS = {".pdf"}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
}

# OCR Labels

PRINTED_TEXT = "printed"
HANDWRITTEN_TEXT = "handwritten"

# Checkbox Values

CHECKED = "Checked"
UNCHECKED = "Unchecked"

YES = "Yes"
NO = "No"

# Common Document Fields

FIELD_APPLICANT_NAME = "Applicant Name"
FIELD_FATHER_NAME = "Father Name"
FIELD_MOTHER_NAME = "Mother Name"
FIELD_SPOUSE_NAME = "Spouse Name"

FIELD_GENDER = "Gender"

FIELD_DATE_OF_BIRTH = "Date of Birth"

FIELD_AGE = "Age"

FIELD_MOBILE = "Mobile Number"

FIELD_EMAIL = "Email"

FIELD_AADHAAR = "Aadhaar Number"

FIELD_VOTER_ID = "Voter ID"

FIELD_PAN = "PAN"

FIELD_RATION_CARD = "Ration Card"

FIELD_ADDRESS = "Address"

FIELD_STATE = "State"

FIELD_DISTRICT = "District"

FIELD_BLOCK = "Block"

FIELD_VILLAGE = "Village"

FIELD_PINCODE = "Pincode"

FIELD_OCCUPATION = "Occupation"

FIELD_INCOME = "Income"

FIELD_CASTE = "Caste"

FIELD_CATEGORY = "Category"

FIELD_REMARKS = "Remarks"

# Gender

GENDER_MALE = "Male"

GENDER_FEMALE = "Female"

GENDER_OTHER = "Other"

# OCR Confidence

LOW_CONFIDENCE = 0.40

MEDIUM_CONFIDENCE = 0.70

HIGH_CONFIDENCE = 0.90


# Excel

DEFAULT_SHEET_NAME = "Extracted Data"

# Common Keywords

LABEL_KEYWORDS = [
    "Name",
    "Father",
    "Mother",
    "Spouse",
    "Address",
    "Village",
    "District",
    "Block",
    "State",
    "Age",
    "Gender",
    "DOB",
    "Occupation",
    "Income",
    "Mobile",
    "Email",
    "Aadhaar",
    "Voter",
    "PAN",
    "Ration",
    "Pincode",
]

# Image Processing
MIN_CHECKBOX_SIZE = 8

MAX_CHECKBOX_SIZE = 80

MIN_TEXT_AREA = 10

MAX_TEXT_AREA = 100000
from __future__ import annotations

from encodings import aliases
import re
from collections import defaultdict

from rapidfuzz import fuzz

from models.document import (
    Document,
    ExtractedField,
)
from utils.logger import app_logger

from models.document import (
    Document,
    ExtractedField,
    MappedField,
)


class FieldMappingError(Exception):
    """Raised when field mapping fails."""


class FieldMapper:
    """
    Maps semantic extracted fields into a canonical schema.
    """

    MATCH_THRESHOLD = 88

    def __init__(self) -> None:
        self.schema = self._build_schema()

    def _build_schema(self) -> dict[str, list[str]]:
        return {
            "Name": ["name", "full name", "candidate name", "patient name"],
            "Date of Birth": ["dob", "date of birth", "birth date"],
            "Age": ["age"],
            "Gender": ["gender", "sex"],
            "Email": ["email", "email address"],
            "Mobile Number": [
                "mobile",
                "mobile number",
                "phone",
                "phone number",
            ],
            "Address": [
                "address",
                "home address",
                "residential address",
            ],
            "Aadhaar Number": [
                "aadhaar",
                "aadhaar number",
            ],
            "PAN Number": [
                "pan",
                "pan number",
            ],
            "IFSC Code": [
                "ifsc",
                "ifsc code",
            ],
            "Bank Account Number": [
                "account number",
                "bank account",
            ],
            "Nationality": [
                "nationality",
            ],
            "Category": [
                "category",
                "caste",
            ],
            "Occupation": [
                "occupation",
                "profession",
                "job",
            ],
            "Company": [
                "company",
                "organization",
            ],
            "Signature": [
                "signature",
                "signed by",
            ],
        }

    def map_document(
        self,
        document: Document,
    ) -> dict[str, str]:

        app_logger.info(
            "Mapping extracted fields..."
        )

        mapped = self._build_mapped_fields(
            document
        )

        app_logger.success(
            "Field mapping completed."
        )

        return mapped

    def _find_best_match(
        self,
        field_name: str,
    ) -> str | None:

        field_name = field_name.lower()

        best = None
        best_score = 0

        for canonical, aliases in self.schema.items():
            for alias in aliases:
                score = fuzz.ratio(
                    field_name,
                    alias.lower(),
                )

                if score > best_score:
                    best_score = score
                    best = canonical

            score = fuzz.ratio(
                field_name,
                canonical.lower(),
            )

            if score > best_score:
                best_score = score
                best = canonical

        if best_score >= self.MATCH_THRESHOLD:
            return best

        return None
    
    def _normalize_value(
        self,
        field_name: str,
        value,
    ) -> str:
        """
        Normalize extracted values according to field type.
        """

        if value is None:
            return ""

        value = str(value).strip()

        if not value:
            return ""

        normalizers = {

            "Aadhaar Number": self._normalize_aadhaar,

            "PAN Number": self._normalize_pan,

            "Mobile Number": self._normalize_phone,

            "Phone Number": self._normalize_phone,

            "IFSC Code": self._normalize_ifsc,

            "Gender": self._normalize_gender,

            "Category": self._normalize_category,

            "Date of Birth": self._normalize_date,

            "DOB": self._normalize_date,

            "Email": self._normalize_email,

        }

        fn = normalizers.get(field_name)

        if fn:

            return fn(value)

        return self._clean_text(value)

    @staticmethod
    def _clean_text(
        value: str,
    ) -> str:

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    @staticmethod
    def _normalize_gender(
        value: str,
    ) -> str:

        value = value.lower()

        mapping = {

            "m": "Male",

            "male": "Male",

            "f": "Female",

            "female": "Female",

            "other": "Other",

            "transgender": "Other",

        }

        return mapping.get(
            value,
            value.title(),
        )

    @staticmethod
    def _normalize_category(
        value: str,
    ) -> str:

        mapping = {

            "sc": "SC",

            "st": "ST",

            "obc": "OBC",

            "ews": "EWS",

            "gen": "General",

            "general": "General",

        }

        return mapping.get(

            value.lower(),

            value.upper(),
        )

    @staticmethod
    def _normalize_phone(
        value: str,
    ) -> str:

        digits = re.sub(
            r"\D",
            "",
            value,
        )

        if len(digits) > 10:

            digits = digits[-10:]

        return digits

    @staticmethod
    def _normalize_aadhaar(
        value: str,
    ) -> str:

        digits = re.sub(
            r"\D",
            "",
            value,
        )

        if len(digits) != 12:
            return digits

        return (
            f"{digits[:4]} "
            f"{digits[4:8]} "
            f"{digits[8:]}"
        )

    @staticmethod
    def _normalize_pan(
        value: str,
    ) -> str:

        return re.sub(
            r"[^A-Za-z0-9]",
            "",
            value,
        ).upper()

    @staticmethod
    def _normalize_ifsc(
        value: str,
    ) -> str:

        return re.sub(
            r"\s+",
            "",
            value,
        ).upper()

    @staticmethod
    def _normalize_email(
        value: str,
    ) -> str:

        return value.lower().strip()

    @staticmethod
    def _normalize_date(
        value: str,
    ) -> str:

        value = value.replace(
            "-",
            "/",
        )

        value = value.replace(
            ".",
            "/",
        )

        parts = value.split("/")

        if len(parts) != 3:
            return value

        d, m, y = parts

        return (
            f"{d.zfill(2)}/"
            f"{m.zfill(2)}/"
            f"{y}"
        )
        
    def _build_mapped_fields(
        self,
        document: Document,
    ) -> dict[str, MappedField]:
        """
        Build final canonical fields with validation and conflict resolution.
        """

        mapped: dict[str, MappedField] = {}

        for page in document.pages:

            for field in page.extracted_fields:

                canonical = self._find_best_match(
                    field.name
                )

                if canonical is None:
                    continue

                normalized = self._normalize_value(
                    canonical,
                    field.value,
                )

                candidate = MappedField(
                    canonical_name=canonical,
                    value=normalized,
                    confidence=field.confidence,
                    source=field.source,
                    page_number=field.page_number,
                    original_value=str(field.value),
                )

                self._validate_field(
                    candidate
                )

                existing = mapped.get(
                    canonical
                )

                if existing is None:

                    mapped[canonical] = candidate
                    continue

                if self._is_better(
                    candidate,
                    existing,
                ):

                    mapped[canonical] = candidate

        return mapped

    @staticmethod
    def _is_better(
        candidate: MappedField,
        current: MappedField,
    ) -> bool:

        priority = {
            "handwritten": 4,
            "checkbox": 3,
            "printed": 2,
            "table": 1,
            "unknown": 0,
        }

        if (
            priority.get(
                candidate.source,
                0,
            )
            >
            priority.get(
                current.source,
                0,
            )
        ):
            return True

        return (
            candidate.confidence
            >
            current.confidence
        )

    def _validate_field(
        self,
        field: MappedField,
    ) -> None:

        validators = {

            "Aadhaar Number": self._validate_aadhaar,

            "PAN Number": self._validate_pan,

            "IFSC Code": self._validate_ifsc,

            "Mobile Number": self._validate_mobile,

            "Phone Number": self._validate_mobile,

            "Email": self._validate_email,

            "Date of Birth": self._validate_date,

        }

        validator = validators.get(
            field.canonical_name
        )

        if validator is None:

            field.validated = True
            return

        field.validated = validator(
            field
        )

    @staticmethod
    def _validate_aadhaar(
        field: MappedField,
    ) -> bool:

        digits = re.sub(
            r"\D",
            "",
            field.value,
        )

        if len(digits) != 12:

            field.add_error(
                "Invalid Aadhaar number."
            )

            return False

        return True

    @staticmethod
    def _validate_pan(
        field: MappedField,
    ) -> bool:

        if not re.fullmatch(
            r"[A-Z]{5}[0-9]{4}[A-Z]",
            field.value,
        ):

            field.add_error(
                "Invalid PAN format."
            )

            return False

        return True

    @staticmethod
    def _validate_ifsc(
        field: MappedField,
    ) -> bool:

        if not re.fullmatch(
            r"[A-Z]{4}0[A-Z0-9]{6}",
            field.value,
        ):

            field.add_error(
                "Invalid IFSC code."
            )

            return False

        return True

    @staticmethod
    def _validate_mobile(
        field: MappedField,
    ) -> bool:

        if not re.fullmatch(
            r"[6-9][0-9]{9}",
            field.value,
        ):

            field.add_error(
                "Invalid mobile number."
            )

            return False

        return True

    @staticmethod
    def _validate_email(
        field: MappedField,
    ) -> bool:

        if not re.fullmatch(
            r"[^@]+@[^@]+\.[^@]+",
            field.value,
        ):

            field.add_error(
                "Invalid email address."
            )

            return False

        return True

    @staticmethod
    def _validate_date(
        field: MappedField,
    ) -> bool:

        if not re.fullmatch(
            r"\d{2}/\d{2}/\d{4}",
            field.value,
        ):

            field.add_error(
                "Invalid date."
            )

            return False

        return True
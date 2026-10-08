from __future__ import annotations

import re

# Patterns for requests that must never be processed by the assistant.
OTP_PATTERNS = [
    r"\bOTP\b",
    r"\b(one[- ]time password|one time password)\b",
    r"\bpassword\b",
    r"\bPIN\b",
    r"\bCVV\b",
    r"\bsecurity code\b",
    r"\bsecret code\b",
    r"\bauthentication credential\b",
]

# Patterns for requests to guarantee a lending decision.
APPROVAL_PATTERNS = [
    r"guarantee\s+(the\s+)?loan\s+.*approved",
    r"guarantee\s+.*approval",
    r"guarantee\s+.*sanction",
    r"guarantee\s+(that\s+)?[^\n]*\bapproved\b",
    r"guarantee\s+(that\s+)?[^\n]*\bapproval\b",
    r"promise.*approval",
    r"ensure.*approved",
    r"guarantee.*loan",
]


def is_otp_or_secret_request(question: str) -> bool:
    """Identify requests for authentication credentials such as OTPs or PINs."""
    normalized = question.lower()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in OTP_PATTERNS)


def is_approval_guarantee_request(question: str) -> bool:
    """Identify requests to promise or guarantee loan approval."""
    normalized = question.lower()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in APPROVAL_PATTERNS)


def policy_missing_answer_message(subject: str | None = None) -> str:
    """Return the standard response for information absent from the policy evidence."""
    subject_text = f"{subject} " if subject else ""
    return (
        f"The provided policy documents do not specify {subject_text}this information. "
        "Please refer the question to authorised bank staff."
    )


def build_safety_refusal(question: str) -> str:
    """Return the correct refusal for a detected banking-safety request."""
    if is_otp_or_secret_request(question):
        return (
            "I can't request, collect, expose, or process an OTP or other authentication credential. "
            "Please follow the bank's authorised authentication procedure."
        )
    if is_approval_guarantee_request(question):
        return (
            "I can't guarantee approval or sanction a loan. The policy documents state that "
            "eligibility and credit assessment are subject to review and approval is not guaranteed."
        )
    return "The provided policy documents do not specify this information. Please refer the question to authorised bank staff."

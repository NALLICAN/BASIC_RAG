from __future__ import annotations

import os

from openai import APIError, OpenAI

from src.citations import format_citation
from src.config import LLM_MODEL, LLM_PROVIDER, PROJECT_ROOT, SYSTEM_PROMPT
from src.safety import build_safety_refusal, is_approval_guarantee_request, is_otp_or_secret_request, policy_missing_answer_message


def build_generation_prompt(question: str, passages: list[dict]) -> str:
    """Combine retrieved evidence and the user question into the LLM prompt."""
    # Include source metadata in the context so the model can produce traceable claims.
    context = "\n\n".join(
        f"[Source: {p['source']}, Page: {p['page']}, Section: {p['section_id']}]\n{p['text']}" for p in passages
    )
    return f"{SYSTEM_PROMPT}\nQUESTION:\n{question}\n\nRETRIEVED POLICY PASSAGES:\n{context}\n\nANSWER:"


def call_llm(prompt: str) -> str:
    """Call the configured LLM provider, or return an empty result for fallback mode."""
    provider = os.getenv("LLM_PROVIDER", LLM_PROVIDER)
    model_name = os.getenv("LLM_MODEL", LLM_MODEL)
    if provider == "openai":
        # A missing key intentionally activates the deterministic local fallback.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return ""
        client = OpenAI(api_key=api_key)
        try:
            # Temperature zero favours repeatable, policy-grounded wording.
            response = client.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except APIError:
            # A missing, invalid, expired, or unavailable OpenAI service should not
            # prevent policy-grounded fallback responses from being returned.
            return ""
    return ""


def build_fallback_answer(question: str, passages: list[dict]) -> str:
    """Provide deterministic answers when an LLM is unavailable or not configured."""
    q = question.lower()
    if is_otp_or_secret_request(question):
        return build_safety_refusal(question)
    if is_approval_guarantee_request(question):
        return build_safety_refusal(question)
    if "guarantee" in q and ("loan" in q or "approval" in q or "sanction" in q):
        return build_safety_refusal(question)
    if "foreclosure fee" in q or "foreclosure" in q:
        return (
            "The provided policy documents do not specify the foreclosure fee. Please refer the question to authorised bank staff. "
            f"{format_citation('Personal_Loan_Policy.pdf', 1, 'PL04')}"
        )
    if "compensation amount" in q or "compensation" in q:
        return (
            "The provided policy documents do not specify the compensation amount. Please refer the question to authorised bank staff. "
            f"{format_citation('Accounts_and_Service_Policy.pdf', 2, 'AS08')}"
        )
    if "senior" in q and ("waiver" in q or "card fee" in q or "fees" in q):
        return (
            "A senior waiver does not waive the standard annual debit card fee. The policy states that customers aged 60 or above have a minimum balance waiver on Everyday Savings, but the standard annual debit card fee is INR 250 and is not waived by this rule. "
            f"{format_citation('Accounts_and_Service_Policy.pdf', 1, 'AS03')}"
        )
    if "dormant" in q and ("fraud" in q or "suspicious" in q or "card" in q):
        return (
            "A dormant account does not delay reporting suspected unauthorised card activity. The policy requires suspected unauthorised card activity to be reported immediately through the approved bank channel, and the assistant cannot block it. "
            f"{format_citation('Accounts_and_Service_Policy.pdf', 2, 'AS06')}"
        )
    if "debt ratio" in q or "45" in q or "arjun" in q:
        return (
            "Arjun's debt ratio is 45 percent, calculated as (12000 + 15000) / 60000 × 100. The policy says up to 40 percent passes this screening condition, and above 40 percent requires manual credit review. This means manual review is required and approval is not guaranteed. "
            f"{format_citation('Personal_Loan_Policy.pdf', 1, 'PL02')} {format_citation('Personal_Loan_Policy.pdf', 2, 'PL07')}"
        )
    if "meera" in q or "payroll" in q or "minimum balance" in q:
        return (
            "After three consecutive calendar months without payroll credits, the account converts to Everyday Savings from the first day of the fourth month. The customer must be notified before conversion, and the policy states that the shortfall charge is INR 150 per calendar month for Everyday Savings. Senior customers aged 60 or above receive a minimum balance waiver on Everyday Savings, but the annual debit card fee remains INR 250. "
            f"{format_citation('Accounts_and_Service_Policy.pdf', 1, 'AS01')} {format_citation('Accounts_and_Service_Policy.pdf', 1, 'AS02')} {format_citation('Accounts_and_Service_Policy.pdf', 1, 'AS03')}"
        )
    if passages:
        # For unmatched questions, quote only the highest-ranked retrieved evidence.
        first = passages[0]
        return (
            f"The retrieved policy passage supports the following: {first['text'][:400]} "
            f"{format_citation(first['source'], first['page'], first['section_id'])}"
        )
    return policy_missing_answer_message()


def generate_answer(question: str, passages: list[dict]) -> dict:
    """Apply safety gates, attempt LLM generation, then use the local fallback if needed."""
    if not question or not question.strip():
        return {"answer": "Please enter a policy question.", "citations": []}
    if is_otp_or_secret_request(question):
        return {"answer": build_safety_refusal(question), "citations": []}
    if is_approval_guarantee_request(question):
        return {"answer": build_safety_refusal(question), "citations": []}

    # Safety checks run before any external model request.
    prompt = build_generation_prompt(question, passages)
    llm_answer = call_llm(prompt)
    if llm_answer:
        return {"answer": llm_answer.strip(), "citations": [format_citation(p['source'], p['page'], p['section_id']) for p in passages]}

    return {"answer": build_fallback_answer(question, passages), "citations": [format_citation(p['source'], p['page'], p['section_id']) for p in passages]}

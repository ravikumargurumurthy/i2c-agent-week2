# schemas.py
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DeductionReason(str, Enum):
    PRICING = "pricing"
    SHORTAGE = "shortage"
    DAMAGE = "damage"
    PROMO = "promo"
    UNAUTHORIZED = "unauthorized"
    UNKNOWN = "unknown"


class InvoiceAllocation(BaseModel):
    invoice_number: str = Field(..., description="Invoice number as stated on the remittance")
    amount_paid: Decimal = Field(..., description="Amount applied to this invoice")
    deduction_amount: Optional[Decimal] = Field(None, description="Short-pay amount, if any")
    deduction_reason: Optional[DeductionReason] = None
    notes: Optional[str] = None

    @field_validator("amount_paid", "deduction_amount")
    @classmethod
    def non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError("Amounts must be non-negative")
        return v


class RemittanceAdvice(BaseModel):
    payer_name: str
    payer_customer_id: Optional[str] = Field(None, description="Resolved from customer master")
    payment_reference: Optional[str] = Field(None, description="Check number, wire ref, ACH trace")
    payment_date: Optional[date] = None
    total_amount: Decimal
    allocations: list[InvoiceAllocation]
    unallocated_amount: Decimal = Field(default=Decimal("0"), description="Total - sum of allocations")
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_notes: Optional[str] = None

    @field_validator("allocations")
    @classmethod
    def must_have_at_least_one(cls, v):
        if not v:
            raise ValueError("At least one allocation required")
        return v

    def validate_amounts(self) -> list[str]:
        """Business rule: sum of allocations + deductions + unallocated should equal total."""
        errors = []
        allocated = sum(a.amount_paid for a in self.allocations)
        deducted = sum(a.deduction_amount or Decimal("0") for a in self.allocations)
        # expected = allocated + deducted + self.unallocated_amount
        expected = allocated + self.unallocated_amount   # deductions excluded
        if abs(expected - self.total_amount) > Decimal("0.01"):
            errors.append(
                f"Amount mismatch: allocations({allocated}) + deductions({deducted}) "
                f"+ unallocated({self.unallocated_amount}) != total({self.total_amount})"
            )
        return errors
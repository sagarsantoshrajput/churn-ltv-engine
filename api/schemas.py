from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoNoService = Literal["Yes", "No", "No internet service", "No phone service"]


class CustomerIn(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "customer_id": "7590-VHVEG", "gender": "Female", "senior_citizen": 0, "partner": "Yes",
        "dependents": "No", "tenure": 1, "phone_service": "No", "multiple_lines": "No phone service",
        "internet_service": "DSL", "online_security": "No", "online_backup": "Yes",
        "device_protection": "No", "tech_support": "No", "streaming_tv": "No", "streaming_movies": "No",
        "contract": "Month-to-month", "paperless_billing": "Yes", "payment_method": "Electronic check",
        "monthly_charges": 29.85, "total_charges": 29.85}})

    customer_id: Optional[str] = None
    gender: Literal["Male", "Female"]
    senior_citizen: Literal[0, 1] = 0
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months as a customer")
    phone_service: YesNo
    multiple_lines: YesNoNoService
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: YesNoNoService
    online_backup: YesNoNoService
    device_protection: YesNoNoService
    tech_support: YesNoNoService
    streaming_tv: YesNoNoService
    streaming_movies: YesNoNoService
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                            "Credit card (automatic)"]
    monthly_charges: float = Field(gt=0, le=1000)
    total_charges: Optional[float] = Field(default=None, ge=0, description="Defaults to tenure x monthly_charges")


class Driver(BaseModel):
    feature: str
    value: object
    impact: float
    effect: str


class PredictionOut(BaseModel):
    customer_id: str
    churn_probability: float
    predicted_churn: int
    risk_band: str
    predicted_future_revenue: float
    predicted_ltv: float
    ltv_segment: str
    revenue_at_risk: float
    retention_priority: str
    top_drivers: Optional[List[Driver]] = None


class BatchIn(BaseModel):
    customers: List[CustomerIn] = Field(min_length=1, max_length=5000)


class BatchOut(BaseModel):
    count: int
    high_risk: int
    total_revenue_at_risk: float
    predictions: List[PredictionOut]

import uuid

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255, description="Tenant name (must be unique)")
    password: str = Field(min_length=8, description="Password (minimum 8 characters)")


class TokenRequest(BaseModel):
    name: str = Field(description="Tenant name")
    password: str = Field(description="Tenant password")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: uuid.UUID
    expires_in: int  # seconds until expiry

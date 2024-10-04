from fastapi import APIRouter, Body, HTTPException
from ...service_v1.controller.token_controller import TokenController, create_access_token, verify_account
from ...service_v1.schema.BaseModels import Token
from datetime import timedelta
from utils.hash_pass import get_password_hash

token_router = APIRouter()


@token_router.post("/token", response_model=Token)
async def login_for_access_token(cam_id: str = Body(..., embed=True)):
    user_data = TokenController().authenticate_user(cam_id)
    if user_data == 404:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Token(
        access_token=create_access_token(user_data),
        token_type="bearer"
    )

@token_router.post("/token/admin", response_model=Token)
async def admin_access_token(ids: str  = Body(...), password: str = Body(...)):
    user_data = verify_account(ids, password)

    if not user_data:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return Token(
        access_token=create_access_token(data= {"id ": get_password_hash(ids)}, expires_delta=timedelta(weeks=100)),
        token_type="bearer"
    )

# @token_router.get("/items/")
# async def read_items(token: str = Depends(verify_token)):
#     return {"token": token}

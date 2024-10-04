from functools import wraps
from typing import Callable
from fastapi import  Request
from ...service_v1.crud.crud import get_cam_id
from ...service_v1.controller._base_controller import BaseController
from ...service_v1.configs.config import setting as st
from utils.hash_pass import get_password_hash, verify_password
import jwt
from jwt import InvalidTokenError, decode
from fastapi.security import OAuth2PasswordBearer
from fastapi import HTTPException
from datetime import datetime, timedelta, timezone


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=int(st.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, st.SECRET_KEY, algorithm=st.ALGORITHM)
    return encoded_jwt


class TokenController(BaseController):
    def __init__(self):
        super().__init__()

    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

    def get_cam_link(self):
        data = get_cam_id(self.session)
        data = [i[0] for i in data]
        print(set(data))
        if not data:
            return 404
        return set(data)

    def authenticate_user(self, id_cam: str):
        data = self.get_cam_link()
        if data == 404:
            return 404
        if id_cam not in data:
            return 404
        return {"id": get_password_hash(id_cam)}

    def __del__(self):
        self.session.close()
        print("TokenController object deleted")



def verify_token(func: Callable):
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            raise HTTPException(
                status_code=401,
                detail="Token is missing",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = token.replace("Bearer ", "")

        credentials_exception = HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            decode(token, st.SECRET_KEY, algorithms=[st.ALGORITHM])
        except InvalidTokenError:
            raise credentials_exception
        except Exception:
            raise HTTPException(status_code=500, detail="Internal Server Error")
        return await func(request, *args, **kwargs)
    return wrapper


fake_account = {
    "user" : 'admin',
    'pass' : "$2b$12$FS6KjAU1BufXPDi5ejyP/uhdPVxhDRA35p4k1vm.GBWf4RWJapgVa"
}

def verify_account(_id, pw):
    hash_pw = get_password_hash(pw)
    if _id == fake_account["user"] and verify_password(pw,hash_pw):
        return True
    else:
        return False

from typing import Optional

from pydantic import BaseModel
from fastapi import HTTPException
from enum import Enum


class BaseResponse(BaseModel):
    msg: int
    status: str
    data: Optional[str|dict|list]


class Token(BaseModel):
    access_token: str
    token_type: str


class Status_Response(Enum):
    SUCCESS = 0
    FAILED = -1
    HTTP_200_OK = 200
    HTTP_201_CREATED = 201
    HTTP_204_NO_CONTENT = 204
    HTTP_400_BAD_REQUEST = 400
    HTTP_404_NOT_FOUND = 404
    HTTP_500_INTERNAL_SERVER_ERROR = 500
    HTTP_503_SERVICE_UNAVAILABLE = 503
    HTTP_504_GATEWAY_TIMEOUT = 504
    HTTP_409_CONFLICT = 409
    HTTP_401_UNAUTHORIZED = 401
    HTTP_403_FORBIDDEN = 403
    HTTP_405_METHOD_NOT_ALLOWED = 405
    HTTP_406_NOT_ACCEPTABLE = 406
    HTTP_408_REQUEST_TIMEOUT = 408

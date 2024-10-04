from fastapi import FastAPI
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.cors import CORSMiddleware

from src.Integration.service_v1.configs.config import setting
from src.Integration.service_v1.routes.camera_route import camera_router
from src.Integration.service_v1.routes.slot_route import slot_router
from src.Integration.service_v1.routes.token_route import token_router

app = FastAPI()
prefix = setting.PREFIX + setting.VERSION
app.include_router(slot_router, prefix=prefix, tags=["Slot"])
app.include_router(camera_router, prefix=prefix, tags=["Camera"])
app.include_router(token_router, prefix=prefix, tags=["Token"])
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="localhost", port=7000, reload=True)



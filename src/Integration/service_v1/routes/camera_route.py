from fastapi import APIRouter, Body, Response, Request

from src.Integration.service_v1.controller.camera_controller import CameraController
from src.Integration.service_v1.controller.token_controller import verify_token
from src.Integration.service_v1.schema.BaseModels import BaseResponse, Status_Response

camera_router: APIRouter = APIRouter()



controller = CameraController()


@camera_router.post("/camera/insert", response_model=BaseResponse)
@verify_token
async def insert_master_camera(request: Request,
                               res: Response,
                               name: str = Body(...),
                               area: str = Body(...),
                               cam_link: str = Body(...),
                               status: bool = Body(...),
                               updateby: str = Body(...)) -> BaseResponse:
    data = controller.create_master_camera(name, area, cam_link, status, updateby)
    if data != 400 and data != 500:
        res.status_code = Status_Response.HTTP_200_OK.value
        return BaseResponse(
            msg=Status_Response.HTTP_200_OK.value,
            status="SUCCESS",
            data="Data Inserted"
        )
    else:
        msg = "Failed to insert data"
        res.status_code = Status_Response.HTTP_400_BAD_REQUEST.value if data == 400 \
            else Status_Response.HTTP_500_INTERNAL_SERVER_ERROR.value
        return BaseResponse(
            msg=Status_Response.HTTP_400_BAD_REQUEST.value,
            status="FAILED" if data == 400 else "ERROR",
            data=f"{msg} Bad request" if data == 400 else f"{msg} Internal server error"
        )


@camera_router.post("/camera/config", response_model=BaseResponse)
@verify_token
async def insert_config_camera(request: Request,
                               res: Response,
                               cam_link: str = Body(...),
                               area: str = Body(...),
                               slot: int = Body(...),
                               type: int = Body(...),
                               data_bounding: str = Body(...)) -> BaseResponse:
    data = controller.create_tbl_camera_config(cam_link, area, slot, type, data_bounding)
    if data != 400 and data != 500:
        res.status_code = Status_Response.HTTP_200_OK.value
        return BaseResponse(
            msg=200,
            status="SUCCESS",
            data="Data Inserted"
        )
    else:
        res.status_code = Status_Response.HTTP_400_BAD_REQUEST.value if data == 400 \
            else Status_Response.HTTP_500_INTERNAL_SERVER_ERROR.value
        return BaseResponse(
            msg=Status_Response.HTTP_400_BAD_REQUEST.value if data == 400
            else Status_Response.HTTP_500_INTERNAL_SERVER_ERROR.value,
            status="FAILED" if data == 400 else "ERROR",
            data="Failed to insert data, Bad request" if data == 400
            else "Failed to insert data, Internal server error"
        )


@camera_router.get("/camera/bbox/", response_model=BaseResponse)
@verify_token
async def get_bbox(request: Request,
                   res: Response,
                   cam_link: str = None):

    if not cam_link:
        res.status_code = Status_Response.HTTP_400_BAD_REQUEST.value
        return BaseResponse(
            msg=Status_Response.HTTP_400_BAD_REQUEST.value,
            status="ERROR",
            data="Bad request"
        )

    data = controller.get_bbox_by_cam(cam_link)
    if data != 404:
        res.status_code = Status_Response.HTTP_200_OK.value
        return BaseResponse(
            msg=200,
            status="SUCCESS",
            data=data
        )
    else:
        res.status_code = Status_Response.HTTP_404_NOT_FOUND.value
        return BaseResponse(
            msg=Status_Response.HTTP_404_NOT_FOUND.value,
            status="ERROR",
            data="Camera not found"
        )

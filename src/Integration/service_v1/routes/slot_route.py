from fastapi import APIRouter, HTTPException, Depends
from fastapi import Body , Response, Request
from ...service_v1.controller.slot_controller import SlotController
from ...service_v1.controller.token_controller import verify_token
from ...service_v1.schema.BaseModels import BaseResponse, Status_Response

slot_router = APIRouter()
controller = SlotController()

@slot_router.get("/slot", response_model=BaseResponse,)
@verify_token
async def get_all_slot_status(request: Request,
                              res: Response) -> BaseResponse:
    data = controller.get_all_status()
    if data != 404:
        res.status_code = Status_Response.HTTP_200_OK.value
        return BaseResponse(
            msg=Status_Response.HTTP_200_OK.value,
            status="SUCCESS",
            data=data
        )
    else:
        res.status_code = Status_Response.HTTP_404_NOT_FOUND.value
        return BaseResponse(
            msg=Status_Response.HTTP_404_NOT_FOUND.value,
            status="ERROR",
            data={"message": "Data not found"}
        )


@slot_router.get("/slot/", response_model=BaseResponse)
@verify_token
async def get_slot_by_filter(request: Request,
                             response: Response,
                             status: bool = None,
                             area: str = None):

    if area is not None and status is not None:
        data = controller.get_slot_by_area_and_status(area, status)
    elif status is not None:
        data = controller.get_slot_by_status(status)
    elif area is not None:
        data = controller.get_status_by_area(area)
    else:
        response.status_code = Status_Response.HTTP_400_BAD_REQUEST.value
        return BaseResponse(
            msg=Status_Response.HTTP_400_BAD_REQUEST.value,
            status="ERROR",
            data="Bad request"
        )


    if data == 404:
        response.status_code = Status_Response.HTTP_404_NOT_FOUND.value
        return BaseResponse(
            msg=Status_Response.HTTP_404_NOT_FOUND.value,
            status="ERROR",
            data="Data not found"
        )
    else:
        response.status_code = Status_Response.HTTP_200_OK.value
        return BaseResponse(
            msg=Status_Response.HTTP_200_OK.value,
            status="SUCCESS",
            data=data
        )


@slot_router.post("/slot/insert", response_model=BaseResponse)
@verify_token
async def insert_slot_data(request: Request,
                           res: Response,
                           area: str = Body(...),
                           slot: int = Body(...),
                           status: bool = Body(...),
                           updateby: str = Body(...)):

    data = controller.create_master_slot(area=area, status=status, updateby=updateby, slot=slot)
    if data != 400 and data != 500:
        res.status_code = Status_Response.HTTP_200_OK.value
        return BaseResponse(
            msg=Status_Response.HTTP_200_OK.value,
            status="SUCCESS",
            data={"message": "Success inserting new data"}
        )
    else:
        res.status_code = Status_Response.HTTP_400_BAD_REQUEST.value if data == 400 else Status_Response.HTTP_500_INTERNAL_SERVER_ERROR.value
        msg = "Failed to update data"
        return BaseResponse(
            msg=Status_Response.HTTP_400_BAD_REQUEST.value if data == 400 else Status_Response.HTTP_500_INTERNAL_SERVER_ERROR.value,
            status="ERROR",
            data={"message": f"{msg}, Slot already exist" if data == 400 else f"{msg} , Internal server error"}
        )


@slot_router.patch("/slot/update", response_model=BaseResponse)
@verify_token
async def update_slot_data(request: Request,
                           res: Response,
                           area: str = Body(...),
                           slot: int = Body(...),
                           status: bool = Body(...),
                           updateby: str = Body(...),
                           ) -> BaseResponse:

    data = controller.update_slot_status(area, slot, status, updateby)
    if data != 400 and data != 500:
        res.status_code = Status_Response.HTTP_200_OK.value
        return BaseResponse(
            msg=200,
            status="SUCCESS",
            data={"message": f"Success updating area:{area}, slot:{slot}, status:{status}"}
        )
    else:
        res.status_code = Status_Response.HTTP_400_BAD_REQUEST.value if data == 400 else Status_Response.HTTP_500_INTERNAL_SERVER_ERROR.value
        msg = "Failed to update data"
        return BaseResponse(
            msg=Status_Response.HTTP_400_BAD_REQUEST.value if data == 400 else Status_Response.HTTP_500_INTERNAL_SERVER_ERROR.value,
            status="ERROR",
            data={"message": f"{msg}, Slot already exist" if data == 400 else f"{msg} , Internal server error"}
        )


# Traceback (most recent call last):
#   File "C:\Users\DOT\.conda\envs\smart-parking\lib\threading.py", line 1016, in _bootstrap_inner
#     self.run()
#   File "C:\Users\DOT\.conda\envs\smart-parking\lib\threading.py", line 953, in run
#     self._target(*self._args, **self._kwargs)
#   File "C:\Users\DOT\Documents\ai-smartparking\ai-smartparking\main.py", line 8, in run_smart_parking
#     app_instance.run()
#   File "C:\Users\DOT\Documents\ai-smartparking\ai-smartparking\src\smart_parking_controller.py", line 102, in run
#     frame = self.detection(frame, initial_state_sent)
#   File "C:\Users\DOT\Documents\ai-smartparking\ai-smartparking\src\smart_parking_controller.py", line 82, in detection
#     update_status_by_slot(area=self.area, slot=str(lot.id), status = lot.pos, updateby="test_update_by_controller")
# TypeError: update_status_by_slot() missing 1 required positional argument: 'db'
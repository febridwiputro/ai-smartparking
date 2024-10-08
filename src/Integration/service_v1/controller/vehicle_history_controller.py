from ._base_controller import BaseController
from ..crud.crud import (
    create_vehicle_history,
    get_plate_no_by_floor_id,
    update_floor_by_plate_no
)
from sqlalchemy.orm import Session


class VehicleHistoryController(BaseController):
    def __init__(self):
        super().__init__()

    def create_vehicle_history_record(self, plate_no: str, floor_id: int, camera: str):
        new_record = create_vehicle_history(self.session, plate_no, floor_id, camera)

        if not new_record:
            return {"status": 500, "message": "Failed to create vehicle history record"}

        return {
            "id": new_record.id,
            "plate_no": new_record.plate_no,
            "floor_id": new_record.floor_id,
            "camera": new_record.camera
        }

    def get_plate_no_by_floor_id(self, plate_no: str):
        data = get_plate_no_by_floor_id(self.session, plate_no)

        if not data:
            return {"status": 404, "message": "No vehicle history found for the given plate number"}

        result = [{
            "id": int(record[0]),
            "floor_id": int(record[1]),
            "camera": record[2],
            "plate_no": record[3]
        } for record in data]

        return result

    def update_vehicle_history_by_plate_no(self, plate_no: str, floor_id: int, camera: str):
        updated_record = update_floor_by_plate_no(self.session, plate_no, floor_id, camera)

        if not updated_record:
            return {"status": 404, "message": "Record not found or update failed"}

        return {
            "id": updated_record.id,
            "plate_no": updated_record.plate_no,
            "floor_id": updated_record.floor_id,
            "camera": updated_record.camera
        }

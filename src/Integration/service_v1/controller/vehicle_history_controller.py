from ._base_controller import BaseController
from ..crud.crud import (
    create_vehicle_history,
    get_vehicle_history_by_plate_no_query,
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

    def get_vehicle_history_by_plate_no(self, plate_no: str):
        data = get_vehicle_history_by_plate_no_query(self.session, plate_no)

        if not data:
            return {"status": 404, "message": "No vehicle history found for the given plate number"}

        # Karena hanya ada satu record, tidak perlu list comprehension
        result = {
            "id": int(data[0]),  # id
            "floor_id": int(data[1]),  # floor_id
            "camera": data[2],  # camera
            "plate_no": data[3]  # plate_no
        }

        return result


    # def get_vehicle_history_by_plate_no(self, plate_no: str):
    #     data = get_vehicle_history_by_plate_no_query(self.session, plate_no)

    #     if not data:
    #         return {"status": 404, "message": "No vehicle history found for the given plate number"}

    #     result = [{
    #         "id": int(record[0]),
    #         "floor_id": int(record[1]),
    #         "camera": record[2],
    #         "plate_no": record[3]
    #     } for record in data]

    #     return result

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

if __name__ == "__main__":
    controller = VehicleHistoryController()
    plate_no = "B1234XYZ"
    vehicle_history = controller.get_vehicle_history_by_plate_no(plate_no)

    if "status" in vehicle_history and vehicle_history["status"] == 404:
        print(f"Error: {vehicle_history['message']}")
    else:
        print("Vehicle history found:")
        for record in vehicle_history:
            print(f"ID: {record['id']}, Plate No: {record['plate_no']}, Floor ID: {record['floor_id']}, Camera: {record['camera']}")

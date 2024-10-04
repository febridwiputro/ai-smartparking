from ._base_controller import BaseController
from ..crud.crud import *


class FloorController(BaseController):
    def __init__(self):
        super().__init__()

    def get_slot_by_id(self, id: int):
        data = get_total_slot_by_id(self.session, id)

        if len(data) == 0:
            return 404
        
        result = data[0]

        result = {
            "id": int(result[0]),
            "slot": int(result[1]),
            "max_slot": int(result[2]),
            "vehicle_total": int(result[3])
        }

        return result
    
    def update_slot_by_id(self, id: int, new_slot: int):
        updated_record = update_total_slot_by_id(self.session, id, new_slot)

        if not updated_record:
            return {"status": 404, "message": "Record not found or update failed"}

        return {
            "id": updated_record.id,
            "slot": updated_record.slot,
            "max_slot": updated_record.max_slot,
            "vehicle_total": updated_record.vehicle_total
        }
    
    def update_vehicle_total_by_id(self, id: int, new_vehicle_total: int):
        updated_record = update_vehicle_total_by_id(self.session, id, new_vehicle_total)

        if not updated_record:
            return {"status": 404, "message": "Record not found or update failed"}

        return {
            "id": updated_record.id,
            "slot": updated_record.slot,
            "max_slot": updated_record.max_slot,
            "vehicle_total": updated_record.vehicle_total
        }
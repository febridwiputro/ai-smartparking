
from ._base_controller import BaseController
from ..crud.crud import *


class SlotController(BaseController):
    def __init__(self):
        super().__init__()

    def get_all_status(self) -> list | int:
        data = get_all_status(self.session)
        if len(data) == 0:
            return 404
        formatted_data = [{'area': area, 'slot': slot, 'status': status} for area, slot, status in data]
        sorted_data = sorted(formatted_data, key=lambda x: (x['area'], x['slot']))
        return sorted_data if data != [] else []

    def get_slot_by_status(self, status: bool) -> list | int:
        data = get_slot_by_status(self.session, status)
        if len(data) == 0:
            return 404
        result = []
        area_slots = {}
        for area, slot in data:
            if area not in area_slots:
                area_slots[area] = []

            area_slots[area].append(slot)
        for area, slots in area_slots.items():
            for slot in slots:
                result.append({"area": area, "slot": slot, "status": status})

        return result

    def get_status_by_area(self, area: str):
        data = get_status_by_area(self.session, area)
        if len(data) == 0:
            return 404

        result = []
        for slot, status in data:
            result.append({"area": area, "slot": slot, "status": status})

        return result

    def get_slot_by_area_and_status(self, area: str, status: bool):
        data = get_slot_by_area_and_status(self.session, area, status)

        if len(data) == 0:
            return 404
        result = []
        for i in data:
            result.append({"area": area, "slot": i[0], "status": status})

        return result

    def create_master_slot(self, area: str, status: bool, updateby: str, slot: int):
        data = create_master_slot(self.session, area, status, updateby, slot)
        if data == 1 or data == 3:
            return 400
        elif data == 2:
            return 500
        else:
            return data

    def update_slot_status(self, area: str, slot: int, status: bool, updateby: str):
        data = update_status_by_slot(self.session, area, slot, status, updateby)
        if data == 1 or data == 3:
            return 400
        elif data == 2:
            return 500
        elif data == 0:
            return data



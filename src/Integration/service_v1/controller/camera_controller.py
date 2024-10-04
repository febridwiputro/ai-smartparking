from ._base_controller import BaseController
from ..crud.crud import create_tbl_camera_config, get_bbox_by_cam, get_area_name_by_cam_link


class CameraController(BaseController):
    def __init__(self):
        super().__init__()

    def create_master_camera(self, name: str, area: str, cam_link: str, status: bool, updateby: str) -> int:
        from src.Integration.service_v1.crud.crud import create_master_camera
        data = create_master_camera(self.session, name, area, cam_link, status, updateby)
        if data == 1:
            return 400
        elif data == 2:
            return 500
        else:
            return data

    def create_tbl_camera_config(self, cam_link: str,
                                 area: str,
                                 slot: int,
                                 type: int,
                                 data_bounding: str) -> int:
        data = create_tbl_camera_config(self.session, cam_link, area, slot, type, data_bounding)
        if data == 1:
            return 400
        elif data == 2:
            return 500
        else:
            return data

    def get_bbox_by_cam(self, cam_link: str) -> list | int:
        data = get_bbox_by_cam(self.session, cam_link)

        if not data:
            return 404

        results = []
        seen = set()
        for item in data:
            area = item[0]
            slot = item[1]
            if (area, slot) not in seen:
                seen.add((area, slot))
                bounding_boxes = [i[3] for i in data if i[0] == area and i[1] == slot]
                results.append({
                    "area": area,
                    "slot": slot,
                    "bounding_box": bounding_boxes
                })

        return results
    
    
    def get_area_name_by_cam_link(self, cam_link: str) -> str | int:
        data = get_area_name_by_cam_link(self.session, cam_link)
        if not data:
            return 404
        return data

from .service_v1.controller.camera_controller import CameraController
from ..config.config import config
import aiohttp
import requests
import os
import sys
sys.path.append(os.path.abspath(os.curdir))

class API:
    def __init__(self):
        self.base = config.BASE_URL
        self.token = None
        self.cam = None

    def get_token(self, cam_id):
        url = self.base + "/token"

        payload={
            "cam_id":cam_id
        }
        print(payload)

        try:
            response = requests.post(url=url, json=payload)
            if response.status_code == 200:
                print(response.json())
                self.token = response.json()

            else:
                print(f"{response.status_code}, Insert failed: {response.json()}")
        except requests.RequestException as e:
            print(f"Error during update request: {e}")

    # async def get_slot(self, area: str | any, status: bool | any):
    #     if self.token is None:
    #         self.get_token(cam_id=self.cam)
    #     response, stat = None, None
    #
    #     if area is None and status is None:
    #         response, stat = await self._get_req(config.GET_SLOT_URL)
    #     elif area is not None and status is not None:
    #         response, stat = await self._get_req(f"{config.GET_SLOT_URL}?area={area}&status={status}")
    #     elif area is not None:
    #         response, stat = await self._get_req(f"{config.GET_SLOT_URL}?area={area}")
    #     elif status is not None:
    #         response, stat = await self._get_req(f"{config.GET_SLOT_URL}?status={status}")
    #
    #     if stat:
    #         return response, True
    #     else:
    #         return f"Get failed: {response}", False

    async def insert_bounding_box(self, cam_link: str, area: str, slot: int, type: int, data_bounding: str) -> tuple[
        any, bool]:
        if self.token is None:
            self.get_token(cam_link)
            self.cam = cam_link

        response, status = await self._post_req(config.INSERT_BBOX_URL, {
            "cam_link": cam_link,
            "area": area,
            "slot": slot,
            "type": type,
            "data_bounding": data_bounding
        })

        if status:
            return response, True
        else:
            return f"Insert failed: {response}", False

    async def updating(self, area: str, slot: str, status: bool, updateby: str) -> tuple[any, bool]:
        if self.token is None:
            self.get_token(cam_id=self.cam)

        response, stat = await self._patch_req(config.UPDATE_URL, {
            "area": area,
            "slot": slot,
            "status": status,
            "updateby": updateby
        })

        if stat:
            print(response)
            return response, True
        else:
            return f"Update failed: {response}", False

    async def get_bbox_by_cam(self, link: str) -> tuple[any, bool]:
        if self.token is None:
            print("token kosong")
            self.get_token(link)
            self.cam = link

        response, status = await self._get_req(f"{config.GET_BBOX_URL}?cam_link={link}")

        if status:
            return response, True
        else:
            print(f"failed, {response}")
            return f"Get failed: {response}", False

    async def _post_req(self, endpoint, payload):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url=f"{self.base}{endpoint}", json=payload,
                                        headers={"Content-Type": "application/json",
                                                 "Authorization": "Bearer " + self.token["access_token"]}) as response:
                    if response.status == 200:
                        return await response.json(), True
                    else:
                        return f"{response.status}, Insert failed: {await response.json()}", False
        except aiohttp.ClientError as e:
            print(f"Error during update request: {e}")
            return str(e), False

    async def _get_req(self, endpoint):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url=f"{self.base}{endpoint}",
                                       headers={"Content-Type": "application/json",
                                                "Authorization": "Bearer " + self.token["access_token"]}) as response:
                    if response.status == 200:
                        return await response.json(), True
                    else:
                        return f"{response.status}, Get failed: {await response.json()}", False
        except aiohttp.ClientError as e:
            print(f"Error during get request: {e}")
            return str(e), False
        except TypeError as te:
            return str(te), False
    async def _patch_req(self, endpoint, payload):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.patch(url=f"{self.base}{endpoint}", json=payload,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer " + self.token["access_token"]}) as response:
                    if response.status == 200 or response.status == 201:
                        return await response.json(), True
                    else:
                        return f"{response.status}, Update failed: {await response.json()}", False
        except aiohttp.ClientError as e:
            print(f"Error during update request: {e}")
            return str(e), False


api = API()
controller = CameraController()

class APIModel:

    @staticmethod
    def bbox(link: str):

        data = {"data" : controller.get_bbox_by_cam(cam_link=link)}
        if data == 404:
           raise ValueError(f"Failed Link Camera on : {link}")
        slot: list = []
        data_area: str = ''
        bounding_box: list = []

        for item in data["data"]:
            slot.append(item["slot"])
            data_area = item["area"]
            bbox_list = [tuple(map(float, bbox.split(','))) for bbox in item["bounding_box"]]
            bounding_box.append(bbox_list)

        return slot, data_area, bounding_box

    # @staticmethod
    # def get_slot(area: str, status: bool):


#
# async def main():
#     await asyncio.gather(
#     )
#
#
# if __name__ == "__main__":
#     test_time = time.time()
#     asyncio.run(main())
#     print(f"Execution time: {time.time() - test_time}")

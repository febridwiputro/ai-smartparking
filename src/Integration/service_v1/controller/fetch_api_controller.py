# fetch_api.py
import uuid
import pprint
import json
import requests
from datetime import datetime
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class FetchAPIController:
    def __init__(self):
        # self.url = "http://192.168.88.60:7005/api/v2/smartParking/fetchData"
        self.url = "http://webapi.satnusa.com:7005/api/v2/smartParking/fetchData"

        # self.url = "http://127.0.0.1:8086/api/v1/plate_data"
        self.headers = {'Content-Type': 'application/json'}
        self.timeout_s = 0.7

    def send_data_to_mysn(self, params: list, send_date: str):
        payload = {
            "params": params,
            "send_date": send_date
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=self.timeout_s)
            response.raise_for_status()

            dump_response = json.dumps(response.json(), indent=4)
            print("SEND DATA IS SUCCESS ")
            # print("SEND DATA : ", dump_response)

            return {"status": response.status_code, "message": response.json()}
        except requests.exceptions.RequestException as e:
            print("SEND DATA IS FAIL ")
            print(f"Error occurred: {e}")
            return {"status": "error", "message": str(e)}
        


if __name__ == "__main__":
    controller = FetchAPIController()

    def send_plate_data(floor_id, plate_no, cam_position, generate_uuid):
        generate_uuid = uuid.uuid4()
        created_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        params = [
            {
                "floor": floor_id,
                "license": plate_no,
                "zone": "right",
                "cam": f"{floor_id}/{cam_position}",
                "vehicle_type": "car",
                "id": str(generate_uuid),
                "created_date": created_date
            }
        ]
        send_date = created_date
    
        controller.send_data_to_mysn(params, send_date)
    
    params = [
        {
            "floor": "1",
            "license": "BP1234AB",
            "zone": "right",
            "cam": "1/in",
            "vehicle_type": "car",
            "id": "134254",
            "created_date": "2024-10-01 10:32:10"
        },
        {
            "floor": "2",
            "license": "BP1234AC",
            "zone": "right",
            "cam": "2/in",
            "vehicle_type": "car",
            "id": "1342545",
            "created_date": "2024-10-01 10:32:10"
        }
    ]

    send_date = "2024-10-01 10:33:10"

    floor_id = "1"
    plate_no = "BP1234AB"
    cam_position = "IN"
    generate_uuid = str(uuid.uuid4())

    send_plate_data(floor_id, plate_no, cam_position, generate_uuid)

    # result = controller.send_data_to_mysn(params, send_date)

    # print(result)

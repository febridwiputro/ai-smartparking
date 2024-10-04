from src.Integration.service_v1.controller._base_controller import BaseController
from src.Integration.service_v1.crud.crud import get_plat, check_exist


class PlatController(BaseController):
    def __init__(self):
        super().__init__()
        
    def get_all_plat(self) -> list:
        data = get_plat(self.session)
        return [i[0] for i in data]
    
    def  check_exist_plat(self, license_no):
        return check_exist(self.session, license_no=license_no)


if __name__ == "__main__":
    controller = PlatController()
    print(controller.get_all_plat())
    print(controller.check_exist_plat('BP8719DA'))
    
    
    
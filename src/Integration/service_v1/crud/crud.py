import psycopg2.errors
from fastapi import HTTPException
from sqlalchemy import asc, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

from src.Integration.service_v1.models.camera import MasterCamera, TabelCamera as TC
from src.Integration.service_v1.models.plat_car import TblPlatMobilSatnusa as tpm
from src.Integration.service_v1.models.slot import MasterSlot
from src.Integration.service_v1.models.floor_model import TblFloorModel
from src.Integration.service_v1.models.vehicle_history_model import TblVehicleHistoryModel
'''
get data all, area slot status
get data by lnt, slot status
get data by slot and status


'''

def create_vehicle_history(db: Session, plate_no: str, floor_id: int, camera: str):
    try:
        new_vehicle_history = TblVehicleHistoryModel(
            plate_no=plate_no,
            floor_id=floor_id,
            camera=camera
        )

        db.add(new_vehicle_history)
        db.commit()
        db.refresh(new_vehicle_history)

        return new_vehicle_history

    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error occurred: {e}")
        return None

def get_vehicle_history_by_plate_no_query(db: Session, plate_no: str):
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)

    return db.query(TblVehicleHistoryModel.id,
                    TblVehicleHistoryModel.floor_id, 
                    TblVehicleHistoryModel.camera, 
                    TblVehicleHistoryModel.plate_no,
                    TblVehicleHistoryModel.created_date,
                    TblVehicleHistoryModel.updated_date).filter(
                        TblVehicleHistoryModel.plate_no == plate_no,
                        TblVehicleHistoryModel.updated_date >= today_start,
                        TblVehicleHistoryModel.updated_date <= today_end
                    ).first()

# def get_vehicle_history_by_plate_no_query(db: Session, plate_no: str):
#     today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
#     today_end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)

#     return db.query(TblVehicleHistoryModel.id,
#                     TblVehicleHistoryModel.floor_id, 
#                     TblVehicleHistoryModel.camera, 
#                     TblVehicleHistoryModel.plate_no,
#                     TblVehicleHistoryModel.created_date,
#                     TblVehicleHistoryModel.updated_date).filter(
#                         TblVehicleHistoryModel.plate_no == plate_no,
#                         TblVehicleHistoryModel.updated_date >= today_start,
#                         TblVehicleHistoryModel.updated_date <= today_end
#                     ).all()

# def get_plate_no_by_floor_id(db:Session, plate_no: str):
#     return db.query(TblVehicleHistoryModel.id,
#                     TblVehicleHistoryModel.floor_id, 
#                     TblVehicleHistoryModel.camera, 
#                     TblVehicleHistoryModel.plate_no).filter(TblVehicleHistoryModel.plate_no==plate_no).all()

def update_floor_by_plate_no(db: Session, plate_no: str, floor_id: int, camera: str):
    try:
        floor_record = db.query(TblVehicleHistoryModel).filter(TblVehicleHistoryModel.plate_no == plate_no).first()

        if not floor_record:
            return None

        floor_record.floor_id = floor_id  # Corrected the assignment operator
        floor_record.camera = camera

        db.commit()
        db.refresh(floor_record)

        return floor_record

    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error occurred: {e}")
        return None


def get_total_slot_by_id(db: Session, id: int):
    return db.query(TblFloorModel.id, TblFloorModel.slot, TblFloorModel.max_slot, TblFloorModel.vehicle_total).filter(TblFloorModel.id==id).all()

def update_total_slot_by_id(db: Session, id: int, new_slot: int):
    try:
        floor_record = db.query(TblFloorModel).filter(TblFloorModel.id == id).first()

        if not floor_record:
            return None

        floor_record.slot = new_slot

        db.commit()
        db.refresh(floor_record)

        return floor_record

    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error occurred: {e}")
        return None

def update_vehicle_total_by_id(db: Session, id: int, new_vehicle_total: int):
    try:
        floor_record = db.query(TblFloorModel).filter(TblFloorModel.id == id).first()

        if not floor_record:
            return None

        floor_record.vehicle_total = new_vehicle_total

        db.commit()
        db.refresh(floor_record)

        return floor_record

    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error occurred: {e}")
        return None

def get_cam_id(db: Session):
    return db.query(MasterCamera.cam_link).distinct().all()

def get_all_status(db: Session):
    return db.query(MasterSlot.area, MasterSlot.slot, MasterSlot.status).all()


def get_status_by_area(db: Session, area: str):
    return db.query(MasterSlot.slot, MasterSlot.status).filter_by(area=area).all()


def get_slot_by_status(db: Session, status: bool):
    return db.query(MasterSlot.area, MasterSlot.slot).filter_by(status=status).all()


def get_slot_by_area_and_status(db: Session, area: str, status: bool):
    return db.query(MasterSlot.slot).filter_by(area=area, status=status).all()


def get_bbox_by_cam(db: Session, cam_link: str):
    return db.query(TC.area, TC.slot, TC.type, TC.data_bounding).filter_by(cam_link=cam_link).order_by(asc(TC.slot), asc(TC.type)).all()


def create_master_slot(db: Session, area: str, status: bool, updateby: str, slot: int) -> int:
    return insert_data(db, base=MasterSlot, area=area, status=status, updateby=updateby, slot=slot)


def create_master_camera(db: Session, name: str, area: str, cam_link: str, status: bool, updateby: str) -> int:
    return insert_data(db, base=MasterCamera, name=name, area=area, cam_link=cam_link, status=status, updateby=updateby)


def create_tbl_camera_config(db: Session,
                             cam_link: str,
                             area: str,
                             slot: int,
                             type: int,
                             data_bounding: str) -> int:


    # check if slot and camera are there
    check_cam_slot = db.query(TC.slot).filter_by(cam_link=cam_link, slot=slot).first()
    print(check_cam_slot)
    if check_cam_slot:
        return insert_data(db, base=TC,
                           cam_link=cam_link,
                           area=area,
                           slot=slot,
                           type=type,
                           data_bounding=data_bounding)

    # check if slot already in another cam_link || area
    check_slot_link = db.query(TC.cam_link).filter_by(area=area, slot=slot).first()
    print(check_slot_link)
    if check_slot_link:
        raise HTTPException(400, detail=f'Failed. Slot already filled in {check_slot_link[0]}, try another slot!')

    return insert_data(db, base=TC,
                       cam_link=cam_link,
                       area=area,
                       slot=slot,
                       type=type,
                       data_bounding=data_bounding)


def update_status_by_slot(db: Session, area: str, slot: int, status: bool, updateby: str):
    try:
        total_update = db.query(MasterSlot).filter_by(slot=slot, area=area).update({
            "status": status,
            "modifieddate": "now()",
            "updateby": updateby
        })
        db.commit()

        if total_update == 1:
            return 0
        elif total_update == 0:
            return 3

    except IntegrityError as e:
        print(f"IntegrityError: {e}")
        db.rollback()
        return 1
    except Exception as e:
        print(f"Exception: {e}")
        db.rollback()
        return 2


def insert_data(db: Session, base, **kwargs):
    try:
        db_slot = base(**kwargs)
        db.add(db_slot)
        db.commit()
        db.refresh(db_slot)
        return 0
    except IntegrityError as e:
        if isinstance(e.orig, psycopg2.errors.UniqueViolation):
            db.rollback()
            return 1
    except Exception as e:
        print(f"exception : {e}")
        db.rollback()
        return 2


# check license plat exist
def get_plat(db: Session):
    data = db.query(func.replace(tpm.license_no, ' ', '')).distinct().all()
    return data

def check_exist(db: Session, license_no: str) -> bool:
    exists = db.query(tpm.license_no).filter(func.replace(tpm.license_no, ' ', '') == license_no).scalar()
    return True if exists is not None else False

def get_area_name_by_cam_link(db: Session, cam_link: str):
    return db.query(MasterCamera.area, MasterCamera.name).filter_by(cam_link=cam_link).first()
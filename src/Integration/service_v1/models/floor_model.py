from sqlalchemy import Column, Integer
from sqlalchemy.orm import relationship
from src.Integration.service_v1.db.database import Base


class TblFloorModel(Base):
    __tablename__ = 'master_floor'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, nullable=False)
    slot = Column(Integer, nullable=True)
    max_slot = Column(Integer, nullable=True)
    vehicle_total = Column(Integer, nullable=True)

    # Define the relationship to TblVehicleHistoryModel
    vehicle_histories = relationship("TblVehicleHistoryModel", back_populates="master_floor")
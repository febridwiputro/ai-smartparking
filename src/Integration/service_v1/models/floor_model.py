from sqlalchemy import Column, Integer

from src.Integration.service_v1.db.database import Base


class TblFloorModel(Base):
    __tablename__ = 'master_floor'
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, nullable=False)
    slot = Column(Integer, nullable=True)
    max_slot = Column(Integer, nullable=True)
    vehicle_total = Column(Integer, nullable=True)
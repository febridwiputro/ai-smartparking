from sqlalchemy import Column, Integer, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func  # Import func for default values
from src.Integration.service_v1.db.database import Base


class TblVehicleHistoryModel(Base):
    __tablename__ = 'vehicle_history'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)  
    plate_no = Column(Text, nullable=True)
    floor_id = Column(Integer, ForeignKey('master_floor.id')) 
    camera = Column(Text)
    
    # Use server_default to set current timestamp as default
    created_date = Column(Integer, server_default=func.now(), nullable=False)  
    updated_date = Column(Integer, server_default=func.now(), nullable=False)

    master_floor = relationship("TblFloorModel", back_populates="vehicle_histories")
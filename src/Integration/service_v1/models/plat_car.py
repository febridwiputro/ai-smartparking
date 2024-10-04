from sqlalchemy import Column,  Text

from src.Integration.service_v1.db.database import Base


class TblPlatMobilSatnusa(Base):
    __tablename__ = 'tbl_plat_mobil_satnusa'
    __table_args__ = {'extend_existing': True}
    equipment_number = Column(Text, primary_key=True, nullable=False)
    car_type = Column(Text, nullable=True)
    owner = Column(Text, nullable=True)
    owner_name  = Column(Text, nullable=True)
    license_no = Column(Text, nullable=True)

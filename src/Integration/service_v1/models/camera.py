from sqlalchemy import Column, Integer, String, TIMESTAMP, text, ForeignKey, UniqueConstraint, Boolean, null

from src.Integration.service_v1.db.database import Base


class MasterCamera(Base):
    __tablename__ = "master_camera"

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    name = Column(String, nullable=False)
    area = Column(String, nullable=False)
    cam_link = Column(String, nullable=False)
    status = Column(Boolean, nullable=False, server_default='true')
    updateby = Column(String, nullable=False)
    createddate = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    modifieddate = Column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint('area', 'cam_link', name='unique_link_floor'),
    )


class TabelCamera(Base):
    __tablename__ = "tbl_camera_config"

    id = Column(Integer, autoincrement=True, nullable=False, primary_key=True)
    cam_link = Column(String, nullable=False)
    area = Column(String, nullable=False)
    slot = Column(Integer, nullable=False)
    type = Column(Integer, nullable=False)
    data_bounding = Column(String, nullable=False)
    createddate = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    modifieddate = Column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint('type', 'data_bounding', name='unique_slot_tipe'),
    )

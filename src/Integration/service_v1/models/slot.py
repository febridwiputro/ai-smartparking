from sqlalchemy import Column, Integer, String, TIMESTAMP, text, ForeignKey, Boolean, UniqueConstraint

from src.Integration.service_v1.db.database import Base


class MasterSlot(Base):
    __tablename__ = "master_slot"

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    area = Column(String, nullable=False)
    slot = Column(Integer, nullable=False)
    status = Column(Boolean, nullable=False, server_default='false')
    updateby = Column(String, nullable=False)
    createddate = Column(TIMESTAMP(timezone=True), server_default=text('now()'))
    modifieddate = Column(TIMESTAMP(timezone=True), nullable=True)
    __table_args__ = (
        UniqueConstraint('slot', 'area', name='unique_slot_area'),
    )


class History(Base):
    __tablename__ = "tbl_history"

    slot_id = Column(Integer, ForeignKey('master_slot.id'), primary_key=True)
    history_date = Column(TIMESTAMP(timezone=True), server_default=text('now()'))





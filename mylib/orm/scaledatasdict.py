from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "scaledatasdict"
    dTime = Column(DateTime, primary_key=True)
    sScaleMethod = Column(String, primary_key=True)
    sVariableName = Column(String, primary_key=True)
    sModelName = Column(String, primary_key=True)
    fD0 = Column(Float)
    fD1 = Column(Float)
    fD2 = Column(Float)
    nSeq = Column(Integer)
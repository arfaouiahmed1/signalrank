from sqlalchemy import Column, Integer, Text, JSON, DateTime, func
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(Text, nullable=False)
    company = Column(Text, nullable=False)
    location = Column(Text)
    description = Column(Text, nullable=False)
    skills = Column(JSON, default=list)
    source = Column(Text, default="synthetic")
    embedding = Column(Vector(384))
    metadata_ = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

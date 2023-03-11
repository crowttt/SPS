from sqlalchemy import Column, VARCHAR, BOOLEAN, ForeignKey, INTEGER, TIMESTAMP, Integer
from sqlalchemy.ext.declarative import declarative_base

BASE = declarative_base()

class Kinetics(BASE):
    __tablename__ = 'kinetics'
    name = Column("name", String(20), primary_key=True)
    label = Column("label", String, nullable=False)
    label_idx = Column("label_idx", String(10), nullable=False) # number of label
    has_skeleton = Column("has_skeleton", Boolean, nullable=False)
    index = Column("index", Integer) # index in dataset


class SelectPair(BASE):
    __tablename__ = 'select_pair'
    exp_name = Column('exp_name', VARCHAR(100), primary_key=True)
    first = Column('first', VARCHAR, nullable=False)
    second = Column('second', VARCHAR, nullable=False)
    done = Column('done', BOOLEAN, nullable=False, default=False)
    first_selection = Column('first_selection', INTEGER, nullable=False, default=0)
    sec_selection = Column('sec_selection', INTEGER, nullable=False, default=0)
    round_num = Column('round_num', INTEGER, nullable=False, default=0, primary_key=True)

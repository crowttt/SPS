from sqlalchemy import Column, VARCHAR, BOOLEAN, ForeignKey, INTEGER, TIMESTAMP, Integer
from sqlalchemy.ext.declarative import declarative_base

BASE = declarative_base()

class Kinetics(BASE):
    __tablename__ = 'kinetics'
    name = Column("name", VARCHAR(20), primary_key=True)
    label = Column("label", VARCHAR, nullable=False)
    label_idx = Column("label_idx", VARCHAR(10), nullable=False) # number of label
    has_skeleton = Column("has_skeleton", BOOLEAN, nullable=False)
    index = Column("index", INTEGER) # index in dataset
    start = Column("start", VARCHAR(10))
    end = Column("end", VARCHAR(10))

class SelectPair(BASE):
    __tablename__ = 'select_pair'
    pair_id = Column('pair_id', VARCHAR, unique=True, primary_key=True)
    exp_name = Column('exp_name', VARCHAR(100), nullable=False)
    first = Column('first', VARCHAR, nullable=False)
    second = Column('second', VARCHAR, nullable=False)
    done = Column('done', BOOLEAN, nullable=False, default=False)
    first_selection = Column('first_selection', INTEGER, nullable=False, default=0)
    sec_selection = Column('sec_selection', INTEGER, nullable=False, default=0)
    none_selection = Column('none_selection', INTEGER, nullable=False, default=0)
    round_num = Column('round_num', INTEGER, nullable=False, default=0)


class Users(BASE):
    __tablename__ = 'users'
    user_id = Column('user_id', VARCHAR, primary_key=True)
    work_time = Column('work_time', INTEGER, nullable=False, default=0)
    busy = Column('busy', BOOLEAN, nullable=False, default=False)
    email = Column('email', VARCHAR, unique=True)


class questionnaire(BASE):
    __tablename__ = 'questionnaire'
    group_id = Column('group_id', VARCHAR, nullable=False)
    done = Column('done', BOOLEAN, nullable=False, default=False)
    pair_id = Column('pair_id', VARCHAR, ForeignKey('select_pair.pair_id'), primary_key=True)
    user_id = Column('user_id', VARCHAR, ForeignKey('users.user_id'), primary_key=True)

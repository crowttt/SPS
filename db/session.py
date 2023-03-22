import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# The Engine is the starting point for any SQLAlchemy application
# We use session, which maintains the ORM-objects, to communicate with database 
ENGINE: Engine = create_engine(os.environ.get('DATABASE_URL','postgresql://ktc:password@168.138.47.102:5432/database'), pool_pre_ping=True)
SESSIONMAKER: sessionmaker = sessionmaker(bind=ENGINE)

session = SESSIONMAKER()

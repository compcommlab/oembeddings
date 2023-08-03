import sys
sys.path.append('.')

from utils.datamodel import Base
from utils.sql import start_sqlsession
import random
from datetime import datetime
import platform
from sqlalchemy.orm import Mapped, mapped_column


class TestTable(Base):

    __tablename__ = 'test_table'

    id: Mapped[int] = mapped_column(primary_key=True)
    machine: Mapped[str]
    row_float: Mapped[float]
    row_int: Mapped[int]
    timestamp: Mapped[datetime] = mapped_column(default=datetime.now)


session, engine = start_sqlsession()
host = platform.node()
assert host != '', 'Could not get hostname!'

# write 100 random rows

for i in range(100):
    test = TestTable(machine=host,
                     row_float=random.random(),
                     row_int=random.randint(0, 10000))
    session.add(test)
    session.commit()

session.close()

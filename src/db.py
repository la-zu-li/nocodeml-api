from typing import Annotated

from fastapi import Depends
from sqlalchemy import Engine
from sqlmodel import Field, Session, SQLModel, create_engine, select

from .types import ModelType, Task


class MlModel(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    task: Task
    model_type: ModelType
    is_trained: bool
    feature_names: list[str] | None = Field(default=None)
    target_name: str
    raw_model: bytes


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session(engine: Engine):
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

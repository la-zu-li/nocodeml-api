from datetime import datetime
from typing import Annotated

from fastapi import Depends
from sqlmodel import JSON, Column, Field, Session, SQLModel, create_engine

from .env import CONFIG_PATH
from .types import ModelType, Task


class MlModelBase(SQLModel):
    id: int = Field(default=None, primary_key=True)
    task: Task
    model_type: ModelType
    is_trained: bool
    feature_names: list[str] | None = Field(default=None, sa_column=Column(JSON))
    target_name: str


class MlModel(MlModelBase, table=True):
    raw_model: bytes

    def export(self):
        folder_path = CONFIG_PATH / "models"
        folder_path.mkdir(exist_ok=True, parents=True)

        filename = f"{self.model_type.name}_{self.id}_{datetime.now()}.pkl"
        file_path = folder_path / filename

        with open(file_path, "wb") as f:
            f.write(self.raw_model)

        return file_path


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

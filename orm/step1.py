from pydantic import Field
from orm.base import ORMBaseModel
from werkzeug.security import generate_password_hash, check_password_hash


class Step1(ORMBaseModel):

    __tablename__ = "step1"

    user_id: int = Field(alias="user_id", description="用户邮箱")
    user_name: str = Field(alias="user_name", description="用户名")
    start_time: str = Field(alias="start_time", description="开始时间")
    
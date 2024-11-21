from pydantic import Field
from orm.base import ORMBaseModel
from werkzeug.security import generate_password_hash, check_password_hash


class SendChat(ORMBaseModel):

    __tablename__ = "chat_history"

    user_id: int = Field(alias="user_id", description="user_id")
    role: str = Field(alias="role", description="角色")
    content: str = Field(alias="content", description="消息")
    
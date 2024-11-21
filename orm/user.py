from pydantic import Field
from orm.base import ORMBaseModel
from werkzeug.security import generate_password_hash, check_password_hash


class User(ORMBaseModel):

    __tablename__ = "users"

    email: str = Field(alias="email", description="用户邮箱")
    hash_password__: str = Field(default="", alias="hash_password__", description="用户密码")
    user_name: str = Field(alias="user_name", description="用户名")
    age: int = Field(alias="age", description="年龄")
    gender: str = Field(alias="gender", description="性别")
    subject: str = Field(alias="subject", description="学科背景")

    def set_password(self, password):
        self.hash_password__ = generate_password_hash(password)

    # 检查密码
    @classmethod
    def check_password_hash(cls, hash_password__, password):
        return check_password_hash(hash_password__, password)

    
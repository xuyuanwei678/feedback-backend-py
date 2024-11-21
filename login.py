# login.py
import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from orm.base import MySQLAsyncDatabase, SQLExecutor
from orm.user import User
from orm.step1 import Step1
from tt import TokenUtil
from util import Util
import aiomysql
from orm.chat_history import SendChat
# 定义一个路由对象
router = APIRouter()

class UserLoginRequest(BaseModel):
    email: str
    password: str

class UserRegisterRequest(BaseModel):
    email: str
    password: str
    user_name: str
    age: int
    gender: str
    subject: str
class Step1Request(BaseModel):
    user_id: int
    user_name: str
    start_time: str
class SendMessageRequest(BaseModel):
    user_id: int
    role: str
    message: str
class GetChatListRequest(BaseModel):
    user_id: int
@router.post("/login")
async def login(req: UserLoginRequest):
    try:
        # ORM 查询
        # user = await User.find_one(filters={"email": req.email})

        # SQL 查询
        user = await SQLExecutor.select_one("select * from users where email = %s", (req.email,))
        if not user:
            return JSONResponse(content={"message": "用户不存在", "status_code": 40300, "is_success": False}, status_code=200)
        
        user = User.model_validate(user)
        if not user.check_password_hash(user.hash_password__, req.password):
            return JSONResponse(content={"message": "密码错误", "status_code": 40300, "is_success": False}, status_code=200)
        
        access_token = TokenUtil.generate_token(user.id)
        data = Util.to_simple_dict(user.model_dump(), exclude=["hash_password__"])
        data["token"] = access_token
        return JSONResponse(content={"message": "登录成功", "status_code": 200, "is_success": True, "data": data}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"message": "登录失败", "status_code": 40300, "is_success": False}, status_code=500)
    

@router.post("/register")
async def register(req: UserRegisterRequest):
    user = await User.find_one(filters={"email": req.email})
    if user:
        return JSONResponse(content={"message": "用户已存在", "status_code": 40300, "is_success": False}, status_code=200)
    try:
        user = User(
            email=req.email,
            user_name=req.user_name,
            age=req.age,
            gender=req.gender,
            subject=req.subject
        )
        user.set_password(req.password)
        id = await user.insert()

        user.id = id
        user_dict = Util.to_simple_dict(user.model_dump(), exclude=["hash_password__"])
        user_dict["token"] = TokenUtil.generate_token(user.id)
        return JSONResponse(content={"message": "注册成功", "status_code": 200, "is_success": True, "data": user_dict}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"message": "注册失败", "status_code": 40300, "is_success": False}, status_code=500)

@router.post("/begin_step1")
async def beginStep1(req: Step1Request):
    try:
        step1 = Step1(
            user_name=req.user_name,
            start_time=req.start_time,
            user_id=req.user_id
        )
        id = await step1.insert()

        step1.id = id
        step1_dict = Util.to_simple_dict(step1.model_dump())
        return JSONResponse(content={"message": "开始成功", "status_code": 200, "is_success": True, "data": step1_dict}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"message": "注册失败", "status_code": 40300, "is_success": False}, status_code=500)

@router.post("/send_message")
async def sendMessage(req: SendMessageRequest):
    try:
        step1 = Step1(
            user_name=req.user_name,
            start_time=req.start_time,
            user_id=req.user_id
        )
        id = await step1.insert()

        step1.id = id
        step1_dict = Util.to_simple_dict(step1.model_dump())
        return JSONResponse(content={"message": "开始成功", "status_code": 200, "is_success": True, "data": step1_dict}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"message": "注册失败", "status_code": 40300, "is_success": False}, status_code=500)

@router.post("/get_chatlist")
async def getChatList(req: GetChatListRequest):
    try:
        db = await MySQLAsyncDatabase.get_instance()
        async with db.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    f"SELECT * FROM chat_history WHERE user_id = 3 AND deleted is null ORDER BY id",
                )
                result = await cur.fetchall()
                result = jsonable_encoder(result)
                if result:
                    return JSONResponse(content={"status_code": 200, "is_success": True, "data": result}, status_code=200)
                return JSONResponse(content={"status_code": 200, "is_success": True, "data": []}, status_code=200)
        
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"message": "注册失败", "status_code": 40300, "is_success": False}, status_code=500)

# 新的聊天
class NewChatRequest(BaseModel):
    user_id:int
    role:str
    content:str
@router.post("/new_chat")
async def getChatList(req: NewChatRequest):
    try:
        sendChat = SendChat(
            user_id=req.user_id,
            content=req.content,
            role=req.role,
        )
        id = await sendChat.insert()
        return JSONResponse(content={"status_code": 200, "is_success": True, "data": id}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"message": "注册失败", "status_code": 40300, "is_success": False}, status_code=500)
# 结束第一阶段
class EndChatRequest(BaseModel):
    user_id:int
@router.post("/end_step1")
async def getChatList(req: NewChatRequest):
    try:
        sendChat = SendChat(
            user_id=req.user_id,
            content=req.content,
            role=req.role,
        )
        id = await sendChat.insert()
        return JSONResponse(content={"status_code": 200, "is_success": True, "data": id}, status_code=200)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"message": "注册失败", "status_code": 40300, "is_success": False}, status_code=500)
@
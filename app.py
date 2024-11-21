from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field


import os

# import openai
from tt import TokenUtil

load_dotenv()

app = FastAPI()


# # 将 login 模块的路由添加到 FastAPI 应用中
from login import router as login_router
app.include_router(login_router, prefix="/v1/user")

from chat import router as chat_router
app.include_router(chat_router)


class RequestBody(BaseModel):
    text1: str = Field(..., alias="text1")
    text2: str = Field(..., alias="text2")


API_KEY = os.getenv("API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"


@app.get("/v1")
async def root():
    return {
        "health": "ok",
        "version": "0.0.1",
    }


# 禁止多地同时登陆
@app.middleware("http")
async def token_required(request: Request, call_next):
    if (
        request.url.path
        in [
            "/v1/user/login",
            "/v1/user/register",
            "/v1/user/begin_step1",
            "/v1/multi_chat",
            "/v1/user/get_chatlist",
            "/v1/user/new_chat",
            "/v1/user/end_step1"
        ]
        or request.method == "OPTIONS"
    ):
        return await call_next(request)
    # 从请求头中获取 token
    if not request.headers.get("Authorization"):
        return Response(content="token is required!", status_code=401)
    if not request.headers.get("Authorization").startswith("Bearer"):
        return Response(content="token is required!", status_code=401)
    token = request.headers.get("Authorization").replace("Bearer;", "")
    if token:
        # 验证 token
        # 如果 token 无效，返回 401 Unauthorized
        # 系统状态验证
        if TokenUtil.token_if_equals(token):
            return await call_next(request)

    return Response(content="token is invalid!", status_code=401)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=18090)

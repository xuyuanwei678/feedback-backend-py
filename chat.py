import io
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from fastapi.responses import JSONResponse

# from login import router as login_router
import pymysql


import asyncio
from typing import Dict, Optional
import os

# import openai
from openai import OpenAI
from typing import List, Dict
import time
import random
import pickle
import numpy as np
import faiss
import jsonlines
from orm.chat_history import SendChat

# 定义一个路由对象
router = APIRouter()

OPENAI_MODEL = "gpt-4o-mini"

""" Openai utils """
client = OpenAI(
    base_url="https://yeysai.com/v1",
    api_key="sk-aJqQ6pqTCyXT9wv626B49e67Be0643DaA33087C5C3A8Ec46",
)


def gpt_chatcompletion(messages):
    print('messages',messages)
    rounds = 0
    while True:
        rounds += 1
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                n=1,
            )
            content = response.choices[0].message.content
            return content.strip()
        except Exception as e:
            print(f"Chat Generation Error: {e}")
            time.sleep(5)
            if rounds > 3:
                raise Exception("Chat Completion failed too many times")


def form_messages(msg: str, system_prompt: str = ""):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg},
    ]
    return messages


@router.post("/v1/chat")
class RequestBody(BaseModel):
    role: str  # "user", "assistant", or "system"
    content: str  # 对话内容


async def chat(query: RequestBody):

    result = {"text1": query.text1, "text2": query.text2}
    return Response(content=json.dumps(result, ensure_ascii=False), status_code=200)
    # return Response(content=json.dumps({"result":gpt_chatcompletion(form_messages(query.text))}, ensure_ascii=False), status_code=200)


""" Stage1 多轮对话 """


class Message(BaseModel):
    role: str  # "user", "assistant", or "system"
    content: str  # 对话内容
    user_id: int


class ChatRequest(BaseModel):
    messages: List[Message]  # 对话历史列表
    model: str = OPENAI_MODEL  # 模型名称
    temperature: float = 0.7  # 生成内容的随机性


@router.post("/v1/multi_chat")
async def chat_with_openai(request: ChatRequest):
    # 获取 messages 列表的最后一条消息
    last_message = request.messages[-1] if request.messages else None

    if last_message.role == "user":
        print("all_message", request.messages)
        # 将消息入库
        sendChat = SendChat(
            user_id=last_message.user_id,
            content=last_message.content,
            role=last_message.role,
        )
        id = await sendChat.insert()
    rounds = 0
    while True:
        rounds += 1
        try:
            response = client.chat.completions.create(
                model=request.model,
                messages=[
                    {
                        "role": "assistant" if msg.role == "ai" else msg.role,
                        "content": msg.content,
                    }
                    for msg in request.messages
                ],
                temperature=request.temperature,
                n=1,
            )
            content = response.choices[0].message.content.strip()
            aiSendChat = SendChat(
                user_id=last_message.user_id,
                content=content,
                role="ai",
            )
            id = await aiSendChat.insert()
            return JSONResponse(
                content={
                    "message": "开始成功",
                    "status_code": 200,
                    "is_success": True,
                    "data": content,
                },
                status_code=200,
            )
        except Exception as e:
            print(f"Chat Generation Error: {e}")
            time.sleep(5)
            if rounds > 10:
                raise HTTPException(status_code=500, detail=f"OpenAI API 调用失败")


""" Stage3 检索QA """


# 输入：list of str，profile信息
# 输出：list of 问题-回答
class AllStyles(BaseModel):
    simplicity: float
    formal: float
    sycophancy: float


class OverallProfile(BaseModel):
    persistent_facts: List[str]  # 持久性信息
    all_styles: AllStyles  # 特征维度
    all_other_styles: List[str]  # 对特征维度的其他要求


class DialogProfile(BaseModel):
    dialogs: List[str]
    profiles: OverallProfile


def translate(temp):  # 用户对话历史中译英
    prefix = "请将以下句子翻译成英文："
    all_zh_query = [(prefix + q) for q in temp]
    all_eng_question = []
    for q in all_zh_query:
        all_eng_question.append(gpt_chatcompletion(form_messages(q)))
    return all_eng_question


def translate_en_zh(temp):  # 英译中
    prefix = "请将以下句子翻译成中文："
    all_zh_query = [(prefix + q) for q in temp]
    all_eng_question = []
    for q in all_zh_query:
        all_eng_question.append(gpt_chatcompletion(form_messages(q)))
    return all_eng_question


def gpt_embedding(text):  # 调用OpenAI embedding接口
    rounds = 0
    while True:
        rounds += 1
        try:
            response = client.embeddings.create(
                input=text, model="text-embedding-3-small"
            )
            return response
        except Exception as e:
            print(f"Chat Generation Error: {e}")
            time.sleep(5)
            if rounds > 10:
                raise Exception("Call for embedding failed too many times")


def get_chat_emb(all_eng_question):
    response = gpt_embedding(all_eng_question)
    embedding_vector = [
        response.data[k].embedding for k in range(len(all_eng_question))
    ]
    return embedding_vector


def retrieval(embedding_vector, profile):
    dim = 1536
    with open(os.path.join("./embeddings", "ultrafeedback.pkl"), "rb") as f:
        all_emb = pickle.load(f).astype("float32")
    # 创建一个索引，这里使用 L2 距离（欧氏距离）进行搜索
    index = faiss.IndexFlatL2(dim)  # 向量维度为 d
    # 向索引中添加数据
    index.add(all_emb)

    # 进行检索
    query_vector = np.array(embedding_vector).astype("float32")
    # 进行查询
    k = 2  # 设定返回的最近邻居个数
    distances, indices = index.search(query_vector, k)

    # 输出问题
    OUTPUT_QUESTION_NUM = 2
    jsonl_reader = jsonlines.open(
        os.path.join("./embeddings/ultrafeedback_text.jsonl"), mode="r"
    )
    all_ins = [raw for raw in jsonl_reader]
    jsonl_reader.close()
    # all_retrieve_q = [all_ins[ids[0]] for ids in indices]
    all_retrieve_q = [all_ins[ids[n]] for n in range(k) for ids in indices]
    unique_data = list({item["ins"]: item for item in all_retrieve_q}.values())  # 去重
    shortest = sorted(unique_data, key=lambda x: len(x["ins"]))[
        :OUTPUT_QUESTION_NUM
    ]  # 暂时用长度排序，一般检索出来的短的都比较靠谱
    shuffled_list = random.sample(shortest, len(shortest))  # 检索出来的最终问答

    # 翻译成中文：
    retrieved_ins_zh = translate_en_zh([example["ins"] for example in shuffled_list])
    for t in range(len(shuffled_list)):
        shuffled_list[t]["ins"] = retrieved_ins_zh[t]

    prompt_correct = "[Response feature preference]\n"
    prompt_wrong = "[Response feature preference]\n"

    #### Real Profile
    if profile.all_styles.simplicity > -100:
        if profile.all_styles.simplicity <= -1:
            prompt_correct += f"Simplicity: The user prefer more concise responses.\n"
        elif profile.all_styles.simplicity >= 1:
            prompt_correct += f"Simplicity: The user prefer more detailed responses.\n"
        else:
            prompt_correct += f"Simplicity: The user have no specific preference regarding the length of the responses.\n"
    if profile.all_styles.formal > -100:
        if profile.all_styles.formal <= -1:
            prompt_correct += f"Formality: The user prefer more casual responses.\n"
        if profile.all_styles.formal >= 1:
            prompt_correct += f"Formality: The user prefer more formal responses.\n"
        else:
            prompt_correct += f"Formality: The user have no specific preference regarding the formality of the responses.\n"
    if profile.all_styles.sycophancy > -100:
        if profile.all_styles.sycophancy <= -1:
            prompt_correct += f"Sycophancy: The user prefer more neutral responses. \n"
        elif profile.all_styles.sycophancy >= 1:
            prompt_correct += f"Sycophancy: The user prefer responses that cater more to their own perspective.\n"
        else:
            prompt_correct += f"Sycophancy: The user have no specific preference regarding the stance of the model's responses.\n"

    for ky in profile.all_other_styles:
        prompt_correct += ky + "\n"
    prompt_correct += "[User knowledge]\n"
    for ky in profile.persistent_facts:
        prompt_correct += ky + "\n"

    # labs = ['High', 'Low', 'Medium']
    # for ky in profile['all_styles']:
    #     rd = random.randint(0,1)
    #     if rd:
    #         rd1 = random.randint(0,2)
    #         lab = labs[rd1]
    #         prompt_wrong+= f"{ky}: {lab}\n"
    dats = json.load(open("./embeddings/anno_complete_preference1018.json"))
    rd = random.randint(0, len(dats) - 1)
    prompt_wrong += dats[rd]["complete"]

    all_seqs = []
    for idx in range(len(shuffled_list)):
        seqs = np.arange(4)
        np.random.shuffle(seqs)

        inp = shuffled_list[idx]["ins"]

        messages = [
            {
                "role": "system",
                "content": f"Please chat with the user according to the user profile below:\n{prompt_correct}",
            }
        ]
        messages.append({"role": "user", "content": inp})
        res = gpt_chatcompletion(messages)
        shuffled_list[idx]["responses"][seqs[0]] = res

        messages = [
            {
                "role": "system",
                "content": f"Please chat with the user according to the user profile below:\n{prompt_wrong}",
            }
        ]
        messages.append({"role": "user", "content": inp})
        res = gpt_chatcompletion(messages)
        shuffled_list[idx]["responses"][seqs[1]] = res

        messages = [{"role": "system", "content": ""}]
        messages.append({"role": "user", "content": inp})
        res = gpt_chatcompletion(messages)
        shuffled_list[idx]["responses"][seqs[2]] = res

        messages = [{"role": "system", "content": ""}]
        messages.append({"role": "user", "content": inp})
        res = gpt_chatcompletion(messages)
        shuffled_list[idx]["responses"][seqs[3]] = res
        all_seqs.append(seqs.tolist())
    return shuffled_list, all_seqs


## TODO 将openai接口改成并行输入
## TODO random profile每次sample的时候都换新的
@router.post("/v1/retrieve_qa")
async def retrieve_qa(request: DialogProfile):
    all_eng_question = translate(request.dialogs)
    embedding_vector = get_chat_emb(all_eng_question)
    result_qas, seqs = retrieval(embedding_vector, request.profiles)
    output_dict = {
        "ins": [result_qas[k]["ins"] for k in range(len(result_qas))],
        "answers": [result_qas[k]["responses"] for k in range(len(result_qas))],
        "seqs": seqs,
    }
    return Response(
        content=json.dumps(output_dict, ensure_ascii=False), status_code=200
    )


""" Stage3 refine回答 """


## 传入问题(str)，回答(str)，修改意见(str)
## 输出新回答(str)
class RefineRequest(BaseModel):
    ins: str = Field(..., alias="ins")
    ans: str = Field(..., alias="ans")
    suggestion: str = Field(..., alias="suggestion")


@router.post("/v1/refine")
async def chat(query: RefineRequest):
    prefix = """
        现在有一个问题和它对应的回答，一个用户对这个回答提出了一些修改意见。
        你需要根据修改意见对回答进行调整，直接输出修改后的回答即可，请不要输出任何多余信息。
        #### 输入格式：
        问题内容
        回答内容
        修改意见内容
    """
    input_prompt = """
        问题：{ins}
        回答：{ans}
        修改意见：{suggestion}
    """
    msg = input_prompt.format(ins=query.ins, ans=query.ans, suggestion=query.suggestion)
    return Response(
        content=json.dumps(
            {
                "result": gpt_chatcompletion(
                    form_messages(msg=msg, system_prompt=prefix)
                )
            },
            ensure_ascii=False,
        ),
        status_code=200,
    )
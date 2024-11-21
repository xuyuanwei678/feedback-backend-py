import asyncio
from typing import Any, Dict, List, Optional, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime
import aiomysql

T = TypeVar("T")


class MySQLAsyncDatabase:
    _instance = None
    pool: aiomysql.Pool = None

    @classmethod
    async def get_instance(cls):
        if not cls._instance:

            cls.pool = await aiomysql.create_pool(
                host="localhost",
                port=3306,
                user="xuyuanwei",
                password="5yxmB3vQL98SIe2UYsjdfQ==",
                db="feedback",
                autocommit=True,
            )
        return cls

    async def close_pool(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()


class ORMBaseModel(BaseModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.now)
    deleted: int = Field(default=0)

    __tablename__ = None

    class Config:
        orm_mode = True

    @classmethod
    async def get(cls, obj_id: int) -> Optional[Dict[str, Any]]:
        db = await MySQLAsyncDatabase.get_instance()
        async with db.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    f"SELECT * FROM {cls.__tablename__} WHERE id = %s AND deleted = 0",
                    (obj_id,),
                )
                result = await cur.fetchone()
                if result:
                    return result
                return None

    @classmethod
    async def find(cls, filters: dict) -> List[dict]:
        db = await MySQLAsyncDatabase.get_instance()
        filter_clause = " AND ".join([f"{k} = %s" for k in filters.keys()])
        filter_values = list(filters.values())
        async with db.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    f"SELECT * FROM {cls.__tablename__} WHERE {filter_clause} AND deleted = 0",
                    filter_values,
                )
                results = await cur.fetchall()
                return results

    @classmethod
    async def find_one(cls, filters: dict = None) -> Dict:
        db = await MySQLAsyncDatabase.get_instance()
        filter_clause = " AND ".join([f"{k} = %s" for k in filters.keys()])
        filter_values = list(filters.values())
        async with db.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    f"SELECT * FROM {cls.__tablename__} WHERE {filter_clause} AND deleted = 0",
                    filter_values,
                )
                result = await cur.fetchone()
                if result:
                    return result
                return None

    async def insert(self) -> int:
        db = await MySQLAsyncDatabase.get_instance()
        fields = [f for f in self.model_dump() if f != "id" and f != "deleted"]
        values = [getattr(self, f) for f in fields]
        placeholders = ", ".join(["%s"] * len(values))
        field_names = ", ".join(fields)
        async with db.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"INSERT INTO {self.__tablename__} ({field_names}) VALUES ({placeholders})",
                    values,
                )
                await conn.commit()
                return cur.lastrowid

    @classmethod
    async def update(cls, filter: dict = None, update: dict = None) -> None:
        db = await MySQLAsyncDatabase.get_instance()
        filter_clause = " AND ".join([f"{k} = %s" for k in filter.keys()])
        filter_values = list(filter.values())
        update_clause = ", ".join([f"{k} = %s" for k in update.keys()])
        update_values = list(update.values())
        async with db.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE {cls.__tablename__} SET {update_clause} WHERE {filter_clause}",
                    update_values + filter_values,
                )
                await conn.commit()

    @classmethod
    async def delete(cls, id) -> None:
        db = await MySQLAsyncDatabase.get_instance()
        async with db.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"UPDATE {cls.__tablename__} SET deleted = 1 WHERE id = %s", (id,)
                )
                await conn.commit()


class SQLExecutor:
    @classmethod
    async def insert(cls, sql: str, params: Optional[tuple] = None) -> int:
        db = await MySQLAsyncDatabase.get_instance()
        async with db.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                await conn.commit()
                return cur.lastrowid

    @classmethod
    async def update(cls, sql: str, params: Optional[tuple] = None) -> None:
        db = await MySQLAsyncDatabase.get_instance()
        async with db.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                await conn.commit()

    @classmethod
    async def delete(cls, sql: str, params: Optional[tuple] = None) -> None:
        db = await MySQLAsyncDatabase.get_instance()
        async with db.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                await conn.commit()

    @classmethod
    async def select(cls, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        db = await MySQLAsyncDatabase.get_instance()
        async with db.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(sql, params)
                result = await cur.fetchall()
                return result
            
    @classmethod
    async def select_one(cls, sql: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        db = await MySQLAsyncDatabase.get_instance()
        async with db.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(sql, params)
                result = await cur.fetchone()
                return result
            
    @classmethod
    async def count(cls, sql: str, params: Optional[tuple] = None) -> int:
        db = await MySQLAsyncDatabase.get_instance()
        async with db.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                result = await cur.fetchone()
                return result[0]

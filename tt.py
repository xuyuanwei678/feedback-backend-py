import threading
import jwt
from datetime import datetime, timedelta
from redis import Redis

class CacheUtil:
    def __init__(self):
        # 连接到 Redis
        self.redis = Redis(
            host="localhost",
            port="6379",
            db="6",
            decode_responses=True,
        )
        self.lock = threading.Lock()

    def exists(self, key) -> bool:
        with self.lock:
            return self.redis.exists(key)

    def get(self, key):
        with self.lock:
            return self.redis.get(key)

    def set(self, key, value, ttl):
        with self.lock:
            self.redis.set(key, value, ex=ttl)

    def delete(self, key):
        with self.lock:
            self.redis.delete(key)

    def clear(self):
        with self.lock:
            self.redis.flushdb()

    def all(self):
        with self.lock:
            return self.redis.keys()


tokenCache = CacheUtil()

class TokenUtil:

    expire_time = 60 * 60 * 24 * 30  # 7天

    salt = "thunlpFIT4506"

    @staticmethod
    def generate_token(user_id: str):
        """
        生成token
        """
        token = jwt.encode(
            {
                "user_id": user_id,
                "exp": datetime.now()
                - timedelta(hours=8)
                + timedelta(seconds=TokenUtil.expire_time),
            },
            key=TokenUtil.salt,
            algorithm="HS256",
            headers={"alg": "HS256", "typ": "JWT"},
        )

        tokenCache.set(f"{user_id}", token, TokenUtil.expire_time)

        return token

    @staticmethod
    def delete_token(user_id: str):
        """
        删除token
        """
        tokenCache.delete(f"{user_id}")
        return {"code": 200, "message": "success"}

    @staticmethod
    def token_required(func):
        """
        token验证
        """

        def wrapper(*args, **kwargs):
            token = args[0].headers.get("Authorization")
            if not token:
                return {"code": 401, "message": "token is required"}
            token = token.split(" ")[-1]
            try:
                jwt.decode(
                    token,
                    key=TokenUtil.salt,
                    algorithms=["HS256"],
                    options={"verify_exp": True},
                )
            except jwt.ExpiredSignatureError:
                return {"code": 401, "message": "token is expired"}
            except jwt.InvalidTokenError:
                return {"code": 401, "message": "token is invalid"}
            return func(*args, **kwargs)

        return wrapper

    # token 功能暂不生效
    @staticmethod
    def token_identify(token: str, user_id: str):
        try:
            payload = jwt.decode(
                token,
                key=TokenUtil.salt,
                algorithms=["HS256"],
                options={"verify_exp": True},
            )
            token_user_id = payload.get("user_id")
            return token_user_id == user_id
        except jwt.ExpiredSignatureError:
            print("code: 401  message: token is expired")
        except jwt.InvalidTokenError:
            print("code: 401  message: token is invalid")
        return False

    # 禁止多地同时登陆
    @staticmethod
    def token_if_equals(token: str):
        try:
            payload = jwt.decode(
                token,
                key=TokenUtil.salt,
                algorithms=["HS256"],
                options={"verify_exp": True},
            )
            token_user_id = payload.get("user_id")
            if token == tokenCache.get(f"{token_user_id}"):
                return True
            else:
                return False
        except jwt.ExpiredSignatureError:
            print("code: 401  message: token is expired")
            raise jwt.ExpiredSignatureError("message: token is expired")
        except jwt.InvalidTokenError:
            print("code: 401  message: token is invalid")
            raise jwt.InvalidTokenError("message: token is invalid")




from datetime import datetime


class Util:

    @staticmethod
    def to_simple_dict(data: dict, include=None, exclude=None):
        """
        将对象转换为字典

        :param data: 对象
        """

        if not data:
            return {}

        if include:
            data = {key: data[key] for key in include if key in data}

        if exclude:
            data = {key: data[key] for key in data if key not in exclude}

        # 序列化datetime
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.strftime("%Y-%m-%d %H:%M:%S")

        return data
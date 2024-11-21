### Conda 环境
```shell
conda activate feedback-env
```

### 启动服务
```shell
python app.py

# debugger
vscode 执行FastAPI 即可
```

### SQL ORM 使用方法

1. ORM 查询

```python
user = await User.find_one(filters={"email": ""})
```

2. SQL 查询
```python
# params 是一个元组，元组中的元素是查询条件的值， 且元组中的元素的顺序要和sql中的占位符一一对应
user = await SQLExecutor.select_one(sql="select * from users where email = %s", params=("",))
```
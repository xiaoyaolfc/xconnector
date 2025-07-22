# 示例 Dockerfile（适配你的目录结构）
FROM python:3.12-slim
WORKDIR /app

# 复制依赖文件（假设 requirements.txt 在 xconnector 根目录）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码（xconnector 目录下的所有内容）
COPY xconnector/ ./xconnector/

# 复制其他必要文件（如 csrc/ 等）
COPY csrc/ ./csrc/

# 启动命令（根据实际入口文件调整）
CMD ["python", "xconnector/main.py"]
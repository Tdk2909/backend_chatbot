
"""
SERVER LANGCHAIN VỚI AGENT VÀ LỊCH SỬ CHAT (THỬ NGHIỆM)
"""

import sys
import os

# Thêm đường dẫn đến thư mục chứa agent.py vào sys.path
current_dir = os.path.dirname('C:\\Users\\PC\\Downloads\\Long 4_7_2024\\AI_lord\\app')
agent_module_dir = os.path.join(current_dir, 'app')  # Thay đổi 'app' thành thư mục chứa agent.py nếu khác
sys.path.append(agent_module_dir)

from agent import chain_with_history_and_agent  # Thay đổi 'app' thành thư mục chứa agent.py nếu khác

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)

# Allow requests from all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


add_routes(
    app,
    chain_with_history_and_agent,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from api.api import api_router

# Document에 출력되는 title과 description, version 표시
app = FastAPI(
    title = "Vector Search Engine",
    description = "search most similar fashion item using Image-Retrieval Search",
    version="0.5.0",
)

# CORS Middleware 설정
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 라우터 실행
app.include_router(api_router)
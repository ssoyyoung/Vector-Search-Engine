from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from api.api import api_router

app = FastAPI(
    title = "Vector Search Engine",
    description = "search most similar fashion item using Image-Retrieval Search",
    version="0.5.0",
)

app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router)
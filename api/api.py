from fastapi import APIRouter
from api.endpoint import create, search

api_router = APIRouter()

api_router.include_router(search.router, prefix="/search_vec", tags=["search"])
api_router.include_router(create.router, prefix="/create_vec", tags=["create"])
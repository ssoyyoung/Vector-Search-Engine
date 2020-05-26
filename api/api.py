from fastapi import APIRouter
from api.endpoint import create, search

# APIRouter 객체 실행
api_router = APIRouter()

# prefix를 설정하여 하위 라우터 생성
api_router.include_router(search.router, prefix="/search_vec", tags=["search"])
api_router.include_router(create.router, prefix="/create_vec", tags=["create"])
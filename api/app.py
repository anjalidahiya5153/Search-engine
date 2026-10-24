# fastapi application entrypoint
from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Distributed Engine Search", version="1.0.0")
app.include_router(router, prefix="/api/v1")
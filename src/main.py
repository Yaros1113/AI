
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.services.ml_service import MLService, ModelConfig
from src.api.routes import router as api_router
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальная переменная для сервиса
ml_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: загружаем модель
    global ml_service
    try:
        model_config = ModelConfig()
        ml_service = MLService(model_config)
        logger.info("✅ ML Service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML Service: {str(e)}")
        raise
    
    yield
    
    # Shutdown: очищаем ресурсы
    ml_service = None
    logger.info("🚪 Application shutdown complete")

app = FastAPI(
    lifespan=lifespan,
    title="T-Bank Logo Detector API",
    description="REST API for detecting T-Bank logos in images",
    version="1.0.0"
)

# Включаем роутер
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "T-Bank Logo Detector API is running!",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

@app.get("/info")
async def info():
    """Информация о текущей конфигурации"""
    return {
        "device": os.getenv("DEVICE", "cuda"),
        "model_path": os.getenv("MODEL_PATH", "not set"),
        "python_version": os.getenv("PYTHON_VERSION", "3.13")
    }
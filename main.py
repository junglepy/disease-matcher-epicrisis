import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.models import EpicrisisQuery, UnifiedMatchResponse, HealthResponse
from src.config import get_settings
from src.epicrisis_orchestrator import EpicrisisOrchestrator

# Настройка логирования
temp_settings = get_settings()
logging.basicConfig(
    level=getattr(logging, temp_settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальные объекты
settings = None
orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global settings, orchestrator
    
    # Startup
    logger.info("Starting Epicrisis Disease Matcher API...")
    
    try:
        # Загружаем настройки
        settings = get_settings()
        
        # Инициализируем orchestrator
        orchestrator = EpicrisisOrchestrator(settings)
        
        logger.info("Epicrisis Disease Matcher API is ready!")
        
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        raise
        
    yield
    
    # Shutdown
    logger.info("Shutting down Epicrisis Disease Matcher API...")


# Создание FastAPI приложения
app = FastAPI(
    title="Epicrisis Disease Matcher API",
    description="Система анализа эпикризов и сопоставления с заболеваниями через HPO",
    version="1.0.0",
    lifespan=lifespan
)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Epicrisis Disease Matcher API",
        "version": "1.0.0",
        "endpoints": {
            "match_epicrisis": "/api/v1/match_epicrisis",
            "health": "/api/v1/health"
        },
        "features": [
            "Russian text epicrisis analysis",
            "Symptom extraction with LLM",
            "HPO term matching",
            "LIRICAL-based disease ranking",
            "Unified response format"
        ]
    }


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния сервиса"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    components = {
        "hpo_loader": bool(orchestrator.hpo_loader and orchestrator.hpo_loader.hpo_terms),
        "symptom_extractor": bool(orchestrator.symptom_extractor),
        "hpo_matcher": bool(orchestrator.hpo_matcher and orchestrator.hpo_matcher.bm25),
        "hpo_selector": bool(orchestrator.hpo_selector),
        "lirical_wrapper": bool(orchestrator.lirical_wrapper),
        "orchestrator": True
    }
    
    # Проверка LIRICAL - пробуем сделать запрос к сервису
    try:
        import httpx
        response = httpx.get(f"{orchestrator.lirical_wrapper.lirical_url}/health", timeout=2.0)
        lirical_status = "available" if response.status_code == 200 else "unavailable"
    except:
        lirical_status = "unavailable"
    
    return HealthResponse(
        status="healthy" if all(components.values()) else "degraded",
        version="1.0.0",
        components=components,
        lirical_status=lirical_status
    )


@app.post("/api/v1/match_epicrisis", response_model=UnifiedMatchResponse)
async def match_epicrisis(query: EpicrisisQuery):
    """Анализ эпикриза и поиск заболеваний"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    logger.info(f"Processing epicrisis: {len(query.epicrisis_text)} characters")
    
    try:
        # Вызываем orchestrator для обработки
        response = await orchestrator.process_epicrisis(query)
        return response
        
    except Exception as e:
        logger.error(f"Error processing epicrisis: {e}", exc_info=True)
        
        from src.models import UnifiedMetadata
        
        return UnifiedMatchResponse(
            results=[],
            metadata=UnifiedMetadata(
                processing_time_ms=0,
                architecture="epicrisis_lirical",
                model=settings.openai_model if settings else "gpt-4-turbo-preview",
                total_results=0
            ),
            extended=None,
            error={
                "code": "PROCESSING_ERROR",
                "message": str(e)
            }
        )


if __name__ == "__main__":
    # Получаем настройки для определения порта
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )
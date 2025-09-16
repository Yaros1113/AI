from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from typing import List
import logging
from PIL import UnidentifiedImageError

from src.models.schemas import DetectionResponse, Detection, BoundingBox, ErrorResponse
from src.services.ml_service import MLService, ModelConfig

router = APIRouter()
logger = logging.getLogger(__name__)

# Создаем конфигурацию
model_config = ModelConfig()

# Инициализируем MLService
ml_service = MLService(model_config)

@router.post(
    "/detect",
    response_model=DetectionResponse,
    responses={
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Detect T-Bank Logo",
    description="""
    This endpoint accepts an image file and returns bounding boxes
    of all detected T-Bank logos. Ignores Tinkoff logos.
    **Supported formats:** JPEG, PNG, BMP, WEBP.
    """
)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении
    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)
    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    # 1. Валидация формата файла
    allowed_content_types = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp']
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format. Supported: {', '.join(allowed_content_types)}"
        )

    try:
        # 2. Чтение файла
        image_bytes = await file.read()
        
        # 3. Дополнительная валидация, что это действительно изображение
        try:
            from src.services.image_service import validate_image
            validate_image(image_bytes)
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))

        # 4. Вызов ML-сервиса для предсказания
        predictions = await ml_service.predict(image_bytes)

        # 5. Форматирование ответа согласно контракту
        detections_list = []
        for pred in predictions:
            bbox = BoundingBox(
                x_min=pred["x_min"],
                y_min=pred["y_min"],
                x_max=pred["x_max"],
                y_max=pred["y_max"]
            )
            detections_list.append(Detection(bbox=bbox))

        return DetectionResponse(detections=detections_list)

    except UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Invalid image file. Cannot identify image format.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during image processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during image processing.")

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model_loaded": ml_service.model is not None,
        "device": model_config.device,
        "confidence_threshold": model_config.confidence_threshold
    }
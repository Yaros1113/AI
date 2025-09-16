from ultralytics import YOLO
import logging
from pydantic_settings import BaseSettings
from src.services.image_service import process_image

logger = logging.getLogger(__name__)

class ModelConfig(BaseSettings):
    """Конфигурация модели"""
    model_path: str = '/app/data/models/yolo11m_tbank_best.pt'
    device: str = 'cuda'
    confidence_threshold: float = 0.5

    class Config:
        env_file = ".env"
        env_prefix = "MODEL_"

class MLService:
    def __init__(self, config: ModelConfig):
        """
        Загружает обученную модель YOLO для детекции логотипов.
        Args:
            config (ModelConfig): Конфигурация модели
        """
        self.config = config
        self.model = YOLO(config.model_path)
        self.model.to(config.device)
        logger.info(f"Model loaded from {config.model_path} on device {config.device}")
        logger.info(f"Confidence threshold: {config.confidence_threshold}")

    async def predict(self, image_bytes: bytes) -> list:
        """
        Предсказание bbox для загруженного изображения.
        Args:
            image_bytes (bytes): Изображение в виде байтов.
        Returns:
            list: Список словарей с координатами и уверенностью для каждого обнаруженного логотипа.
        """
        try:
            # Конвертируем bytes в PIL Image -> numpy array
            pil_image = await process_image(image_bytes)

            # Инференс с порогом из конфига
            results = self.model(pil_image, conf=self.config.confidence_threshold, verbose=False)

            # Пост-обработка результатов
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Извлекаем координаты, уверенность и класс
                        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = box.conf[0].cpu().numpy().item()
                        # class_id = int(box.cls[0].cpu().numpy()) # Если классов несколько

                        # Для вашей задачи class_id всегда должен быть 0 (tbank_logo)
                        detections.append({
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max,
                            "confidence": confidence
                        })
            return detections
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
from ultralytics import YOLO
import numpy as np
import logging
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Загружает обученную модель YOLO для детекции логотипов.
        Args:
            model_path (str): Путь к файлу .pt обученной модели (например, 'best.pt')
            device (str): Устройство для вычислений ('cuda' или 'cpu')
        """
        self.model = YOLO(model_path)
        self.model.to(device)
        logger.info(f"Model loaded from {model_path} on device {device}")

    async def predict(self, image_bytes: bytes) -> list:
        """
        Предсказание bbox для загруженного изображения.
        Args:
            image_bytes (bytes): Изображение в виде байтов.
        Returns:
            list: Список словарей с координатами и уверенностью для каждого обнаруженного логотипа.
        """
        # Конвертируем bytes в PIL Image -> numpy array
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)

        # Инференс! Указываем conf=0.25 для отсечения слабых срабатываний:cite[8]
        results = self.model(image_np, conf=0.25, verbose=False)

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
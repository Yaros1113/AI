# Используем официальный образ Python 3.13
FROM python:3.13-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt сначала для кэширования зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --upgrade pip
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаем необходимые директории
RUN mkdir -p data/models data/input data/output

# Копируем модель
COPY data/models/yolo11m_tbank_best.pt /app/data/models/

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/data/models/yolo11m_tbank_best.pt
ENV PORT=8080
ENV HOST=0.0.0.0
ENV DEVICE=cuda
ENV MODEL_CONFIDENCE_THRESHOLD=0.5

# Открываем порт
EXPOSE 8080

# Запускаем приложение (без --reload для production)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
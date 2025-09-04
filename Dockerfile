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
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаем необходимые директории
RUN mkdir -p data/models data/input data/output

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/data/models/yolov8n_tbank.pt
ENV PORT=8000
ENV HOST=0.0.0.0
ENV DEVICE=cuda

# Открываем порт
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
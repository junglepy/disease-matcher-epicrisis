# Simple Python image for epicrisis service
FROM python:3.9-slim

WORKDIR /app

# Копируем файлы requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы приложения
COPY . .

# Создаем директории для логов
RUN mkdir -p logs

# Порт
EXPOSE 8004

# Запуск
CMD ["python", "main.py"]
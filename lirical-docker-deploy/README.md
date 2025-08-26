# LIRICAL Service Docker Deployment

Минимальная конфигурация для запуска LIRICAL сервиса в Docker.

## Быстрый запуск

```bash
# Запуск с автоматической сборкой
docker-compose up --build

# Или ручная сборка и запуск
docker build -t lirical-deploy:latest .
docker run --rm -p 8083:8083 lirical-deploy:latest
```

## Проверка работы

```bash
# Проверка здоровья сервиса
curl http://localhost:8083/health

# Тестовый запрос
curl -X POST http://localhost:8083/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"hpo_codes": ["HP:0001263", "HP:0001250"], "max_results": 5}'
```

## Структура проекта

```
lirical-docker-deploy/
├── app.py                    # Flask приложение
├── lirical_wrapper.py        # обертка для LIRICAL
├── phenopacket_builder.py    # создание phenopacket
├── metrics_collector.py      # сбор метрик
├── lirical-cli-2.2.0.jar     # LIRICAL CLI
├── lib/                      # Java библиотеки
├── data/                     # HPO онтология и данные
├── requirements.txt          # Python зависимости
├── Dockerfile               # конфигурация образа
├── docker-compose.yml       # оркестрация
└── README.md               # эта инструкция
```

## Характеристики

- **Версия LIRICAL**: 2.2.0
- **Java**: OpenJDK 17
- **Python**: 3.9+
- **Порт**: 8083
- **Размер образа**: ~2.2GB
- **Память**: 2-4GB RAM

## Команды

```bash
# Сборка образа
docker build -t lirical-deploy:latest .

# Запуск контейнера
docker run --rm -p 8083:8083 lirical-deploy:latest

# Запуск в фоне
docker run -d -p 8083:8083 --name lirical lirical-deploy:latest

# Просмотр логов
docker logs lirical

# Остановка
docker stop lirical

# Использование docker-compose
docker-compose up --build -d
docker-compose logs -f
docker-compose down
```
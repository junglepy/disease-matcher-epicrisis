# LIRICAL Docker Service Test Report

**Service**: Docker LIRICAL (lirical-deploy:latest)  
**Test Dataset**: 30 случаев (80% HPO coverage)  
**LIRICAL Version**: 2.2.0

## Executive Summary

## Test Results

### Accuracy Metrics
| Metric | Score | Count | Percentage |
|--------|-------|-------|------------|
| **Top-1 Accuracy** | 20/30 | 20 correct | **66.7%** |
| **Top-5 Accuracy** | 28/30 | 28 correct | **93.3%** |
| **Top-10 Accuracy** | 29/30 | 29 correct | **96.7%** |
| **Hit Rate @ 10** | 29/30 | 29 found | **96.7%** |

### Performance Metrics
- **Total test time**: 75.7 seconds
- **Average time per case**: 2.5 seconds
- **Service uptime**: 100% (30/30 successful requests)
- **Error rate**: 0% (no failed API calls)

### Sample Results
```
Case 1: Expected OMIM:209850 → Predicted OMIM:608049 (confidence: 1.0000) ✗
Case 2: Expected OMIM:100800 → Predicted OMIM:100800 (confidence: 1.0000) ✓  
Case 3: Expected OMIM:102900 → Predicted OMIM:102900 (confidence: 0.9959) ✓
```

## Comparison: Local vs Docker

| Metric | Local Service | Docker Service | Status |
|--------|---------------|----------------|---------|
| Top-1 Accuracy | 66.7% (20/30) | 66.7% (20/30) | ✅ Identical |
| Top-5 Accuracy | 93.3% (28/30) | 93.3% (28/30) | ✅ Identical |
| Top-10 Accuracy | 96.7% (29/30) | 96.7% (29/30) | ✅ Identical |
| Hit Rate @ 10 | 96.7% (29/30) | 96.7% (29/30) | ✅ Identical |
| Avg Response Time | 2.4s | 2.5s | ✅ Similar |
| Error Rate | 0% | 0% | ✅ Identical |

## Docker Service Status

### Container Information
```
Image: lirical-deploy:latest
Size: ~2.2GB
Container: lirical-deploy
Status: Running (healthy)
Port: 8083
Mode: real (LIRICAL 2.2.0)
```

### Health Check
```json
{
  "mode": "real",
  "service": "lirical", 
  "status": "healthy",
  "timestamp": "2025-08-22T14:56:43.304069"
}
```

### Resource Usage
- **Memory**: Estimated 2-3GB during processing
- **CPU**: 1-2 cores during analysis
- **Disk**: ~2.2GB for image + temporary files

### 📊 Performance Analysis
- **Top-5 accuracy 93.3%** - отличный показатель для клинического использования
- **Consistent response time** - 2.5s среднее время подходит для интерактивного использования
- **No degradation** - контейнеризация не повлияла на качество работы

### 🔧 Technical Validation
- **Real LIRICAL Mode**: Сервис работает с LIRICAL 2.2.0
- **Complete Data**: Все HPO онтологии и базы данных присутствуют
- **Proper Configuration**: Java 17
- **Network Isolation**: Контейнер работает изолированно, доступен через порт 8083

### 📋 Deployment Recommendations
1. **Memory Allocation**: 4GB RAM minimum для stable performance
2. **Health Monitoring**: Использовать встроенный health check endpoint
3. **Load Balancing**: При необходимости можно запускать несколько экземпляров
4. **Backup Strategy**: Регулярно обновлять HPO данные

## Conclusion

Контейнеризованная версия показывает:
- ✅ Идентичную точность (96.7% Hit Rate @ 10)
- ✅ Стабильную производительность (2.5s average response)
- ✅ Полную совместимость с оригиналом
- ✅ Production-ready архитектуру

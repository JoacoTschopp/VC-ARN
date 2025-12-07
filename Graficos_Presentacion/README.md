# Graficos_Presentacion

Tablero liviano para explorar métricas de entrenamiento y probar el mejor modelo con muestras propias.

## Requisitos

1. **Python 3.10+** (ya presente en el proyecto).
2. **Entorno virtual**: activar siempre `source .venv/bin/activate` antes de correr cualquier comando Python.
3. **Dependencias opcionales para inferencia**: `fastapi`, `uvicorn[standard]`, `pydantic`, `torchvision` (solicitar instalación si no están disponibles).

## Cómo levantar el tablero de métricas

1. Activá el entorno virtual:
   ```bash
   source .venv/bin/activate
   ```
2. Ubicate en `Graficos_Presentacion/`.
3. Levantá un servidor HTTP simple:
   ```bash
   python -m http.server 8001
   ```
4. Abrí [http://localhost:8001/index.html](http://localhost:8001/index.html) en tu navegador.
5. El tablero muestra:
   - Resumen general de los experimentos a partir de `experiments_log.jsonl`.
   - Curvas de ganancia y pérdida con anotaciones dinámicas.
   - Leyendas ordenadas por desempeño.

## Backend de clasificación (FastAPI)

1. Activá el entorno virtual.
2. Ubicate en `Graficos_Presentacion/`.
3. Asegurate de tener `best_model.pth` en esta carpeta.
4. Levantá la API:
   ```bash
   uvicorn predict_api:app --reload --port 8002
   ```
   - La API expone `GET /health` y `POST /classify` (imagen en base64, devuelve clase/emoji/confianza).
   - La arquitectura cargada corresponde a la ResNet `[9, 9, 9]` utilizada en entrenamiento.

## Probar el mejor modelo con imágenes propias

1. Con el servidor anterior corriendo, visitá [http://localhost:8001/ejemplos.html](http://localhost:8001/ejemplos.html).
2. Subí una imagen `.png` o `.jpg` usando el selector provisto.
3. Presioná **“Clasificar”** para enviar el canvas (32×32) a `http://localhost:8002/classify`.
4. Se mostrará:
   - Vista previa original y canvas escalado 300%.
   - Resultado con nombre de clase CIFAR-10, emoji asociado y confianza.

> **Nota:** si el backend no está levantado en el puerto `8002`, la sección de resultado mostrará un mensaje de error.

## Estructura relevante

```
Graficos_Presentacion/
├── index.html        # Tablero principal con las series y métricas
├── ejemplos.html     # Página de prueba + llamada Fetch al backend
├── app.js            # Lógica de lectura de logs y Plot.js
├── predict_api.py    # Servicio FastAPI que carga best_model.pth
├── experiments_log.jsonl
└── README.md
```

## Consejos

- Para actualizar los gráficos, reemplazá/actualizá `experiments_log.jsonl` y refrescá la página.
- Si necesitás realizar cambios en los scripts, recordá reiniciar el servidor (`Ctrl+C`) y volver a ejecutar `python -m http.server 8001` tras guardar.

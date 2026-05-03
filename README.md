# Handwritten-text-recognition-using-VLM

## Архитектура
whiteboard_ocr/
│
├── pipeline/
│   ├── pipeline.py        # ← главный оркестратор, базовый класс приложения
│   ├── preprocessing.py   # нормализация гистограммы, исправление перспективных искажений
│   ├── detection.py       # обнаружение регионов
│   ├── recognition.py     # VLM API
│   ├── postprocessing.py
│
├── configs/
│   └── config.yaml
│
├── ui/
│   └── streamlit_app.py
│
├── cli/
│   └── run_ocr.py
│
└── utils/
    ├── visualization.py   # bbox + overlay
    ├── latex.py           # проверка/рендер


## Web
* Загрузка изображение/фото
* Предпросмотр результатов детекции регионов
* Переключение провайдеров/моделей
* Отображение bbox (детекции)
* Отображение распознанного LaTeX
* Сравнение “raw vs processed”

## Простой консольный интерфейс:
`python run_ocr.py file.jpg --config configs/config.yaml `

## Препроцессинг

* detect whiteboard region - SAM3, в перспективе можно дообучить модельку
* contrast enhancement (CLAHE or histogram equalization)
* binarize (adaptive threshold) - всё, что выше/ниже определённого порога, закрашивается белым/чёрным

* lighting normalization - устранение неравномерностей освещения
* perspective transform/dewarping (rectify)
* convert to grayscale - cv2.cvtColor(image, cv2.COLOR.BGR2GRAY)
* denoise - проходимся фильтром-свёрткой для сглаживания и удаления высокочастотных шумов


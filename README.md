# Handwritten-text-recognition-using-VLM

## Архитектура
whiteboard_ocr/ <br/>
│ <br/>
├── pipeline/ <br/>
│   ├── pipeline.py        # ← главный оркестратор, базовый класс приложения <br/>
│   ├── preprocessing.py   # нормализация гистограммы, исправление перспективных искажений <br/>
│   ├── detection.py       # обнаружение регионов <br/>
│   ├── recognition.py     # VLM API <br/>
│   ├── postprocessing.py <br/>
│ <br/>
├── configs/ <br/>
│   └── config.yaml <br/>
│ <br/>
├── ui/ <br/>
│   └── streamlit_app.py <br/>
│ <br/>
├── cli/ <br/>
│   └── run_ocr.py <br/>
│ <br/>
└── utils/ <br/>
    ├── visualization.py   # bbox + overlay <br/>
    ├── latex.py           # проверка/рендер <br/>


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


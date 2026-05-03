### Pipeline

Примерная архитектура
```python
class OCRPipeline:
    def __init__(self, config):
        self.config = config

    def run(self, image):
        if self.config.pipeline.use_detection:
            regions = detect(image)
        else:
            regions = [image]

        results = []

        for region in regions:
            if self.config.pipeline.use_preprocessing:
                region = preprocess(region)

            text = recognize(region)

            text = postprocess(text)

            results.append(text)

        return results
```
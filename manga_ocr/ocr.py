import time
from pathlib import Path

import jaconv
from PIL import Image
from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


class MangaOcr:
    def __init__(self, pretrained_model_name_or_path="kha-white/manga-ocr-base"):
        t0 = time.time()
        print(f"Loading from {pretrained_model_name_or_path}... ", end="")
        self.processor: ViTImageProcessor = ViTImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)

        # example_path = Path(__file__).parent / "assets/example.jpg"
        # if not example_path.is_file():
        #     raise FileNotFoundError(f"Missing example image {example_path}")
        # self(example_path)

        t1 = time.time()
        print(f"OCR ready in {t1 - t0:0.02f}s")

    def __call__(self, img_or_path):
        if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f"img_or_path must be a path or PIL.Image, instead got: {img_or_path}")

        img = img.convert("L").convert("RGB")

        x = self._preprocess(img)
        x = self.model.generate(x[None].to(self.model.device), max_length=300)[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        x = post_process(x)
        return x

    def _preprocess(self, img):
        pixel_values: ViTImageProcessor = self.processor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()


def post_process(text):
    text = "".join(text.split())
    text = jaconv.h2z(text, ascii=True, digit=True)

    text = text.replace("…", "")
    text = text.replace(":", "")
    text = text.replace("。", "")
    text = text.replace("・", "")
    # text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)

    return text

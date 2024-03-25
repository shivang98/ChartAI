from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import requests
from PIL import Image

processor = Pix2StructProcessor.from_pretrained('google/matcha-chart2text-pew')
model = Pix2StructForConditionalGeneration.from_pretrained('google/matcha-chart2text-pew')

# url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/20294671002019.png"
url = "https://scikit-learn.org/stable/_images/sphx_glr_plot_partial_dependence_006.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, text="partial dependency plots", return_tensors="pt")
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))

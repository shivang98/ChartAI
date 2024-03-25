import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

response = client.chat.completions.create(
  model="gpt-4-vision-preview", 
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Act as a expert data scientist and summarize what you see in this chart image within 2 sentences. One sentence to explain family of chart, next to explain what chat insights are observed?"},
        {
          "type": "image_url",
          "image_url": {
            # "url": "https://cdn1.byjus.com/wp-content/uploads/2018/11/maths/2016/06/03072323/Control-Charts.jpg",
              "url": "https://scikit-learn.org/stable/_images/sphx_glr_plot_partial_dependence_006.png",
          },
        },
      ],
    }
  ],
  max_tokens=500,
)

print(response.choices[0].message.content)

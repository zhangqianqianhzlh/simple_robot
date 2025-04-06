from openai import OpenAI
from PIL import Image
import io
import base64
import yaml
def convert_image_to_webp_base64(input_image_path):
    try:
        with Image.open(input_image_path) as img:
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='webp')
            byte_arr = byte_arr.getvalue()
            base64_str = base64.b64encode(byte_arr).decode('utf-8')
            return base64_str
    except IOError:
        print(f"Error: Unable to open or convert the image {input_image_path}")
        return None

if __name__ == "__main__":

    with open("llm.yaml", "r") as f:
        llm_config = yaml.safe_load(f)

    client = OpenAI(api_key=llm_config["SiliconCloud"]["api_key"], 
                    base_url=llm_config["SiliconCloud"]["base_url"])
    

    base64_image = convert_image_to_webp_base64("views/view_0_RotateLeft.png")
    
    response = client.chat.completions.create(
        # model='Pro/deepseek-ai/DeepSeek-R1',
        model="Qwen/Qwen2.5-VL-32B-Instruct",
        messages=[
            {
                "role": "user",
                "content":[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail":"low"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe the image in 20 words"
                    }
                ]
            }
        ],
        stream=True
    )


    for chunk in response:
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        if chunk.choices[0].delta.reasoning_content:
            print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import os

def main():
    print(os.getcwd())
    try: 
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        device = None
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(device)

        model.to(device)
        # url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        local_image_path = 'project/screenshots/outlook0.jpg'
        image = Image.open(local_image_path).convert("RGB")
        prompt = "who is sending the email in this image?"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(generated_text)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory error occurred.")
        else:
            print("An error occurred:", e)

if __name__ == '__main__':
    main()
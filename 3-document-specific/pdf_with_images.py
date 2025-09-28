import os
from unstructured.partition.pdf import partition_pdf
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
import os
from dotenv import load_dotenv
from PIL import Image
import base64
import io

load_dotenv()
llm = ChatOpenAI(model="gpt-4-vision-preview") #type:ignore
current_dir = os.path.dirname(__file__)
default_filename=os.path.join(current_dir,"this_pdf_has_images.pdf")


def chunk_pdf(filename:str=default_filename):
    chunks = partition_pdf(
        filename=filename,
        extract_images_in_pdf=True,
        infer_table_structure=True, 
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path="static/pdfImages/",
    )
    print(chunks)
    return chunks

def image_to_base64(image_path):
    with Image.open(image_path) as image:
        buffered = io.BytesIO()
        image.save(buffered, format=image.format)
        img_str = base64.b64encode(buffered.getvalue())
        return img_str.decode('utf-8')

def get_summary():
    image_str = image_to_base64("static/pdfImages/figure-15-6.jpg")

    chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024) # type:ignore

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text" : "Please give a summary of the image provided. Be descriptive"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_str}"
                        },
                    },
                ]
            )
        ]
    )

    print(msg.content)

if __name__ == '__main__':
    chunks = chunk_pdf()
    for chunk in chunks:
        print(chunk.metadata.text_as_html)
        print("\n\n---------------")
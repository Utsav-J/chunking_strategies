import os
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json
current_dir = os.path.dirname(__file__)
default_filename=os.path.join(current_dir,"this_pdf_has_tables.pdf")


def chunk_pdf(filename:str=default_filename):
    chunks = partition_pdf(
        filename=filename,
        strategy="hi_res", 
        infer_table_structure=True, 
        model_name="yolox"
    )
    print(chunks)
    return chunks

if __name__ == '__main__':
    chunks = chunk_pdf()
    for chunk in chunks:
        print(chunk.metadata.text_as_html)
        print("\n\n---------------")
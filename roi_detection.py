import zipfile
import os
from tqdm import tqdm
import json
from openai import OpenAI
import pandas as pd
import base64
from datetime import datetime
import shutil
import csv
from pathlib import Path

def unzip_files(input_files, output_folder):
    patients = []
    for input_file in input_files:
        extract_to = os.path.join(output_folder, input_file.split('/')[-1].split('.')[0])
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(input_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        patients.append(input_file.split('/')[-1].split('.')[0])
    return patients

def load_images(image_dir):
    images = os.listdir(image_dir)
    images = [os.path.join(image_dir, i) for i in images if '.jpg' in i]
    return images

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def is_roi(client, example_images_ROI, example_images_NOT_ROI, base64_image, model_name, max_tokens, temperature):
    
    encoded_example_images_ROI = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}}
        for path in example_images_ROI
    ]
    
    encoded_example_images_NOT_ROI = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(path)}"}}
        for path in example_images_NOT_ROI
    ]        
    response = client.chat.completions.create(
        model = model_name,
        messages=[
            {
                "role": "system",
                "content": """
                You are an expert pathologist’s assistant trained to support diagnostic workflows by identifying regions of interest (ROIs) in bone marrow histology images.
                Your role is to determine whether an image region is suitable for further diagnostic analysis, such as cell morphology assessment and lineage quantification.
                """,
            },
            {
                "role": "user",
                "content": [
                {
                "type": "text",
                "text": f"""
                Here are few example ROI images that are considered diagnostically relevant (ROI = Yes):
                """
                },
                *encoded_example_images_ROI,
                {
                "type": "text",
                "text": f"""
                Here are few example ROI images that are considered diagnostically NOT relevant (ROI = No):
                """
                },
                *encoded_example_images_NOT_ROI,                
                {
                    "type": "text",
                    "text": """A diagnostically useful tile (ROI = Yes) must meet **all** of the following criteria:
                    - It is sharply focused and clearly stained.
                    - It shows evenly distributed cellular content with distinguishable cells.
                    - It has adequate cellularity
                    - It avoids artifacts such as:
                        - Tissue folding or crushing
                        - Empty or white areas
                        - Necrotic, peripheral, or non-representative zones
                        - or dominated by dark "crumbly" debris-like regions
                
                    Typical cells that should be visible include:
                    - Erythroid precursors
                    - Myeloid cells
                    - Megakaryocytes (if present)
                
                    Tiles with predominantly fat, background, damaged tissue, or poor staining/focus should be rejected (ROI = No), even if a small amount of tissue is present.Your output should follow this exact JSON format:{{"Thoughts": "<your brief analysis and reasoning>", "ROI": "<Yes or No>"}}.
                    Now, evaluate the following new image:
                    """
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
                ]
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature)
    
    file_content = response.choices[0].message.content
    return file_content   

if __name__ == "__main__":
    
    
    unzipFiles = False
    MAIN_DIR = '/mnt/bulk-ganymede/narmin/narmin/AML_Project/roi_detection'
    IMG_DIR = '/mnt/bulk-ganymede/narmin/narmin/AML_Project/tiles'
    OUT_DIR = MAIN_DIR + '/output'
    os.makedirs(OUT_DIR, exist_ok = True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"run_{timestamp}"
    OUT_DIR = os.path.join(OUT_DIR, folder_name)
    os.makedirs(OUT_DIR, exist_ok = True)
    
    shutil.copy('/mnt/bulk-ganymede/narmin/narmin/AML_Project/AML_Project/roi_detection.py', os.path.join(OUT_DIR, 'roi_detection.txt'))

    example_images_ROI = ['/mnt/bulk-ganymede/narmin/narmin/AML_Project/tiles/AML_Box1_OT07/tile_(3072.5200400801677, 28676.8537074149).jpg',
                      '/mnt/bulk-ganymede/narmin/narmin/AML_Project/tiles/AML_Box1_OT07/tile_(3584.6067134268624, 29188.940380761593).jpg',
                      '/mnt/bulk-ganymede/narmin/narmin/AML_Project/tiles/AML_Box1_OT07/tile_(5120.866733466946, 14338.42685370745).jpg',
                      '/mnt/bulk-ganymede/narmin/narmin/AML_Project/tiles/AML_Box1_OT16/tile_(10753.820140280586, 40966.93386773557).jpg',
                      '/mnt/bulk-ganymede/narmin/narmin/AML_Project/tiles/AML_Box1_OT16/tile_(11777.993486973975, 39430.673847695485).jpg',
                      '/mnt/bulk-ganymede/narmin/narmin/AML_Project/tiles/AML_Box1_OT16/tile_(11777.993486973975, 39942.76052104218).jpg']
    
    example_images_NOT_ROI = ['/mnt/bulk-ganymede/narmin/narmin/AML_Project/roi_detection/output/run_2025-06-23_10-35-28/AML_Box1_OT16/ROIs/(4608.780060120252, 22531.813627254563).jpg',
                    '/mnt/bulk-ganymede/narmin/narmin/AML_Project/roi_detection/output/run_2025-06-23_10-35-28/AML_Box1_OT16/ROIs/(6657.12675350703, 20995.55360721448).jpg',
                    '/mnt/bulk-ganymede/narmin/narmin/AML_Project/roi_detection/output/run_2025-06-23_10-35-28/AML_Box1_OT16/ROIs/(8193.386773547114, 19971.38026052109).jpg',
                    '/mnt/bulk-ganymede/narmin/narmin/AML_Project/roi_detection/output/run_2025-06-23_10-35-28/AML_Box1_OT16/ROIs/(10241.733466933892, 37894.4138276554).jpg']
    
    with open('/mnt/bulk-ganymede/narmin/narmin/MSI_LLM/key.json', 'r') as f:
        config = json.load(f)
    
    client = OpenAI(api_key=config['Pluto'], base_url= 'http://192.168.33.27/v1')
    
    if unzipFiles:
        patients = unzip_files(input_files = ['/mnt/bulk-saturn/chiara/chiara/02_features/HAEMA/UNI/AML_Cache_Cytomorph_Proj/AML_Box1_OT02.56cfab1ca35970e42e8faaf702ecb1d0915f561ad56caf5f71488e670d85a6e5.zip',
                                '/mnt/bulk-saturn/chiara/chiara/02_features/HAEMA/UNI/AML_Cache_Cytomorph_Proj/AML_Box1_OT07.91f38e22c02cba09ca71626faead1cca0b8cc7c7a12cf752bd801384791d0dbb.zip', 
                                '/mnt/bulk-saturn/chiara/chiara/02_features/HAEMA/UNI/AML_Cache_Cytomorph_Proj/AML_Box1_OT16.a0412ad97200ac615ddfcf3aa9fe8ba4ffb625ac9a8948240494c5ba684ce0a2.zip'],
                    output_folder = IMG_DIR)
    else:
        patients = os.listdir(IMG_DIR)
    
    for patient in patients:
        images = load_images(os.path.join(IMG_DIR, patient))
        patient_output_folder = os.path.join(OUT_DIR, patient + '.json')
        roi_descriptions = []
        
        PATIENT_OUT_DIR = os.path.join(OUT_DIR, patient)
        os.makedirs(PATIENT_OUT_DIR, exist_ok = True)
        
        ROI_FOLDER = os.path.join(PATIENT_OUT_DIR, 'ROIs')
        os.makedirs(ROI_FOLDER, exist_ok = True)
        
        NOT_ROI_FOLDER = os.path.join(PATIENT_OUT_DIR, 'NOT_ROIs')
        os.makedirs(NOT_ROI_FOLDER, exist_ok = True)
    
        for image in tqdm(images):
            base64_image = encode_image(image)
            response_str = is_roi(client = client, base64_image = base64_image, example_images_ROI = example_images_ROI, example_images_NOT_ROI = example_images_NOT_ROI, max_tokens = 1000, temperature = 0.2, 
                                     model_name = 'Llama-4-Maverick-17B-128E-Instruct-FP8')
            try:
                response_json = json.loads(response_str)
                roi_descriptions.append({
                    "ROI": 'ROI_' + image.split('_')[-1].replace('.png', ''),
                    "Thoughts": response_json["Thoughts"],
                    "Is_ROI": response_json["ROI"]
                })
                if response_json["ROI"] == "Yes":
                    shutil.copy(image, os.path.join(ROI_FOLDER, image.split('_')[-1]))
                else:
                    shutil.copy(image, os.path.join(NOT_ROI_FOLDER, image.split('_')[-1]))
            except:
                print(image)
                roi_descriptions.append({
                    "ROI": 'ROI_' + image.split('_')[-1].replace('.png', ''),
                    "Error": response_str
                })
            
        with open(patient_output_folder, "w", encoding="utf-8") as f:
            json.dump(roi_descriptions, f, ensure_ascii=False, indent=2)
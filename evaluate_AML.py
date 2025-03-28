# import necessary packages
import os
from tqdm import tqdm
import json
from openai import OpenAI
import pandas as pd
import base64

# Function to get video names using multiple filters 
def get_patient_names(clini_table):
    data = pd.read_excel(clini_table)
    patients_temp = data['Pat_ROI']
    patients = list(set(["_".join(i.split('_')[0:3]) for i in patients_temp]))
    return patients

def load_images(image_dir, patient):
    images = os.listdir(image_dir)
    images = [i for i in images if patient in i]
    images = [os.path.join(image_dir, i) for i in images]
    return images

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
# main function
if __name__ == "__main__":
    
    MAIN_DIR = '/mnt/bulk-ganymede/narmin/narmin/AML_Project'
    IMG_DIR = '/mnt/bulk-saturn/chiara/chiara/03_WSI/ROIs_manuell_Lara/AML_Box_1'
    
    CLINI_TABLE_DIR = MAIN_DIR + '/Cytomorpholgy_results_WSI.xlsx'
    patients = get_patient_names(clini_table = CLINI_TABLE_DIR)

    OUT_DIR = MAIN_DIR + '/output'
    os.makedirs(OUT_DIR, exist_ok = True)
    
    DESCRIPTIONS_DIR = OUT_DIR + '/descriptions'
    os.makedirs(DESCRIPTIONS_DIR, exist_ok = True)
    
    DIAGNOSE_DIR = OUT_DIR + '/diagnose'
    os.makedirs(DIAGNOSE_DIR, exist_ok = True)
    
    with open('/mnt/bulk-ganymede/narmin/narmin/MSI_LLM/key.json', 'r') as f:
        config = json.load(f)
    
    client = OpenAI(api_key=config['Pluto'], base_url='http://pluto/v1-openai/')

    for patient in tqdm(patients):
        images = load_images(image_dir = IMG_DIR, patient = patient)
        description_file = os.path.join(DESCRIPTIONS_DIR, patient.split('/')[-1].replace('.png', '_ROIs.json'))
        if not os.path.exists(description_file):
            for image in images:
                base64_image = encode_image(image)
                response = client.chat.completions.create(
                    model='qwen2.5-vl-7b-instruct',  # Use the appropriate vision model
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a expert ... pathologists...",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Explaiin the morphology..."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=1000
                )
                file_content = response.choices[0].message.content
                entry = {
                        "ROI Id": image.split('/')[-1],
                        "Morphology Description": file_content
                        }
                with open(description_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
    
    descriptions = os.listdir(DESCRIPTIONS_DIR)   
    for description in descriptions:
        with open(os.path.join(DESCRIPTIONS_DIR, description), 'r') as f:
            data = f.readlines()
            data = [json.loads(i) for i in data]
            df = pd.DataFrame(data)
        all_morphology = "\n".join(df['Morphology Description'].head(10).tolist())
        
        response = client.chat.completions.create(
            model='qwen2.5-vl-7b-instruct',  # Use the appropriate vision model
            messages=[
                {
                    "role": "system",
                    "content": "You are a expert ... pathologists...",
                },
                {
                    "role": "user",
                    "content": [
                    {
                    "type": "text",
                    "text": f"Here are 10 morphological observations:\n\n{all_morphology}\n\nPlease provide a ... Befund based on this histopatholgoy descriptions."
                    },
                    ]
                }
            ],
            max_tokens=1000
        )
        diagnose_file = os.path.join(DIAGNOSE_DIR, description)
        diagnose = response.choices[0].message.content
        with open(diagnose_file, "a") as f:
            f.write(json.dumps(entry) + "\n")



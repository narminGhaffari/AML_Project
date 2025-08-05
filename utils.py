import os
import random
import pandas as pd
random.seed(23)
# Generate all the patient names
rois = os.listdir('/mnt/bulk-saturn/chiara/chiara/03_WSI/ROIs_manuell_Lara/AML_Box_1')
rois = [i for i in rois if 'AML_' in i]
print(len(rois))

patients = ['_'.join(i.split('_')[0:3]) for i in rois]
patients = list(set(patients))
print(len(patients))


for patient in patients:
    slides = [i for i in rois if patient+'_' in i]
    if not len(slides) == 10:
        patients.remove(patient)
        print(f'Patient {patient} has {len(slides)} slides, expected 10.')
        
print(len(patients))

df = pd.DataFrame(patients, columns=['Patients'])
#df.to_csv('/mnt/bulk-ganymede/narmin/narmin/AML_Project/tables/all_patients.csv', index=False)
random_selection = df.sample(n=50, random_state=23)
#random_selection.to_csv('/mnt/bulk-ganymede/narmin/narmin/AML_Project/tables/random_50_patients.csv', index=False)


import yt_dlp

def download_audio(youtube_url, output_path="downloaded_audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'noplaylist': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

# Example usage
download_audio("https://www.youtube.com/watch?v=VETpylOwwWI&ab_channel=ARIATVCanada%D8%AA%D9%84%D9%88%DB%8C%D8%B2%DB%8C%D9%88%D9%86%D8%A2%D8%B1%DB%8C%D8%A7")
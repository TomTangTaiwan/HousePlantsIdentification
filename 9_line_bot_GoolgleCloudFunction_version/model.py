import json
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import os

from google.cloud import storage

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


class GreenFinger(object):
    def __init__(self):
        label_dict = {
            'Aglaonema Commutatum (萬年青)': 0,
            'Asplenium Nidus (鳥巢蕨)': 1,
            'Calathea Orbifolia (竹芋)': 2,
            'Chamaedorea Elegans (袖珍椰子)': 3,
            'Dracaena Reflexa (百合竹)': 4,
            'Dracaena Trifasciata (虎尾蘭)': 5,
            'Dypsis Lutescens (散尾葵)': 6,
            'Epipremnum Aureum (黃金葛)': 7,
            'Hoya Carnosa (球蘭)': 8,
            'Maranta Leuconeura (豹紋竹芋)': 9,
            'Monstera Deliciosa (龜背芋)': 10,
            'Nephrolepis Cordifolia (腎蕨)': 11,
            'Pachira Aquatica (馬拉巴栗)': 12,
            'Peperomia Argyreia (西瓜皮椒草)': 13,
            'Peperomia Obtusifolia (圓葉椒草)': 14,
            'Philodendron Gloriosum (錦緞蔓綠絨)': 15,
            'Rhapis Excelsa (棕竹)': 16,
            'Schefflera Arboricola (鵝掌藤)': 17,
            'Tradescantia Zebrina (吊竹梅)': 18,
            'Zamioculcas (金錢樹)': 19
        }
        self.label_list = [k for k in label_dict]
        print(self.label_list)
        self.n_class = len(self.label_list)
        print(self.n_class)
        self.download_model_file()
        print(f'Model file downloaded')
        self.init_model()
        print(f'Model initialized')

    def download_model_file(self):
        # Access Cloud Storage
        bucket_name = 'green_finger_2022'
        project_id = 'fourth-amp-335116'
        model_file = 'epoch_10_2022-08-30 10_38_20.984264.pt'

        # Initialise a client
        print(f'Initialise a client')
        client = storage.Client(project_id)
            
        # Create a bucket object for our bucket
        print(f'Create a bucket object for our bucket')
        bucket = client.get_bucket(bucket_name)
            
        # Create a blob object from the filepath
        print(f'Create a blob object from the filepath')
        blob = bucket.blob(model_file)

        folder = '/tmp/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Download the file to a destination
        blob.download_to_filename(folder + "local_model.pt")

    def init_model(self):

        print(f'Activate model')

        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, self.n_class)
        model.load_state_dict(torch.load('/tmp/local_model.pt', map_location="cpu")["model_state_dict"])
        model.eval()
        device = torch.device("cpu")
        self.model = model.to(device)

        print(f'Model Activated')

    def predict(self, img):
        
        self.img = img

        print(f'Activate transform')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
            256,
            interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f'Transform activated')

        print(f'Send in image to transform')

        img_trans = transform(self.img).unsqueeze(0)

        print(f'Image transform completed')

        print(f'Start prediction')

        pred = None
        with torch.no_grad():
            output = self.model(img_trans)
            prob = output.cpu().numpy()
            prob = np.exp(prob)
            self.prob = prob / np.sum(prob, axis=1, keepdims=True)
            
        print(f'Prediction ends')

        self.reply_text()

    def reply_text(self):
        idxs = np.argsort(self.prob)
        idx1 = int(idxs[0, -1])
        val1 = float(self.prob[0, idx1])
        if val1 < 0.5:
            self.txt = f'人家不太有把握耶，要不要再重拍一張🤔'
        else:
            self.txt = '{} ({:.1f}%)'.format(self.label_list[idx1], 100 * val1)

            if val1 < 0.9:
                idx2 = int(idxs[0, -2])
                val2 = float(self.prob[0, idx2])

                self.txt = self.txt + '\n{} ({:.1f}%)'.format(self.label_list[idx2], 100 * val2)
        
        print(self.txt)

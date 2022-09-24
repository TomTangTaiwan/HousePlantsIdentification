from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    ImageMessage, VideoMessage, AudioMessage
)

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

from model import GreenFinger

model = GreenFinger()

# Channel Access Token
CHANNEL_ACCESS_TOKEN = 'JYiu2547kK14k2qjEipagFX6sFp5l3seJQ6gAoKJnUDMhifZQRZy8F7s5WWNwKMU4uFjh+y5T2sPxdHHEkVkpmqesL8QuY9DCIOKpQrddisvb6Ala7CcNwA/E01/PtSyHT3sBcxe8yVRpSHIi+2L1QdB04t89/1O/w1cDnyilFU='
# Channel Secret
CHANNEL_SECRET = '78d674217e732598fe253489cf82bc5b' 

def linebot(request):
    body = request.get_data(as_text=True)
    try:
        access_token = CHANNEL_ACCESS_TOKEN
        secret = CHANNEL_SECRET
        json_data = json.loads(body)
        print(json_data)
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        if json_data['events'][0]['message']['type'] == 'text':
            text_handler(json_data, line_bot_api)
        elif json_data['events'][0]['message']['type'] == 'image':
            image_handler(json_data, line_bot_api)
        else:
            pass
    except:
        print(request.args)
    return 'OK'

def text_handler(json_data, line_bot_api):
    msg = json_data['events'][0]['message']['text']
    instruction = """
    想要辨識植物🪴嗎?
    請注意以下拍照小訣竅📸:

    1. 請把植物放在照片的➡️中心點⬅️。
    2. 照片中只含有1️⃣種植物。
    3. 確認照片亮度💡是否適中，太亮或太暗都不行唷!
    4. 拍攝近照時至少包含整片葉面🌿，不要太過靠近。
    """
    reply_token = json_data['events'][0]['replyToken']
    line_bot_api.reply_message(reply_token,TextSendMessage(instruction))

def image_handler(json_data, line_bot_api):
    id = json_data['events'][0]['message']['id']
    img = line_bot_api.get_message_content(id).content
    reply_token = json_data['events'][0]['replyToken']
    img = Image.open(BytesIO(img))
    h, w, c = np.array(img).shape

    try:
        model.predict(img)
        pred = model.txt
    except:
        print(request.args)

    line_bot_api.reply_message(reply_token,TextSendMessage(f'{pred}'))
    
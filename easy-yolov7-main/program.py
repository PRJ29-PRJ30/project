import cv2
import json
import os
import threading
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from algorithm.object_detector import YOLOv7
from utils.detections import draw
from multiprocessing import Pool, Lock, freeze_support
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

WEIGHTS = 'best.weights'
CLASSES = 'classes.yaml'
DEVICE = 'gpu'  # 'gpu' kullanarak CUDA GPU hesaplaması için değiştirilebilir
STREAMS = 'streams.txt'

cred = credentials.Certificate('parkingapp1-f0ca5-firebase-adminsdk-injwv-3e3e2975f0.json')
firebase_admin.initialize_app(cred, {'databaseURL': "https://parkingapp1-f0ca5-default-rtdb.firebaseio.com"})

ref = db.reference("/Rezervasyon")
data = ref.get()

keys = list(data.keys())
updated_keys = [key.replace(" ", "") for key in keys]
updated_keys.extend([key.lower() for key in keys])
updated_keys.extend(keys)

ocr_classes = ['license-plate']
yolov7 = YOLOv7()
yolov7.load(WEIGHTS, classes=CLASSES, device=DEVICE)
yolov7.set(ocr_classes=ocr_classes)

texts = {}
lock = Lock()
anlik = datetime.now().strftime("%H:%M")
anlik_obje = datetime.strptime(anlik, "%H:%M")


def process_text(text):
    modified_text = []

    # Ignore işlemi
    ignore_list = ["TR", "TR;", ";TR", "TR "]  # Boşluk karakteri dahil edildi
    modifiye = text
    for ignore in ignore_list:
        modifiye = modifiye.replace(ignore, "")

    # İlk sayıya kadar olan boşluk karakterlerini ignore etme
    split_text = modifiye.split()
    for i, word in enumerate(split_text):
        if word.isnumeric():
            modified_text.append(" ".join(split_text[i:]))
            break

    # Küçük harfle değiştirme
    lower_text = modifiye.lower()
    modified_text.append(lower_text)

    # Büyük harfle değiştirme
    upper_text = modifiye.upper()
    modified_text.append(upper_text)

    # Küçük harfli halinin boşluk karakterinin çıkarılmış hali
    no_spaces_lower_text = ''.join(char for char in lower_text if char != " ")
    modified_text.append(no_spaces_lower_text)

    # Büyük harfli halinin boşluk karakterinin çıkarılmış hali
    no_spaces_upper_text = ''.join(char for char in upper_text if char != " ")
    modified_text.append(no_spaces_upper_text)

    return modified_text


def detect_stream(url):
    stream = cv2.VideoCapture(url)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not stream.isOpened():
        print(f'[!] {url} açılırken hata oluştu.')
        return

    try:
        while True:
            ret, frame = stream.read()
            if not ret:
                break

            detections = yolov7.detect(frame, track=True)
            for detection in detections:
                if detection['class'] in ocr_classes:
                    detection_id = detection['id']
                    text = detection['text']
                    modified_text = process_text(text)

                    print('Modified Text:', ', '.join(modified_text))

                    with lock:
                        for key in keys:
                            found = False
                            for text_elem in modified_text:
                                if key in text_elem:
                                    print("Plaka verisi Firebase'de bulunuyor.")
                                    reff = db.reference(f'/Rezervasyon/{text}/Durum')
                                    ref_saat = db.reference(f"/Rezervasyon/{text}/Çıkış Saati")
                                    data_saat = ref_saat.get()
                                    if data_saat is not None:
                                        saat_obje2 = datetime.strptime(data_saat, "%H:%M")
                                        if saat_obje2 > anlik_obje:
                                            reff.set('T')
                                        else:
                                            pass
                                    found = True
                                    break
                            if found:
                                break

                        if not found:
                            print("Plaka verisi Firebase'de bulunmuyor.")

                        if len(text) > 0:
                            if detection_id not in texts:
                                texts[detection_id] = {
                                    'most_frequent': {
                                        'value': '',
                                        'count': 0
                                    },
                                    'all': {}
                                }

                            if text not in texts[detection_id]['all']:
                                texts[detection_id]['all'][text] = 0

                            texts[detection_id]['all'][text] += 1

                            if texts[detection_id]['all'][text] > texts[detection_id]['most_frequent']['count']:
                                texts[detection_id]['most_frequent']['value'] = text
                                texts[detection_id]['most_frequent']['count'] = texts[detection_id]['all'][text]

            detected_frame = draw(frame, detections)
            print(f'\n{url}:\n', json.dumps(detections, indent=4))

            cv2.imshow("Live Detection", detected_frame)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        pass

    stream.release()
    print(f'[+] {url} kapatıldı.')


def start_detection(url):
    thread = threading.Thread(target=detect_stream, args=(url,))
    thread.start()


if __name__ == '__main__':
    freeze_support()

    with open(STREAMS) as file:
        streams_urls = [line.rstrip() for line in file]
        stream_count = len(streams_urls)
        print(f'[+] {stream_count} adet stream URL bulundu.')
        print('[+] canlı algılama başlatılıyor...')

        for url in streams_urls:
            start_detection(url)

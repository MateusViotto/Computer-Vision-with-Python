import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection   #Declara que será detectado uma face
face = mpFaceDetection.FaceDetection()

cap = cv2.VideoCapture(0) #Declara a webcam, o número indica qual webcam é, pode ser 0, 1...

pTime = 0
cTime = 0

while True:
    success, img = cap.read()   #lê a imagem da webcam

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converte a imagem para RGB
    results = face.process(imgRGB) #processa a face

    if results.detections:
        for id, detection in enumerate(results.detections): #Enumera as landmarks
            #mpDraw.draw_detection(img, detection)  # Detecta e desenha a face de forma automatica

            #Desenha um retangulo em volta do rosto de forma manual
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255,0,255), 2) #cria o retangulo
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)  # Mostra a precisao na tela

    # Calcula o FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # Printa o fps na tela

    cv2.imshow("Imagem", img)
    cv2.waitKey(1)
import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose   #Declara que será detectado uma pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0) #Declara a webcam, o número indica qual webcam é, pode ser 0, 1...

pTime = 0
cTime = 0

while True:
    success, img = cap.read()   #lê a imagem da webcam

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converte a imagem para RGB
    results = pose.process(imgRGB) #processa a pose
    print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) #Desenha a pose

        for id, lm in enumerate(results.pose_landmarks.landmark): #Enumera as landmarks
            h, w, c = img.shape
            #print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h) #Calcula a posição do landmark

            #if id == 0:  # Desenha um circulo rosa no landmark de id 0, checar a imagem dos landmaks caso haja duvida
             #   cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

    # Calcula o FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # Printa o fps na tela

    cv2.imshow("Imagem", img)
    cv2.waitKey(10)
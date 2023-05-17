import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) #Declara a webcam, o número indica qual webcam é, pode ser 0, 1...

mpHands = mp.solutions.hands    #Utiliza a biblioteca mediapipe e declara que será detectado uma mão
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #Desenha na tela de acordo com a detecção, nesse caso desenha pontos na mão

pTime = 0
cTime = 0

while True:
    success, img = cap.read()   #lê a imagem da webcam
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converte a imagem para RGB
    results = hands.process(imgRGB) #Processa se foi detectado uma mão
    #print(results.multi_hand_landmarks) #Mostra se a mão foi detectada e sua coordenada

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: #Verifica se existe mais de uma mão na tela
            for id, lm in enumerate(handLms.landmark): #Pega as informações e posições da mão
                #print(id,lm)
                h, w, c = img.shape #Altura, largura e canal
                cx, cy = int(lm.x*w), int(lm.y*h) #Declara o centro
                #print(id, cx, cy)

                if id == 0: #Desenha um circulo rosa no landmark de id 0, checar a imagem dos landmaks caso haja duvida
                    cv2.circle(img, (cx, cy), 25, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #Desenha pontos na mão

    #Calcula o FPS
    cTime = time.time()
    fps= 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) #Printa o fps na tela
    cv2.imshow("Imagem", img)   #mostra a imagem
    cv2.waitKey(1)



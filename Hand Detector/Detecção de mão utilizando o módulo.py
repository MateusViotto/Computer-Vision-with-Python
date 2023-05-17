import cv2
import time
import DetecMaoModulo as detecMao

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)  # Declara a webcam, o número indica qual webcam é, pode ser 0, 1...
detector = detecMao.handDetector()

while True:
    success, img = cap.read()  # lê a imagem da webcam

    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        print(lmList[0])    #Printa a posição do landmark 0
    # Calcula o FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # Printa o fps na tela
    cv2.imshow("Imagem", img)  # mostra a imagem
    cv2.waitKey(1)

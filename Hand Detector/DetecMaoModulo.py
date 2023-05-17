import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComp = 1, detectionCon=0.5, trackCon=0.5):


        self.mode = mode
        self.maxHands = maxHands
        self.modelComp = modelComp
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands  # Utiliza a biblioteca mediapipe e declara que será detectado uma mão
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComp, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # Desenha na tela de acordo com a detecção, nesse caso desenha pontos na mão


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte a imagem para RGB
        self.results = self.hands.process(imgRGB)  # Processa se foi detectado uma mão
        # print(results.multi_hand_landmarks) #Mostra se a mão foi detectada e sua coordenada

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # Verifica se existe mais de uma mão na tela
                if draw:  # Verifica se quer desenhar
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # Desenha pontos na mão

        return img


    def findPosition(self, img, handNumb=0, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNumb]

            for id, lm in enumerate(myHand.landmark):  # Pega as informações e posições da mão
                #print(id,lm)
                h, w, c = img.shape  # Altura, largura e canal
                cx, cy = int(lm.x * w), int(lm.y * h)  # Declara o centro
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw and id == 0:  # Desenha um circulo rosa no landmark de id 0, checar a imagem dos landmaks caso haja duvida
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Declara a webcam, o número indica qual webcam é, pode ser 0, 1...
    detector = handDetector()

    while True:
        success, img = cap.read()  # lê a imagem da webcam

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[0])    #Printa a posição do landmark 0
        # Calcula o FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # Printa o fps na tela
        cv2.imshow("Imagem", img)  # mostra a imagem
        cv2.waitKey(1)


if __name__ == "__main__":  # se ele estiver rodando esse arquivo
    main()

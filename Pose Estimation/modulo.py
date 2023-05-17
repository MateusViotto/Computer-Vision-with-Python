import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, complexity=1, smooth=True, enableseg = False, smoothseg = True, detecconf = 0.5, trackconf = 0.5):

        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enableseg = enableseg
        self.smoothseg = smoothseg
        self.detecconf = detecconf
        self.trackconf = trackconf


        self.mpPose = mp.solutions.pose  # Utiliza a biblioteca mediapipe e declara que será detectado uma mão
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, self.enableseg, self.smoothseg, self.detecconf, self.trackconf)
        self.mpDraw = mp.solutions.drawing_utils  # Desenha na tela de acordo com a detecção, nesse caso desenha pontos na mão


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte a imagem para RGB
        self.results = self.pose.process(imgRGB)  # Processa se foi detectado uma pose
        # print(results.multi_hand_landmarks) #Mostra se a pose foi detectada e sua coordenada

        if self.results.pose_landmarks:
            if draw:  # Verifica se quer desenhar
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img


    def findPosition(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:

            for id, lm in enumerate(self.results.pose_landmarks.landmark):  # Pega as informações e posições da pose
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
    detector = poseDetector()

    while True:
        success, img = cap.read()  # lê a imagem da webcam

        img = detector.findPose(img)
        lmList = detector.findPosition(img)

       # if len(lmList) != 0:
          #  print(lmList[0])    #Printa a posição do landmark 0
        # Calcula o FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # Printa o fps na tela
        cv2.imshow("Imagem", img)  # mostra a imagem
        cv2.waitKey(1)


if __name__ == "__main__":  # se ele estiver rodando esse arquivo
    main()

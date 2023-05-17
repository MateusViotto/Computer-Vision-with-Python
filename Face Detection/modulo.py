import cv2
import mediapipe as mp
import time

class faceDetector():
    def __init__(self, min_detection_confidence=0.5, model_selection=0):

        self.minDetec = min_detection_confidence
        self.model = model_selection

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = mp.solutions.face_detection  # Declara que será detectado uma face
        self.face = self.mpFaceDetection.FaceDetection(self.minDetec, self.model)


    def findFace(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte a imagem para RGB
        self.results = self.face.process(imgRGB)  # processa a face



        return img


    def findPosition(self, img, draw = True):
        lmList = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):  # Enumera as landmarks
                # mpDraw.draw_detection(img, detection)  # Detecta e desenha a face de forma automatica

                # Desenha um retangulo em volta do rosto de forma manual
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, bbox, (255, 0, 255), 2)  # cria o retangulo
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 3)  # Mostra a precisao na tela

        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # Declara a webcam, o número indica qual webcam é, pode ser 0, 1...
    detector = faceDetector()

    while True:
        success, img = cap.read()  # lê a imagem da webcam

        img = detector.findFace(img)
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

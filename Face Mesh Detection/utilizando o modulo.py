import cv2
import time
import modulo as detfacemesh

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)  # Declara a webcam, o número indica qual webcam é, pode ser 0, 1...
detector = detfacemesh.FaceMeshDetector()

while True:
    success, img = cap.read()  # Lê a imagem
    img, faces = detector.findFaceMesh(img)
    if len(faces) != 0:
        print(len(faces))

    # Calcula o FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # Printa o fps na tela

    cv2.imshow("Imagem", img)  # Mostra na tela

    cv2.waitKey(1)  # Trava a tela
import cv2
import time
import mediapipe as mp

class FaceMeshDetector():

    def __init__(self, mode=False, numfaces=2, refine=False, detection=0.5, tracking=0.5):
        self.mode = mode
        self.numfaces = numfaces
        self.detection = detection
        self.tracking = tracking
        self.refine = refine

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh  # Cria a face mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.mode, self.numfaces, self.refine, self.detection, self.tracking)  # Cria o objeto
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte ára RGB
        results = self.faceMesh.process(imgRGB)  # processa o face mesh

        if results.multi_face_landmarks:  # se detectar multiplas faces
            faces= []
            for faceLms in results.multi_face_landmarks:
                face = []
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, connections=self.mpFaceMesh.FACEMESH_LIPS, landmark_drawing_spec=self.drawSpec)
                for id, lm in enumerate(faceLms.landmark):  # enumerate para dar o ID
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)  # define a webcam que sera capturada
    pTime = 0
    cTime = 0
    detector = FaceMeshDetector()
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

if __name__ == "__main__":
    main()
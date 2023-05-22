import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)   #define a webcam que sera capturada
pTime = 0
cTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh  #Cria a face mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)      #Cria o objeto
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()           #Lê a imagem
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converte ára RGB
    results = faceMesh.process(imgRGB)     #processa o face mesh

    if results.multi_face_landmarks: #se detectar multiplas faces
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, connections=mpFaceMesh.FACEMESH_LIPS, landmark_drawing_spec=drawSpec)

            for id,lm in enumerate(faceLms.landmark): #enumerate para dar o ID
                #print(lm)
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)


    # Calcula o FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # Printa o fps na tela

    cv2.imshow("Imagem", img)  # Mostra na tela

    cv2.waitKey(1)                      #Trava a tela
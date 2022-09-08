import cv2
import face_recognition as fr

imgElon = fr.load_image_file('Elon.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgElonTest = fr.load_image_file('ElonTest.jpg')
imgElonTest = cv2.cvtColor(imgElonTest,cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)

encodeElon = fr.face_encodings(imgElon)[0]
encodeElonTest = fr.face_encodings(imgElonTest)[0]

comparacao = fr.compare_faces([encodeElon],encodeElonTest)
distancia = fr.face_distance([encodeElon],encodeElonTest)

print(comparacao,distancia)
cv2.imshow('Elon',imgElon)
cv2.imshow('Elon Test',imgElonTest)
cv2.waitKey(0)
import cv2
import numpy as np
#import matplotlib.pyplot as pl


#cap = cv2.VideoCapture('asaa.mp4')
cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture("img/")
net = cv2.dnn.readNetFromONNX("E:\VScode\\racce detector\\best (1).onnx") ## diubah ke path model
#classes = ['negroid','indian','asia','unknwon']
classes = ['nigga', 'indian','asian']

while True:
    img = cap.read()[1]
    if img is None:
        break
    #img = cv2.resize(img, (640,350))
    img = cv2.resize(img, (1000,600))
    #img = cv2.resize(img, (1920,1080))
    blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
    net.setInput(blob)
    detections = net.forward()[0]


    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence >= 0.4:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] >= 0.4:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx-w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)
                

    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.4)
    #print(indices)

    for j in indices:
        x1,y1,w,h = boxes[j]
        n_classes = len(classes)-1
        if j > n_classes:
            j = n_classes 
        label = classes[j]
        color_text = (0, 0,0)
        color_rec = (0,0,0)
        if j == 1:
            color_text = (255,255,51)
            color_rec = (255,255,51)
        if j == 2:
            color_text = (255,255,255)
            color_rec =(255,255,255)
        if j ==3:
            color_text = (0,0,255)
            color_rec = (0,0,255)
            

        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),color_rec,2)
        cv2.putText(img, label, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.5,color_text,2)

                    
    cv2.imshow("VIDEO",img)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break
import cv2 as cv
from scipy.spatial import distance
import numpy as np
from collections import OrderedDict

#please provide the paths for resources.
yolomodel = {"config_path":"yolo-obj.cfg",
              "model_weights_path":"yolo-obj_best.weights",
              "dataset_names":"obj.names",
              "confidence_threshold": 0.5,
              "threshold":0.3
             }
             
video_src = "new_Trim_Trim.mp4"

class Tracker:
    def __init__(self, maxLost = 30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.lost = OrderedDict()
        self.maxLost = maxLost

    def addObject(self, new_object_location):
        self.objects[self.nextObjectID] = new_object_location
        self.lost[self.nextObjectID] = 0
        self.nextObjectID += 1

    def removeObject(self, objectID):
        del self.objects[objectID]
        del self.lost[objectID]

    @staticmethod
    def getLocation(bounding_box):
        xlt, ylt, xrb, yrb = bounding_box
        return (int((xlt + xrb) / 2.0), int((ylt + yrb) / 2.0))

    def update(self,  detections):
        if len(detections) == 0:
            lost_ids = list(self.lost.keys())

            for objectID in lost_ids:
                self.lost[objectID] +=1
                if self.lost[objectID] > self.maxLost: self.removeObject(objectID)

            return self.objects

        new_object_locations = np.zeros((len(detections), 2), dtype="int")

        for (i, detection) in enumerate(detections): new_object_locations[i] = \
            self.getLocation(detection)

        if len(self.objects)==0:
            for i in range(0, len(detections)): self.addObject(new_object_locations[i])
        else:
            objectIDs = list(self.objects.keys())
            previous_object_locations = np.array(list(self.objects.values()))
            D = distance.cdist(previous_object_locations, new_object_locations)
            row_idx = D.min(axis=1).argsort()
            cols_idx = D.argmin(axis=1)[row_idx]
            assignedRows, assignedCols = set(), set()

            for (row, col) in zip(row_idx, cols_idx):
                if row in assignedRows or col in assignedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = new_object_locations[col]
                self.lost[objectID] = 0
                assignedRows.add(row)
                assignedCols.add(col)

            unassignedRows = set(range(0, D.shape[0])).difference(assignedRows)
            unassignedCols = set(range(0, D.shape[1])).difference(assignedCols)

            if D.shape[0]>=D.shape[1]:
                for row in unassignedRows:
                    objectID = objectIDs[row]
                    self.lost[objectID] += 1
                    if self.lost[objectID] > self.maxLost:
                        self.removeObject(objectID)
            else:
                for col in unassignedCols:
                    self.addObject(new_object_locations[col])
        return self.objects

net = cv.dnn.readNetFromDarknet(yolomodel["config_path"], yolomodel["model_weights_path"])
labels = open(yolomodel["dataset_names"]).read().strip().split("\n")
np.random.seed(12345)
layer_names = net.getLayerNames()
layer_names = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

bbox_colors = np.random.randint(0, 255, size=(len(labels), 3))
maxLost = 5
tracker = Tracker(maxLost = maxLost)
cap = cv.VideoCapture(video_src)

(H, W) = (None, None)
writer = None

def count_nonblack_np(img):
    return img.any(axis=-1).sum()

def color_detection(image, show = False): #<-- True for debugging

    boundaries = [([17, 15, 100], [50, 56, 200]), #orange
    ([0, 0, 0], [255, 255, 60])] #black
    
    i = 0
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        try:
            mask = cv.inRange(image, lower, upper)
            output = cv.bitwise_and(image, image, mask = mask)
            tot_pix = count_nonblack_np(image)
            color_pix = count_nonblack_np(output)
        except:
            print("strange things..")
            return 'not_sure'
        ratio = color_pix/tot_pix
        print("ratio is:", ratio)
        if ratio > 0.01 and i == 0:
            return 'orange'
        elif ratio > 0.01 and i == 1:
            return 'black'

        i += 1

        if show == True:
            cv.imshow("images", np.hstack([image, output]))
            if cv.waitKey(0) & 0xFF == ord('q'):
              cv.destroyAllWindows()
    return 'not_sure'

while(True):

    success, image = cap.read()

    if not success:
        print("error!")
        break

    if W is None or H is None: (H, W) = image.shape[:2]

    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections_layer = net.forward(layer_names)
    detections_bbox = []
    boxes, confidences, classIDs = [], [], []

    for out in detections_layer:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > yolomodel['confidence_threshold']:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv.dnn.NMSBoxes(boxes, confidences, yolomodel["confidence_threshold"], yolomodel["threshold"])

    if len(idxs)>0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            detections_bbox.append((x, y, x+w, y+h))

            clr = [int(c) for c in bbox_colors[classIDs[i]]]

            if labels[classIDs[i]] == "P":
                color = color_detection(image[y:y+h,x:x+w])
                if color != 'not_sure':
                    if color == 'black':
                        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
                    else:
                        cv.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            else:
                cv.rectangle(image, (x, y), (x+w, y+h), clr, 2)

            cv.putText(image, "{}: {:.4f}".format(labels[classIDs[i]], confidences[i]), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)

    objects = tracker.update(detections_bbox)

    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    cv.imshow("image", image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    if writer is None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter("result.avi", fourcc, 30, (W, H), True)

    writer.write(image)

writer.release()
cap.release()
cv.destroyWindow("image")

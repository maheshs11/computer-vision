import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import time

#from openalpr_ocr import ocr
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess
from playsound import playsound

Count_Helmet =0
Helmet_Detected =0
Motor =0
Hdt=0
Noframes=0
second_pass =False
class YOLO(object):
    _defaults = {
            "model_path": 'C:\\Users\\Admin\\Documents\\3RD SEM\\PROJECT DOCS\\IMPLEMENTATION\\only_weapons\\only_weapons\\Unusual_Txt\\trained_weights_final.h5',
           # "model_path": 'E:/P_2020/Dataset/OIDv4_ToolKit/Waste_Manage/trained_weights_final_waste.h5',
            "anchors_path": 'model_data/yolo_anchors.txt',
            #"classes_path": 'E:/P_2020/Dataset/OIDv4_ToolKit/Waste_Manage/4_CLASS_test_classes_waste.txt',
            "classes_path": 'C:\\Users\\Admin\\Documents\\3RD SEM\\PROJECT DOCS\\IMPLEMENTATION\\only_weapons\\only_weapons\\Unusual_Txt\\4_CLASS_test_classes.txt',
            "score" : 0.3,
            "iou" : 0.45,
            "model_image_size" : (416, 416),
            "text_size" : 1,
    }
  
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class1()
        self.anchors = self._get_anchors1()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate1()

    def _get_class1(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors1(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate1(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],#[image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        global second_pass
        if second_pass == True:
            yolo = YOLO()

        thickness = (image.shape[0] + image.shape[1]) // 600
        fontScale=1
        ObjectsList = []
        global Motor
        global Noframes
        global frame

       
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            #print((score))
            print(predicted_class)
            if (predicted_class =='Hammer') or (predicted_class =='Axe') or (predicted_class =='Helmet') or (predicted_class =='Kitchen_Knife') or (predicted_class =='Knife')  or (predicted_class =='Handgun'):
                  print('&&&&&&&&&&&')

            label = '{} {:.2f}'.format(predicted_class, score)
            #label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom-top)/2+top
            mid_v = (right-left)/2+left
            global Helmet_Detected
            if predicted_class == 'Axe':
                print('abnormal')
            elif predicted_class == 'Hammer':
                print('abnormal')
            elif predicted_class == 'Helmet':
                print('abnormal')
             
            elif predicted_class == 'Kitchen_Knife':
                print('abnormal')

            elif predicted_class == 'Knife':
                print('abnormal')

        return image, ObjectsList

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        #image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        r_image, ObjectsList = self.detect_image(original_image_color)
        return r_image, ObjectsList
##############################################################################################################################################################################
##########################################################################################################################################################################

if __name__=="__main__":
    yolo = YOLO()

    # set start time to current time
    start_time = time.time()
    # displays the frame rate every 2 second
    display_time = 2
    # Set primarry FPS to 0
    fps = 0

    # we create the video capture object cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("We cannot open webcam")

    while True:
        ret, frame = cap.read()
        # resize our captured frame if we need
        frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

        # detect object on our frame
        r_image, ObjectsList = yolo.detect_img(frame)
        
        # show us frame with detection
        cv2.imshow("Weapons_Monitor", r_image)
       
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        # calculate FPS
        fps += 1
        TIME = time.time() - start_time
        if TIME > display_time:
            print("FPS:", fps / TIME)
            fps = 0 
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    yolo.close_session()

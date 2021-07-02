import detectron2
import cv2
import os
from d2go.runner import Detectron2GoRunner
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
from lockit import lock , unlock
from mobile_cv.predictor.api import create_predictor
from d2go.utils.demo_predictor import DemoPredictor

from d2go.runner import GeneralizedRCNNRunner



runner = GeneralizedRCNNRunner()
cfg = runner.get_default_cfg()

#runner = Detectron2GoRunner()
#cfg = runner.get_default_cfg()


#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")

#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml")

cfg.merge_from_file(r"/home/pi/Desktop/project xcv/torchscript_int8@tracing/config2.yml")
cfg.MODEL.WEIGHTS =  os.path.join(r"/home/pi/Desktop/project xcv/torchscript_int8@tracing/data.pth")
cfg.MODEL.DEVICE = "cpu" #"cuda" 
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
#model = runner.build_model(cfg)
#predictor = DefaultPredictor(cfg)


model = runner.build_model(cfg)
#model = create_predictor(predictor_path)
predictor = DemoPredictor(model)

#model = create_predictor(predictor_path)
#predictor = DemoPredictor(model)
#cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture("/home/pi/Desktop/project xcv/video-clip.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
#out = cv2.VideoWriter('output1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
 

"""out = cv2.VideoWriter(
                'newvideo1.mkv',
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                #fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), 
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )"""

while (cap.isOpened()):
	
    ret,frame=cap.read(0)
    frame = cv2.resize(frame, (224, 224))
    print(fps)
    print(num_frames)
    

    try:
      outputs = predictor(frame)
      #v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
      v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), instance_mode=ColorMode.IMAGE_BW)
      v = v.draw_instance_predictions(frame, outputs["instances"].to('cpu'))
      print(outputs["instances"].pred_classes)
      omt = str(outputs["instances"].pred_classes)
      outpclass = omt[8:9]
      print(outpclass)
      """while (cap.isOpened()): #outpclass is printing ang giving 0 if 0 comes then action this loop
          if outpclass == '0':
              #unlock(8) make ur own function to test 
              time.sleep(10) #Lock will remains open for 10 seconds. make this run in loop
              #lock(8)
              #GPIO.cleanup(8)"""
                               
      
      #out.write(v.get_image())
    #cv2_imshow("Moda", v.get_image())
    except:
      break
    
cap.release()
#out.release()
#cv2.destroyAllWindows()

import cv2
import argparse
# import orien_lines
import datetime
from imutils.video import VideoStream
from Common_TFOD import detector_utils as detector_utils
import pandas as pd
from datetime import date
import xlrd
from xlwt import Workbook
from xlutils.copy import copy 
import numpy as np

lst1=[]
lst2=[]
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()


if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.80
    
    #vs = cv2.VideoCapture(0)
    vs = VideoStream(0).start()
    #Oriendtation of machine    
    Orientation= 'bt'


    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    def count_no_of_times(lst):
        x=y=cnt=0
        for i in lst:
            x=y
            y=i
            if x==0 and y==1: 
                cnt=cnt+1
        return cnt 
    try:
        while True:
            frame = vs.read()
            frame = np.array(frame)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            # Run image through tensorflow graph
            boxes, scores, classes,num = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # final_score = np.squeeze(scores)
            count1 = 0
            for i in range(len(classes)):
                if scores is None or scores[i] > score_thresh:
                    count1 = count1 + 1
            num_face_detect = count1
            # Line_Position2=orien_lines.drawsafelines(frame,Orientation,Line_Perc1,Line_Perc2)
            # Draw bounding boxeses and text
            point_1,point_2,ID_1 =detector_utils.get_points_frame(
                num_face_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame,Orientation)

            midpoints,collectn,height = detector_utils.get_mid_point(point_1,point_2,num_face_detect)
            distance = detector_utils.get_distance(midpoints,height,num_face_detect)
            close = detector_utils.get_closest(distance,num_face_detect,300)# After Doing some RnD we found this Thresh Hold
            detector_utils.draw_box(frame,collectn,close,num_face_detect,scores ,ID_1)


            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:
            
                # Display FPS on frame
                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows() 
                    vs.stop()
                    break
        

        print("Average FPS: ", str("{0:.2f}".format(fps)))
        
    except KeyboardInterrupt: 
        no_of_time_hand_detected=count_no_of_times(lst2)
        no_of_time_hand_crossed=count_no_of_times(lst1)
        today = date.today()
        # save_data(no_of_time_hand_detected, no_of_time_hand_crossed)
        print("Average FPS: ", str("{0:.2f}".format(fps)))
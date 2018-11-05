from __future__ import print_function

import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf

from queue import Queue
from threading import Thread
from utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util

# import the necessary packages

from imutils.video import VideoStream
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import time
import datetime
import imutils
import cv2
import os


CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()

def showcamera(video_capture):
    fps = FPS().start()
    while True:
        frame = video_capture.read()
        input_q.put(frame)

        t = time.time()

        if output_q.empty():
            pass  # fill up queue
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            data = output_q.get()
            rec_points = data['rect_points']
            class_names = data['class_names']
            class_colors = data['class_colors']
            for point, name, color in zip(rec_points, class_names, class_colors):
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmax'] * args.width), int(point['ymax'] * args.height)), color, 3)
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmin'] * args.width) + len(name[0]) * 6,
                               int(point['ymin'] * args.height) - 10), color, -1, cv2.LINE_AA)
                cv2.putText(frame, name[0], (int(point['xmin'] * args.width), int(point['ymin'] * args.height)), font,
                            0.3, (0, 0, 0), 1)
            cv2.imshow('Video', frame)
        

    
        fps.update()

        
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    fps.stop()

    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))

    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))


parser = argparse.ArgumentParser()
parser.add_argument('-src', '--source', dest='video_source', type=int,
                    default=0, help='Device index of the camera.')
parser.add_argument('-wd', '--width', dest='width', type=int,
                    default=1280, help='Width of the frames in the video stream.')
parser.add_argument('-ht', '--height', dest='height', type=int,
                    default=720, help='Height of the frames in the video stream.')
args = parser.parse_args()


class ObjectDetectionModelDemo:
    def __init__(self):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = VideoStream('rtsp://admin:12345@192.168.111.112:554/doc/page/main.asp').start()
        self.frame = None
        self.thread = None
        self.stopEvent = None
        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None
        # create a button, that when pressed, will take the current
        # frame and save it to file

        
        self._camera = ""
        self._model = ""
        

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop)
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("Demonstration of various object detection models")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        label2 = tki.Label(self.root, text="Object detection models", fg="blue", font=("Courier", 44))
        label2.pack(side="top")

    def videoLoop(self):
       
        # This try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        fTime=False
        try:
            # keep looping over frames until we are instructed to stop
            input_q = Queue(5)  # fps is better if queue is higher but then more lags
            output_q = Queue()
            for i in range(1):
                t = Thread(target=worker, args=(input_q, output_q))
                t.daemon = True
                t.start()
            #    video_capture = WebcamVideoStream('rtsp://admin:adminGMZTIH@192.168.111.113:554/doc/page/main.asp', width=args.width,height=args.height).start()
            #    video_capture = WebcamVideoStream('rtsp://admin:12345@192.168.0.53:554/doc/page/main.asp', width=args.width,height=args.height).start()
            #vs = WebcamVideoStream('rtsp://admin:12345@192.168.111.112:554/doc/page/main.asp', width=args.width,height=args.height).start()
            #    video_capture = WebcamVideoStream(src=args.video_source, width=args.width, height=args.height).start()
            fps = FPS().start()
            while not self.stopEvent.is_set():

                                
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=1280)
                

                input_q.put(self.frame)

                t = time.time()

                font = cv2.FONT_HERSHEY_SIMPLEX
                data = output_q.get()
                rec_points = data['rect_points']
                class_names = data['class_names']
                class_colors = data['class_colors']
                for point, name, color in zip(rec_points, class_names, class_colors):
                    cv2.rectangle(self.frame, (int(point['xmin'] * 1280), int(point['ymin'] * 720)),
                                  (int(point['xmax'] * 1280), int(point['ymax'] * 720)), color, 3)
                    cv2.rectangle(self.frame, (int(point['xmin'] * args.width), int(point['ymin'] * 720)),
                                  (int(point['xmin'] * 1280) + len(name[0]) * 6,
                                   int(point['ymin'] * 720) - 10), color, -1, cv2.LINE_AA)
                    cv2.putText(self.frame, name[0], (int(point['xmin'] * 1280), int(point['ymin'] * 720)), font,
                                0.3, (0, 0, 0), 1)
                
                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                

                # if the panel is not None, we need to initialize it
                if self.panel is None:

                    self.video_frame=tki.Frame(relief="raised",borderwidth=5)
                    self.live_streem=tki.Label(self.video_frame, text="Display")
                    self.live_streem.pack()

                    self.panel = tki.Label(self.video_frame, image=image)
                    self.panel.image = image
                    self.panel.pack(padx=10, pady=10)
                    self.video_frame.pack(side="top")

                # otherwise, simply update the panel

                else:
                    self.panel.configure(image=image)
                    self.panel.image = image
                if fTime == False:

                    mainframe = tki.Frame(self.root, relief="raised",borderwidth=5)
                    
                    
                    # Add a grid
                    select_cam_frame = tki.Frame(mainframe, relief="raised",borderwidth=5)
                    tki.Label(select_cam_frame, text="Select an IP camera").pack()
                    

                    # Create a Tkinter variable
                    tkvar_camera = tki.StringVar()

                    # Dictionary with options
                    camera_choices = {'Camera 1 in 701','Camera 2 in 701','Camera 1 in 702','Camera 1 in 207'}
                    tkvar_camera.set('Camera 1 in 701') # set the default option
                     
                    tki.OptionMenu(select_cam_frame, tkvar_camera, *camera_choices).pack()
                                     
                    # on change dropdown value
                    def change_dropdown_camera(*args):
                        self._camera = tkvar_camera.get()
                        print( tkvar_camera.get() )
                     
                    # link function to change dropdown
                    tkvar_camera.trace('w', change_dropdown_camera)
                    select_cam_frame.pack(side="left")

                    select_model_frame=tki.Frame(mainframe, relief="raised",borderwidth=5)
                    tki.Label(select_model_frame, text="Select an object detection model").pack()

                    # Create a Tkinter variable
                    tkvar_model = tki.StringVar()

                    # Dictionary with options
                    model_choices = {'SSD Mobilenet V1','YOLO','Faster R-CNN','Mask R-CNN', 'ResNet'}
                    tkvar_model.set('SSD Mobilenet V1') # set the default option
                     
                    tki.OptionMenu(select_model_frame, tkvar_model, *model_choices).pack()

                     
                    # on change dropdown value
                    def change_dropdown_model(*args):
                        self._model = tkvar_model.get()
                        print( tkvar_model.get() )
                     
                    # link function to change dropdown
                    tkvar_model.trace('w', change_dropdown_model)

                    def run_button_pressed():
                        threading.Event().set()

                        self.vs.stop()
                        
                        if self._camera == 'Camera 1 in 701':
                            # Camera 1 in 701
                            self.vs = VideoStream('rtsp://admin:12345@192.168.111.112:554/doc/page/main.asp').start()
                        elif self._camera == 'Camera 2 in 701':
                            # Camera 2 in 701
                            self.vs = VideoStream('rtsp://admin:adminGMZTIH@192.168.111.113:554/doc/page/main.asp').start()
                        elif self._camera == 'Camera 1 in 702':
                            # Camera 1 in 702
                            self.vs = VideoStream('rtsp://admin:Admin12345@192.168.111.129:554/doc/page/main.asp').start()
                        elif self._camera == 'Camera 1 in 207':
                            # Camera 1 in 207
                            self.vs = VideoStream('rtsp://admin:12345@192.168.0.53:554/doc/page/main.asp').start()
                            
                        self.thread = threading.Thread(target=self.videoLoop)
                        print('Run button pressed!')

                    select_model_frame.pack(side="left")
                 

                    # Run Button
                    run = tki.Button(mainframe, text='Run', fg="blue",font=("Courier", 30),command=run_button_pressed)
                    run.pack(side="left")

                    # Quit Button
                    quitbutton = tki.Button(mainframe, text='Exit', fg="red",font=("Courier", 30),command=self.onClose)
                    quitbutton.pack(side="left")
                    mainframe.pack()
                    
                    fTime=True


            video_capture.stop()
            cv2.destroyAllWindows()
            fps.stop()
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")
    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.destroy()

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")




time.sleep(2.0)

demo1 = ObjectDetectionModelDemo()
demo1.root.mainloop()


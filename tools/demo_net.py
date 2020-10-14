import time
import numpy as np
import cv2
import torch
import multiprocessing as mp
import threading
import slowfast.utils.distributed as du
from slowfast.utils import logging
from slowfast.datasets import cv2_transform
from slowfast.datasets.cv2_transform import scale
from model_load import create_model
import queue
import conf
from slowfast.datasets.utils import tensor_normalize

logger = logging.get_logger(__name__)
np.random.seed(20)
global re
class VideoReader(object):

    def __init__(self, cfg):
        self.source = cfg.DEMO.WEB_CAM
        self.display_width = cfg.DEMO.DISPLAY_WIDTH
        self.display_height = cfg.DEMO.DISPLAY_HEIGHT
        self.cap = cv2.VideoCapture(self.source)
        self.queue = mp.Queue()

    def __iter__(self):

        if self.display_width > 0 and self.display_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        else:
            self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.source))
        return self

    def __next__(self):
        was_read, frame = self.queue.get()
        if not was_read:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None

        return was_read, frame

    def put_fn(self):
        prev_time = -1
        while self.cap.isOpened():
            grabeed = self.cap.grab()
            if grabeed:
                time_s = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if int(time_s) > int(prev_time):
                    res, frame = self.cap.retrieve()

            else:
                res = False
                frame = None
            # res,frame=self.cap.read()
            self.queue.put((res, frame))
    def get_fn(self):
        while True:
            res,frame=self.cap.get()
            return res,frame
    def start(self):
        self.p1 = threading.Thread(target=self.put_fn, args=(), daemon=True)
        self.p1.start()
        self.p2 = threading.Thread(target=self.__next__, args=(), daemon=True)
        self.p2.start()

    def clean(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.p1.join()
        self.p2.join()


class demo:

    def __init__(self, cfg):
        print('model start load!')
        self.cfg = cfg

        model, object_predictor, palette, boxes, labels = create_model(self.cfg)

        self.model = model
        self.object_predictor = object_predictor
        self.palette = palette
        self.boxes = []
        self.labels = labels
        self.queue_demo = queue.Queue()
        self.frame_provider = VideoReader(self.cfg)
        self.queue_img=queue.Queue()
        self.output_fps=30

    def obj_detect(self, mid_frame):

        outputs = self.object_predictor(mid_frame)

        fields = outputs["instances"]._fields
        pred_classes = fields["pred_classes"]
        selection_mask = pred_classes == 0
        # acquire person boxes
        # pred_classes = pred_classes[selection_mask]
        pred_boxes = fields["pred_boxes"].tensor[selection_mask]
        # scores = fields["scores"][selection_mask]
        boxes = cv2_transform.scale_boxes(self.cfg.DATA.TEST_CROP_SIZE,
                                          pred_boxes,
                                          self.frame_provider.display_height,
                                          self.frame_provider.display_width)
        boxes = torch.cat([torch.full((boxes.shape[0], 1), float(0)).cuda(), boxes], axis=1)
        # return boxes
        self.queue_demo.put(boxes)
        # return boxes
    def slowfast_predict(self, frames, labels,):  # slow fast

        start_time = time.time()

        inputs = torch.from_numpy(np.array(frames)).float() / 255.0
        print("frame change time is :",time.time()-start_time)
        inputs = tensor_normalize(inputs, self.cfg.DATA.MEAN, self.cfg.DATA.STD)
        inputs = inputs.permute(3, 0, 1, 2)


        inputs = inputs.unsqueeze(0)


        index = torch.linspace(0, inputs.shape[2] - 1, self.cfg.DATA.NUM_FRAMES).long()
        fast_pathway = torch.index_select(inputs, 2, index)


        # Sample frames for the slow pathway.
        index = torch.linspace(0, fast_pathway.shape[2] - 1,
                               fast_pathway.shape[2] // self.cfg.SLOWFAST.ALPHA).long()
        slow_pathway = torch.index_select(fast_pathway, 2, index)
        # logger.info('slow_pathway.shape={}'.format(slow_pathway.shape))
        inputs = [slow_pathway, fast_pathway]

        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda()
        else:
            inputs = inputs.cuda()

        boxes = self.queue_demo.get()

        if not len(boxes):
            preds = torch.tensor([])
        else:
            preds = self.model(inputs, boxes)


        if self.cfg.NUM_GPUS > 1:
            preds = du.all_gather(preds)[0]

        preds = preds.cpu().detach().numpy()
        pred_masks = preds > .1
        label_ids = [np.nonzero(pred_mask)[0] for pred_mask in pred_masks]
        pred_labels = [
            [labels[label_id] for label_id in perbox_label_ids]
            for perbox_label_ids in label_ids
        ]

        # boxes = boxes.cpu().detach().numpy()
        # ratio = np.min(
        #     [self.frame_provider.display_height, self.frame_provider.display_width]
        # ) / self.cfg.DATA.TEST_CROP_SIZE
        #
        # boxes = boxes[:, 1:] * ratio
        detection_time=time.time()
        print(f'slowfast cost time is :{(detection_time-start_time)}')
        # re=pred_labels
        self.fra(pred_labels)

    def predictor(self,):
        # seq_len = self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE

        seq_len=32
        frames=[]
        s = 0.
        self.frame_provider.start()
        for able_to_read, frame in self.frame_provider:
            sleep_time=1/self.output_fps
            if not able_to_read:
                frames = []
                continue

            if len(frames) != seq_len:
                frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_processed = scale(self.cfg.DATA.TEST_CROP_SIZE, frame_processed)
                frames.append(frame_processed)
                if self.cfg.DETECTION.ENABLE and len(frames) == seq_len // 2 - 1:
                    mid_frame = frame
            re = '0.00'
            if len(frames) == seq_len:
                # self.start(mid_frame,frames,self.labels)
                # frames = []
                # pool=mp.Pool()
                # pool.apply_async(func=self.start,args=(mid_frame,frames,self.labels),callback=self.fra)
                # arg=[mid_frame,frames,self.labels]
                # result=threading.Thread(target=self.appy_async,args=(self.start,arg,self.fra))
                # result.start()
                self.obj_detect(mid_frame)
                result=self.slowfast_predict(frames,self.labels)
                re=result
                frames=[]
            cv2.putText(frame, 'Speed: {:}s'.format(re), (10, 25),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=.65, color=(0, 235, 0), thickness=2)
            # Display the frame
            cv2.imshow('SlowFast', frame)
            # time.sleep(sleep_time)

            key = cv2.waitKey(1)
            if key == 27:
                break
        self.clean()
        self.frame_provider.clean()
    def start(self,mid_frame,frames,labels):
        self.thread_1 = threading.Thread(target=self.obj_detect, args=(mid_frame,))
        self.thread_2 = threading.Thread(target=self.slowfast_predict, args=(frames, labels,))
        self.thread_1.start()
        self.thread_2.start()
        return None
    def appy_async(self,func,args,callback):

        # Compute the result
        result = func(*args)

        # Invoke the callback with the result
        callback(result)
    def fra(self,lis):
        print(lis)
        if len(lis) == 0:
            print("NO PERSON")
        elif any(lis) in conf.tar_label:
            print("WARINIG")
        else:
            print("NORMAL")

        return None
def clean(self):
        self.thread_1.join()
        self.thread_2.join()

import threading
import time

import cv2


class CameraV1(threading.Thread):
    def __init__(self, address, fps=20, is_video=False):
        self.capture = cv2.VideoCapture(address)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        assert self.capture.isOpened()
        self.isOpened = self.capture.isOpened()
        self.cond = threading.Condition()
        self.running = False
        self.frame = None
        self.latestnum = 0
        self.callback = None
        self.isVideo = is_video
        super().__init__()
    
    def start(self):
        self.running = True
        super().start()
    
    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()
    
    def run(self):
        counter = 0
        while self.running:
            if self.isVideo:
                time.sleep(0.035)
            # block for fresh frame
            (rv, img) = self.capture.read()
            # assert rv
            counter += 1
            
            # publish the frame
            with self.cond:  # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()
            
            if self.callback:
                self.callback(img)
    
    def read(self, wait=True, seqnumber=None, timeout=None):
        # with no arguments (wait=True), it always blocks for a fresh frame
        # with wait=False it returns the current frame immediately (polling)
        # with a seqnumber, it blocks until that frame is available (or no wait at all)
        # with timeout argument, may return an earlier frame;
        #   may even be (0,None) if nothing received yet
        
        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum + 1
                if seqnumber < 1:
                    seqnumber = 1
                
                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return self.latestnum, self.frame
            
            return self.latestnum, self.frame
    
    def get(self):
        """Get video properties: frame width and height."""
        if not self.capture.isOpened():
            return None, None
        
        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        return height, width

import sys
sys.path.append('/home/pi/ArmPi/')
import Camera
import cv2
import numpy as np

class Color_Track:
    def __init__(self) -> None:
        
        # Start the camera
        self.my_camera = Camera.Camera()
        self.my_camera.camera_open()
        
        pass
    
    def get_frame(self):
        """Gets a frame

        Returns:
            _type_: _description_
        """
        frame = None
        img = self.my_camera.frame
        if img is not None:
            frame = img.copy()
        return frame
    
    def pre_process_image(self, img):
        """Resizes and applies gaussian blur

        Args:
            img (_type_): _description_
        """
        size = (640, 480)
        frame_resize = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        frame_gb = cv2.GaussianBlur(frame_resize, (11, 11), 11)
        return frame_gb
    
    def getMaskROI(self, frame, roi, size):
        """Turns all pixels outside the ROI black

        Args:
            frame (_type_): _description_
            roi (_type_): _description_
            size (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_min, x_max, y_min, y_max = roi
        x_min -= 10
        x_max += 10
        y_min -= 10
        y_max += 10

        if x_min < 0:
            x_min = 0
        if x_max > size[0]:
            x_max = size[0]
        if y_min < 0:
            y_min = 0
        if y_max > size[1]:
            y_max = size[1]

        black_img = np.zeros([size[1], size[0]], dtype=np.uint8)
        black_img = cv2.cvtColor(black_img, cv2.COLOR_GRAY2RGB)
        black_img[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]
        
        return black_img
    
    
    
    
    
    
    def run(self):
        """Runs one timestep
        """
        
        # Start by getting a frame
        frame = self.get_frame()
        if not frame:
            return
        # Pre-process with resizing and blur
        frame = self.pre_process_image(frame)
        
        
        
    
if __name__ == '__main__':
    ct = Color_Track()
    ct.run()
    pass
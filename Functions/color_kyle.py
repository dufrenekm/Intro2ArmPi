
import sys
sys.path.append('/home/pi/ArmPi/')
import Camera
import cv2
import numpy as np
from time import sleep
from LABConfig import color_range
sys.path.append('/home/pi/ArmPi/CameraCalibration/')
from CalibrationConfig import square_length
sys.path.append('/home/pi/ArmPi/ArmIK/')
from Transform import getROI, getCenter, convertCoordinate
import math
import logging
logging_format = "%(asctime)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO,datefmt="%H:%M:%S")
logging.getLogger().setLevel(logging.DEBUG)


class Color_Track:
    def __init__(self) -> None:
        
        # Start the camera
        self.my_camera = Camera.Camera()
        self.my_camera.camera_open()
        self.target = ('red',)
        self.size = (640, 480)
        self.range_rgb = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'black': (0, 0, 0),
            'white': (255, 255, 255)}
        self.last_x, self.last_y = 0, 0
        
    
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
    
    def getAreaMaxContour(self, contours):
        contour_area_temp = 0
        contour_area_max = 0
        area_max_contour = None

        for c in contours:  # Traverse all contours
            contour_area_temp = math.fabs(cv2.contourArea(c))  # Calculate the contour area
            if contour_area_temp > contour_area_max:
                contour_area_max = contour_area_temp
                if contour_area_temp > 300:  # Only when the area is greater than 300, the outline of the largest area is effective to filter interference
                    area_max_contour = c

        return area_max_contour, contour_area_max  # Return the largest contour
    
    def get_contours(self, frame):
        # Convert the image to LAB space
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  
        
        area_max = 0
        areaMaxContour = 0
        for i in color_range:
            if i in self.target:
                detect_color = i
                frame_mask = cv2.inRange(frame_lab, color_range[detect_color][0], color_range[detect_color][1])  # Perform bit operations on the original image and mask
                opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))  # Open operation
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))  # Close operation
                contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # Find the contour
                areaMaxContour, area_max = self.getAreaMaxContour(contours)  # Find the largest contour
        logging.debug(f"Area max contour: {areaMaxContour}, Area max: {area_max}")
        return areaMaxContour, area_max, detect_color
        
        
    def process_contour(self, areaMaxContour, area_max, color, img):
        if area_max < 2500:
            # TODO: Update this
            return False
        
        rect = cv2.minAreaRect(areaMaxContour)
        box = np.int0(cv2.boxPoints(rect))

        roi = getROI(box) #Get roi area
        get_roi = True

        img_centerx, img_centery = getCenter(rect, roi, self.size, square_length)  # Get the center coordinates of the wooden block
        world_x, world_y = convertCoordinate(img_centerx, img_centery, self.size) #Convert to real world coordinates
        
        
        cv2.drawContours(img, [box], -1, self.range_rgb[color], 2)
        cv2.putText(img, '(' + str(world_x) + ',' + str(world_y) + ')', (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.range_rgb[color], 1) #draw center point
        distance = math.sqrt(pow(world_x - self.last_x, 2) + pow(world_y - self.last_y, 2)) #Compare the last coordinates to determine whether to move
        self.last_x, self.last_y = world_x, world_y
        track = True
        return img
    
    
    def run(self):
        """Runs one timestep
        """
        
        # Start by getting a frame
        orig_frame = self.get_frame()
        if orig_frame is None:
            return
        # print(frame.shape)
        logging.debug("Got a valid frame.")
        # Pre-process with resizing and blur
        frame = self.pre_process_image(orig_frame)
        # Get contours 
        shape, area, color = self.get_contours(frame)
        # Process contours
        new_img = self.process_contour(shape, area, color, orig_frame)
        cv2.imshow('Frame', new_img)
        key = cv2.waitKey(1)
        
        
        
        
    
if __name__ == '__main__':
    ct = Color_Track()
    try:
        ct.target = ('red',)
        for i in range(100000):
            ct.run()
            sleep(.1)
    except:
        ct.my_camera.camera_close()
        cv2.destroyAllWindows()
    
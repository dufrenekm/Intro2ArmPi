
import sys
sys.path.append('/home/pi/ArmPi/')
import Camera
from copy import deepcopy
import cv2
import numpy as np
from time import sleep
from LABConfig import color_range
sys.path.append('/home/pi/ArmPi/CameraCalibration/')
from CalibrationConfig import square_length
sys.path.append('/home/pi/ArmPi/ArmIK/')
from Transform import getROI, getCenter, convertCoordinate, getAngle
import math
import logging
import threading
import HiwonderSDK.Board as Board
from ArmIK.ArmMoveIK import ArmIK
import math
from py_bullet_arm_ik.pybullet_arm import PyBulletIK



logging_format = "%(asctime)s: %(message)s"
# logging.basicConfig(format=logging_format, level=logging.INFO,datefmt="%H:%M:%S")
# logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger('color_kyle')
logger.setLevel(logging.DEBUG)
# logger.basicConfig(format=logging_format, level=logging.INFO,datefmt="%H:%M:%S")


class ColorTrack:
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
        color = 'red'
        for i in color_range:
            if i in self.target:
                detect_color = i
                frame_mask = cv2.inRange(frame_lab, color_range[detect_color][0], color_range[detect_color][1])  # Perform bit operations on the original image and mask
                opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))  # Open operation
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))  # Close operation
                contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # Find the contour
                new_areaMaxContour, new_area_max = self.getAreaMaxContour(contours)  # Find the largest contour
                if new_area_max > area_max:
                    area_max = new_area_max
                    areaMaxContour = new_areaMaxContour
                    color = detect_color
        logger.debug(f"Area max: {area_max}")
        return areaMaxContour, area_max, color
        
        
    def process_contour(self, areaMaxContour, area_max, color, img):
        if area_max < 2500:
            # TODO: Update this
            return False, False, False, False
        
        rect = cv2.minAreaRect(areaMaxContour)
        box = np.int0(cv2.boxPoints(rect))

        roi = getROI(box) #Get roi area
        get_roi = True

        img_centerx, img_centery = getCenter(rect, roi, self.size, square_length)  # Get the center coordinates of the wooden block
        world_x, world_y = convertCoordinate(img_centerx, img_centery, self.size) #Convert to real world coordinates
        
        logger.debug(f"Marking {color}.")
        cv2.drawContours(img, [box], -1, self.range_rgb[color], 2)
        cv2.putText(img, '(' + str(world_x) + ',' + str(world_y) + ')', (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.range_rgb[color], 1) #draw center point
        distance = math.sqrt(pow(world_x - self.last_x, 2) + pow(world_y - self.last_y, 2)) #Compare the last coordinates to determine whether to move
        self.last_x, self.last_y = world_x, world_y
        track = True
        return img, rect, world_x, world_y
    
    
    def run(self):
        """Runs one timestep
        """
        
        # Start by getting a frame
        orig_frame = self.get_frame()
        if orig_frame is None:
            return False, False, False, False
        # print(frame.shape)
        logger.debug("Got a valid frame.")
        # Pre-process with resizing and blur
        frame = self.pre_process_image(orig_frame)
        # Get contours 
        shape, area, color = self.get_contours(frame)
        # Process contours
        new_img, rect, x, y = self.process_contour(shape, area, color, orig_frame)
        cv2.imshow('Frame', new_img)
        key = cv2.waitKey(1)
        
        return color, rect, x, y
        
class MoveArm():
    def __init__(self):
        self.AK = ArmIK()
        self.servo1 = 500
        self.servo2_angle = 0
        # Go to initial position
        self.move_initial_pos()
        
        # Variable for determining if action is being executed
        self.operation = 'stack' # or 'sort'
        self.picking = False
        self.number_moves = 0
        # Drop coordinates (x, y, z) of blocks
        self.coordinate = {
        'red':   (-15 + 0.5, 12 - 0.5, 1.5),
        'green': (-15 + 0.5, 6 - 0.5,  1.5),
        'blue':  (-15 + 0.5, 0 - 0.5,  1.5),
        'stack': (-15 + 1, -7 - 0.5, 1.5)}
        

    def move_initial_pos(self):
        
        Board.setBusServoPulse(1, self.servo1 - 50, 300)
        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500) 
        
    def pick_object(self, x, y, angle):      
        # Move above the block          
        result = self.AK.setPitchRangeMoving((x, y - 2, 5), -90, -90, 0)
        if result == False:
            unreachable = True
        else:
            unreachable = False
        sleep(result[2]/1000) #The third item of the return parameter is time
        
        # Open gripper
        Board.setBusServoPulse(1, self.servo1 - 280, 500) 
        # Calculate the angle at which the gripper needs to be rotated, then rotate
        self.servo2_angle = getAngle(x, y, angle)
        Board.setBusServoPulse(2, self.servo2_angle, 500)
        sleep(0.8)
        
        # Lower arm
        self.AK.setPitchRangeMoving((x, y, 2), -90, -90, 0, 1000)  # lower the altitude
        sleep(2)
        
        # Close gripper 
        Board.setBusServoPulse(1, self.servo1, 500)  # Gripper closed
        sleep(1)
        
        # Raise arm
        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((x, y, 12), -90, -90, 0, 1000) # Robotic arm raised
        sleep(1)

    
    def place_object_sort(self, color):
        
        # Move above the drop location
        result = self.AK.setPitchRangeMoving((self.coordinate[color][0], self.coordinate[color][1], 12), -90, -90, 0)   
        sleep(result[2]/1000)
        
        # Rotate wrist to correct orientation
        self.servo2_angle = getAngle(self.coordinate[color][0], self.coordinate[color][1], -90)
        Board.setBusServoPulse(2, self.servo2_angle, 500)
        sleep(0.5)

        # Lower block down mostly
        self.AK.setPitchRangeMoving((self.coordinate[color][0], self.coordinate[color][1], self.coordinate[color][2] + 3), -90, -90, 0, 500)
        sleep(0.5)
        
        # Place block down         
        self.AK.setPitchRangeMoving((self.coordinate[color]), -90, -90, 0, 1000)
        sleep(0.8)
                   
        # Open servo
        Board.setBusServoPulse(1, self.servo1 - 200, 500)  # Open the claws and put down the object
        sleep(0.8)
        
        # Lift arm up                    
        self.AK.setPitchRangeMoving((self.coordinate[color][0], self.coordinate[color][1], 12), -90, -90, 0, 800)
        sleep(0.8)

        # Move back home
        self.move_initial_pos()
        sleep(1.5)
        
    def stack_block(self):
        height = self.coordinate['stack'][2] + 2.5 * self.number_moves
        
        # Move above the drop location
        result = self.AK.setPitchRangeMoving((self.coordinate['stack'][0], self.coordinate['stack'][1], 12), -90, -90, 0)   
        sleep(result[2]/1000)
        
        # Rotate wrist to correct orientation
        self.servo2_angle = getAngle(self.coordinate['stack'][0], self.coordinate['stack'][1], -90)
        Board.setBusServoPulse(2, self.servo2_angle, 500)
        sleep(0.5)

        # Lower block down mostly
        self.AK.setPitchRangeMoving((self.coordinate['stack'][0], self.coordinate['stack'][1], height + 3), -90, -90, 0, 500)
        sleep(0.5)
        
        # Place block down         
        self.AK.setPitchRangeMoving((self.coordinate['stack'][0], self.coordinate['stack'][1], height), -90, -90, 0, 1000)
        sleep(0.8)
                   
        # Open servo
        Board.setBusServoPulse(1, self.servo1 - 200, 500)  # Open the claws and put down the object
        sleep(0.8)
        
        # Lift arm up                    
        self.AK.setPitchRangeMoving((self.coordinate[color][0], self.coordinate[color][1], 12), -90, -90, 0, 800)
        sleep(0.8)

        # Move back home
        self.move_initial_pos()
        sleep(1.5)
        
    
    def execute_arm(self, x, y, angle, color):
        logger.debug(f"Starting arm movement: {self.operation}!")
        self.setBuzzer(0.1)    
        # Pick object
        self.pick_object(x, y, angle)
        
        # Place object
        if ma.operation == 'sort':
            self.place_object_sort(color)
        else: 
            self.stack_block()
        
        # Update counter/allow next move
        self.number_moves += 1
        self.picking = False
        
    def setBuzzer(self, timer):
        Board.setBuzzer(0)
        Board.setBuzzer(1)
        sleep(timer)
        Board.setBuzzer(0)
        
if __name__ == '__main__':
    # ma = MoveArm()
    # ct = ColorTrack()
    
    # # Get user input for stacking or sorting
    # user_in = int(input("What would you like to do? 1 for sorting, 2 for stacking: "))
    # if user_in == 1:
    #     ma.operation = 'sort'
    # elif user_in == 2:
    #     ma.operation = 'stack'
    # else:
    #     logger.error(f"Operation '{user_in}' not implemented!!")
    #     exit()
    
    # try:
    #     ct.target = ('red','blue','green')
    #     while True:
    #         color, rect, x, y = ct.run()
    #         if not color:
    #             # If not valid, we just loop again
    #             continue
            
    #         # If we aren't already doing an operation and we haven't finished
    #         if not ma.picking and ma.number_moves < 3:
    #             ma.picking = True
    #             # Start an operation in another thread
    #             th = threading.Thread(target=ma.execute_arm, args=(x, y, rect[2], color,))
    #             th.setDaemon(True)
    #             th.start()
                
    #         if ma.number_moves == 3:
    #             logger.debug(f"Finished operation: {ma.operation}")
    #             ct.my_camera.camera_close()
    #             cv2.destroyAllWindows()
    #             exit()

    #         sleep(.1)
    # except Exception as e:
    #     logger.error(f"Exception: {e}.")
    #     ct.my_camera.camera_close()
    #     cv2.destroyAllWindows()
    AK = ArmIK()
    IK = PyBulletIK()
    servo1 = 500
    Board.setBusServoPulse(1, servo1 - 50, 300)
    Board.setBusServoPulse(2, 500, 500)
    AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500) 
    # print(Board.getPWMServoAngle(1))
    sleep(4)
    pwm_3, pwm_4, pwm_5, pwm_6 = AK.cur_pulse
    pwms = (pwm_3, pwm_4, pwm_5, pwm_6)
    print(pwms)
    print(AK.pwm_to_angle((pwms)))

    print("here")
    print(AK.cur_pulse)
    # sleep(1)

    # Start pos approx angle
    #servos = AK.transformAngelAdaptArm(-90,104,160,-270)


    # Goal poses 
    goals = [[.2, 0, .2],
    [.1, .2, .1],
    [.15, .15, .2]]

    for goal in goals:

        angles = IK.compute_ik(goal)
        print(angles)
        servos = AK.transform_pybullet(angles[-2], angles[-3], angles[-4], angles[-5])
        AK.servosMove((servos["servo3"], servos["servo4"], servos["servo5"], servos["servo6"]), None)
        sleep(5)

    # for i in range(100):
    #     print(IK.compute_ik([.2, 0, .1]))
    # servos = AK.transform_pybullet(.477,.95,.5,.4)
    # print("Servo vals")
    # print(servos)
    # AK.servosMove((servos["servo3"], servos["servo4"], servos["servo5"], servos["servo6"]), None)

    # sleep(3)
    # AK.set_servo_pulse(3, 139, 0)
    # AK.set_servo_pulse(4, pwm_4, 0)


    # AK.reset_servo(3)
    # AK.reset_servo(4)
    # AK.reset_servo(5)
    # AK.reset_servo(6)
    # AK.reset_servo(2)
    # AK.reset_servo(1)
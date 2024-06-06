import cv2
from Point import Point

class Chopper(Point):
    def __init__(self, name):
        super(Chopper, self).__init__(name)
        self.icon = cv2.imread("pictures/drone.png") /255.0
        self.icon_w = 32
        self.icon_h = 32
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
        self.tips = []
        self.sensors = []

    def create_tips(self):
        y, x = self.get_position()
        top_left = int(y-self.icon_h/2), int(x-self.icon_w/2)
        top_right = int(y-self.icon_h/2), int(x+self.icon_w/2)
        bootom_right = int(y+self.icon_h/2), int(x+self.icon_w/2)
        bottom_left = int(y+self.icon_h/2), int(x-self.icon_w/2)
        
        return [top_left, top_right, bootom_right, bottom_left]

    def create_sensors(self):
        y, x = self.get_position()
        north_sensor = int(y-self.icon_h-10), int(x)
        east_sensor = int(y), int(x+self.icon_w+10)
        south_sensor = int(y+self.icon_h+10), int(x)
        west_sensor = int(y), int(x-self.icon_w-10)
        
        return [north_sensor, east_sensor, south_sensor, west_sensor]
    
    def set_tips(self):
        self.tips = self.create_tips()
    
    def set_sensors(self):
        self.sensors = self.create_sensors()
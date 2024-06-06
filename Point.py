class Point(object):
    def __init__(self, name):
        self.x = 0
        self.y = 0
        self.name = name
    
    def set_position(self, y, x):
        self.y = y
        self.x = x
    
    def get_position(self):
        return (self.y, self.x)
    
    def move(self, del_y, del_x):
        self.x += del_x
        self.y += del_y
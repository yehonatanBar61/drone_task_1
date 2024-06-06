import gym
import cv2 
import time
import random
import numpy as np 
from Map import Map
from numba import njit
import PIL.Image as Image
from IPython import display
from gym import Env, spaces
from Chopper import Chopper
import matplotlib.pyplot as plt


font = cv2.FONT_HERSHEY_COMPLEX_SMALL

class ChopperScape(Env):
    def __init__(self, path: str):
        super(ChopperScape, self).__init__()

        self.map = Map(path)
        
        # Define a 2-D observation space
        self.observation_shape = self.map.map.shape # (600, 800, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)
        
        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(6,)
                        
        # Create a canvas to render the environment images upon 
        self.canvas = np.ones(self.observation_shape) * 1
        
        # Define elements present inside the environment
        self.elements = []
        
        # Maximum fuel chopper can take at once
        self.max_fuel = 1000

        # Permissible area of helicper to be 
        self.y_min = int (self.observation_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int (self.observation_shape[0] * 0.9)
        self.x_max = self.observation_shape[1]

    def draw_elements_on_canvas(self):
        # Init the canvas 
        # self.canvas = np.ones(self.observation_shape) * 1

        # Draw the heliopter on canvas
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x,y = elem.x, elem.y
            self.canvas[int(y - elem_shape[1]/2) : int(y + elem_shape[1]/2), int(x - elem_shape[0]/2 ): int(x + elem_shape[0]/2)] = elem.icon

        text = 'Fuel Left: {} | Rewards: {}'.format(self.fuel_left, self.ep_return)

        # Put the info on canvas 
        self.canvas = cv2.putText(self.canvas, text, (10,20), font,  
                0.8, (255,255,0), 1, cv2.LINE_AA)
        
    @staticmethod
    @njit
    def draw_map_on_canvas(canvas: np.ndarray, map: np.ndarray):
        # Init the canvas 
        canvas = np.ones(map.shape) * 1
        
        h, w, _ = canvas.shape
        for y in range(h):
            for x in range(w):
                canvas[y, x] = map[y, x]
        
        return canvas

    def reset(self):
        # Reset the fuel consumed
        self.fuel_left = self.max_fuel

        # Reset the reward
        self.ep_return  = 0

        # Number of birds
        self.bird_count = 0
        self.fuel_count = 0

        # Determine a place to intialise the chopper in
        x = 100
        y = 50
        
        # Intialise the chopper
        self.chopper = Chopper("chopper")
        self.chopper.set_position(y, x)
        self.chopper.set_tips()
        self.chopper.set_sensors()

        # Intialise the elements 
        self.elements = [self.chopper]

        # Reset the Canvas 
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw elements on the canvas
        self.canvas = self.draw_map_on_canvas(self.canvas, self.map.map)
        self.draw_elements_on_canvas()

        # return the observation
        return self.canvas
    
    def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Simulation", self.canvas)
            cv2.waitKey(100)
        
        elif mode == "rgb_array":
            return self.canvas
    
    def close(self):
        cv2.destroyAllWindows()

    def get_action_meanings(self):
        return {0: "Up", 1: "Right", 2: "Down", 3: "Left"} #, 4: "Do Nothing"}

    def has_collided(self):
        tips = self.chopper.tips
        temp = self.map.map[tips[0][0], tips[0][1]]
        if self.map.is_black(tips[0][0], tips[0][1]): return True
        elif self.map.is_black(tips[1][0], tips[1][1]): return True
        elif self.map.is_black(tips[2][0], tips[2][1]): return True
        elif self.map.is_black(tips[3][0], tips[3][1]): return True
        else: return False

    def step(self, action):
        # Flag that marks the termination of an episode
        done = False
        
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the fuel counter 
        self.fuel_left -= 1 
        
        # Reward for executing a step.
        reward = 1      

        # apply the action to the chopper
        if action == 0:
            self.chopper.move(-2, 0)
        elif action == 1:
            self.chopper.move(0, 2)
        elif action == 2:
            self.chopper.move(2, 0)
        elif action == 3:
            self.chopper.move(0, -2)
        self.chopper.set_tips()
        self.chopper.set_sensors()

        # If chopper has collided
        if self.has_collided():
            # Conclude the episode and remove the chopper from the Env.
            done = True
            reward = -10
            self.elements.remove(self.chopper)
        
        # Increment the episodic return
        self.ep_return += 1

        # Draw elements on the canvas
        self.canvas = self.draw_map_on_canvas(self.canvas, self.map.map)
        self.draw_elements_on_canvas()

        # If out of fuel, end the episode.
        if self.fuel_left == 0:
            done = True

        return self.canvas, reward, done, []

if __name__ == "__main__":
    path_to_map = "pictures/maps/p11.png"
    # env = ChopperScape(path_to_map)
    # obs = env.reset()
    # screen = env.render(mode = "rgb_array")
    # plt.imshow(screen)
    # plt.show()

    env = ChopperScape(path_to_map)
    obs = env.reset()

    while True:
        # Take a random action
        print(env.fuel_left)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        
        # Render the game
        env.render(mode='human')
        env.canvas = np.ones(env.observation_shape) * 1
        
        if done == True:
            break

    env.close()
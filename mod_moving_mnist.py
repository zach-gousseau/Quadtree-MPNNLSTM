from random import choice
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from fastai.vision.all import untar_data, URLs, get_image_files, load_image

class ModMovingMNISTDataset(Dataset):
    def __init__(self,
                 n_samples,
                 input_timesteps,
                 output_timesteps,
                 n_digits=1,
                 gap=0,
                 canvas_size=(32, 32),
                 digit_size=(12, 12),
                 pixel_noise=0.05,
                 velocity_noise=0.25):
        self.mmmnist = ModMovingMNIST(canvas_size, digit_size, pixel_noise, velocity_noise)
        
        x, y = self.mmmnist.create_dataset(n_samples, input_timesteps, output_timesteps, n_digits, gap)
        self.x, self.y = torch.from_numpy(x).type(torch.float32), torch.from_numpy(y).type(torch.float32)
        
        self.image_shape = x.shape[2:4]
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ModMovingMNIST:
    def __init__(self, canvas_size=(32, 32), digit_size=(12, 12), pixel_noise=0.05, velocity_noise=0.25):
        self.canvas_size = canvas_size
        self.digit_size = digit_size
        self.velocity_noise = velocity_noise
        self.pixel_noise = pixel_noise

        fast_ai_path = untar_data(URLs.MNIST)
        self.mnist_files = get_image_files(fast_ai_path/'training')

    @staticmethod
    def resize(img, size=(8, 8)):
        """Resize an image using nearest neighbor interpolation"""
        return cv2.resize(img, dsize=size, interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def noise(shape, std=0.05):
        """Create white noise of shape"""
        return np.random.normal(0, std, size=shape)

    def get_rand_digit(self, size=(12, 12)):
        """
        Fetch a random digit from FastAI's MNIST database, and resize
        to desired size
        """
        img = np.array(load_image(choice(self.mnist_files)))
        # img = np.array(load_image(self.mnist_files[1]))
        # img = np.ones(size) * 255
        img = img / 255
        digit = self.resize(img, size=size)
        return digit

    def get_random_trajectory(self, seq_length):
        "Generate a random trajectory"

        inner_canvas = self.canvas_size - np.array(self.digit_size)

        # Random initial position
        x, y = np.random.random(2) * inner_canvas

        # Object moves one pixel in both x and y 
        v_x, v_y = choice([-1, 1]), choice([-1, 1])

        # x, y = 5, 5
        # v_x, v_y = 1, 1

        out_x, out_y = [], []
        
        for i in range(seq_length):

            # Random noise added to velocity
            v_y_noise, v_x_noise = np.random.normal(0, self.velocity_noise, 2)

            # Take a step along velocity.
            y += (v_y + v_y_noise)
            x += (v_x + v_x_noise)

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= inner_canvas[1]:
                x = inner_canvas[1]
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= inner_canvas[0]:
                y = inner_canvas[0]
                v_y = -v_y
            out_x.append(x)
            out_y.append(y)

        return np.array(out_x, dtype=np.uint8), np.array(out_y, dtype=np.uint8)

    def generate_moving_digit(self, n_frames):
        """Generate a sequence of images with a single digit moving about the canvas"""

        # Fetch a random digit
        digit = self.get_rand_digit(size=self.digit_size)

        # Generate the trajectory
        xs, ys = self.get_random_trajectory(n_frames)
        
        # Create canvas and superimpose the digit along the trajectory
        canvas = np.zeros((n_frames, *self.canvas_size), dtype=np.float16)
        for i,(x,y) in enumerate(zip(xs,ys)):
            canvas[i, y:(y+self.digit_size[1]),x:(x+self.digit_size[0])] = np.array(digit)
        return canvas

    def generate_moving_digits(self, n_frames, n_digits=1):
        """Generate a sequence of images with multiple digit moving about the canvas"""
        return np.stack(np.array([self.generate_moving_digit(n_frames) for n in range(n_digits)])).max(axis=0)

    @staticmethod
    def distance_to_border(shape):
        rows, cols = shape
        distance_array = np.zeros(shape)

        for i in range(rows):
            for j in range(cols):
                distance_array[i, j] = min(i, j, rows-i-1, cols-j-1)
                
        return distance_array

    def create_dataset(self, num_samples, input_timesteps, output_timesteps=1, n_digits=1, gap=0):
        """Create a dataset of arrays with channels (pixel_intensity, x_position, y_position)"""

        x = []
        y = []
        for _ in range(num_samples):
            imgs = self.generate_moving_digits(input_timesteps+output_timesteps+gap, n_digits)
            white_noise = self.noise((len(imgs), *self.canvas_size), std=self.pixel_noise)
            imgs = imgs + white_noise
            imgs = np.swapaxes(imgs, 1, -1)
            x.append(imgs[:input_timesteps])
            y.append(imgs[-output_timesteps:])

        x, y = np.array(x), np.array(y)
        x, y = np.expand_dims(x, -1), np.expand_dims(y, -1)

        return x, y

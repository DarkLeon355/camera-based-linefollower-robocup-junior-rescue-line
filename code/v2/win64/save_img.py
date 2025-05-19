import cv2
import time
import os

class save_img:
    def __init__(self):
        # Get the directory where this file is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.path_img_lines = os.path.join(base_dir, "edited")
        self.path_img = os.path.join(base_dir, "original")
        self.path_dir = os.path.join(base_dir, "directions.txt")
        self.count = 0
        os.makedirs(self.path_img_lines, exist_ok=True)
        os.makedirs(self.path_img, exist_ok=True)
        self.open_file()

    def open_file(self):
        # Check if the file exists and delete it
        if os.path.exists(self.path_dir):
            try:
                os.remove(self.path_dir)
                print(f"Deleted existing file: {self.path_dir}")
            except:
                while True:
                    print("Error 3: Could not delete the existing directions file")
                    time.sleep(5)

        # Create a new file
        try:
            self.f = open(self.path_dir, "a")
        except:
            while True:
                print("Error 3: Could not open the directions file")
                time.sleep(5)

    def save(self, img_lines, img, direction):
        cv2.imwrite(f"{self.path_img_lines}/img_{self.count}.png", img_lines)
        cv2.imwrite(f"{self.path_img}/img_{self.count}.png", img)
        self.f.write(f"{direction}\n")
        self.count += 1

    def close_file(self):
        try:
            self.f.close()
        except:
            while True:
                print("Error 4: Could not close the directions file")
                time.sleep(5)
import cv2
import numpy as np
import save_img
import time
#import motors #uncomment for non simulation use

class LineFollow:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
        self.saver = save_img.save_img()
        #self.motor = motors.Motors() #uncomment for non simulation use

        self.min_green_area = 400
        self.min_line_area = 100
        self.max_distance_between_cx_points = 50

    def handle_colors(self):
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([100, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((5, 5), np.uint8)
        self.green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        self.green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    def convert_to_binary(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        gray[self.green_mask > 0] = 255
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, self.threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    def find_junction(self):
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 5))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 150))
        horizontal = cv2.morphologyEx(self.threshold, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical = cv2.morphologyEx(self.threshold, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        joints = cv2.bitwise_and(horizontal, vertical)
        cnts, _ = cv2.findContours(joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            large_junctions = [cnt for cnt in cnts if cv2.contourArea(cnt) > 3000]
            if large_junctions:
                for cnt in large_junctions:
                    area = cv2.contourArea(cnt)
                    print(f"Junction area: {area}")
                    cv2.drawContours(self.output_frame, [cnt], -1, (0, 255, 0), 3)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        self.junction_center = (cx, cy)
                        cv2.circle(self.output_frame, self.junction_center, 5, (255, 0, 0), -1)

    def look_for_green(self):
        green_contours, _ = cv2.findContours(self.green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if green_contours:
            self.all_green = []
            for contour in green_contours:
                area = cv2.contourArea(contour)
                if area > self.min_green_area:
                    self.find_junction()
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        self.all_green.append((cx, cy))
                        cv2.circle(self.output_frame, (cx, cy), 5, (0, 255, 0), -1)
                    if len(self.all_green) > 1:
                        junction_y = self.junction_center[0][1]
                        self.all_green = [pt for pt in self.all_green if pt[1] >= junction_y]
    
    def process_line(self):
        height, width = self.threshold.shape
        y_levels = np.linspace(int(height * 0.05), int(height * 0.90), 15, dtype=int)
        self.cx_list = []
        prev_cx = 180

        for y in y_levels:
            y_start = max(0, y -2)
            y_end = min(height, y + 3)
            slice_line = self.threshold[y_start:y_end, :]
            contours, _ = cv2.findContours(slice_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > self.min_line_area:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            dist = abs(cx - prev_cx)
                            if dist < self.max_distance_between_cx_points:
                                self.cx_list.append(cx)
                                prev_cx = cx
                                cv2.circle(self.output_frame, (cx, y), 5, (255, 0, 0), -1)
                            else:
                                self.cx_list.append(prev_cx)
                                cv2.circle(self.output_frame, (prev_cx, y), 5, (0, 0, 255), -1)
                                

            if self.cx_list:
                self.cx_avg = sum(self.cx_list) // len(self.cx_list)

                cv2.circle(self.output_frame, (self.cx_avg, y), 5, (0, 255, 255), -1)
                #draw line from average slope




                    
                    
                        





    def run(self):
        while True:
            ret, self.frame = self.cam.read()
            self.output_frame = self.frame.copy()
            self.handle_colors()
            self.convert_to_binary()
            self.process_line()
            cv2.imshow("Line Following", self.output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

 
if __name__ == "__main__":
    robot = LineFollow()
    robot.run()


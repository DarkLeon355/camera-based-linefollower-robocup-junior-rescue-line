import cv2
import numpy as np
import save_img
import time
from scipy import signal
#import motors #uncomment for non simulation use

class LineFollow:
    def __init__(self):
        self.Kp = 0.6
        self.Ki = 0.0
        self.Kd = 0.2
        self.previous_error = 0
        self.integral = 0
        self.integral_max = 1000
        self.integral_min = -1000
        #self.motor = motors.Motors() #uncomment for non simulation use
        self.cx_avg = 0  # Initialize to avoid reference before assignment
        self.radians = 0  # Initialize to avoid reference before assignment
        self.k = 0  # Initialize slope
        self.d = 0  # Initialize intercept

        self.MIN_GREEN_DOT_AREA = 400
        self.MIN_LINE_AREA = 100
        self.MIN_JUNCTION_AREA = 3000  # or another suitable value
        self.JUNCTION_TOLERANCE_Y = 100 #move all cx to the junction center if within this tolerance
        self.center_tolerance_y = 30  # only turn if the junction is within this tolerance

        self.cap = cv2.VideoCapture(0)
        self.saver = save_img.save_img()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

    def process_frame(self, frame):
        output_frame = frame.copy()
        
        # 1. Extract green areas
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([100, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # 2. Check for green before proceeding with junction detection
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_dots = []
        has_significant_green = False
        
        for cnt in green_contours:
            area = cv2.contourArea(cnt)
            if area > self.MIN_GREEN_DOT_AREA:
                has_significant_green = True
                M_green = cv2.moments(cnt)
                if M_green["m00"] != 0:
                    cx_green = int(M_green["m10"] / M_green["m00"])
                    cy_green = int(M_green["m01"] / M_green["m00"])
                    green_dots.append((cx_green, cy_green))

        # 3. Create threshold image for line detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray[green_mask > 0] = 255
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

        # 4. Look for junctions ONLY if green is detected
        junction_center = None
        if has_significant_green:
            print("Green detected - searching for junction")
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 5))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 150))
            horizontal = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            joints = cv2.bitwise_and(horizontal, vertical)

            cnts, _ = cv2.findContours(joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                # Only consider contours with sufficient area
                large_junctions = [cnt for cnt in cnts if cv2.contourArea(cnt) > self.MIN_JUNCTION_AREA]
                if large_junctions:
                    for cnt in large_junctions:
                        area = cv2.contourArea(cnt)
                        print(f"Junction area: {area}")
                        cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 3)
                    largest_joint = max(large_junctions, key=cv2.contourArea)
                    M = cv2.moments(largest_joint)
                    if M["m00"] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        junction_center = (cx, cy)
                        cv2.circle(output_frame, junction_center, 15, (36, 255, 12), -1)

        # 5. Find line segments (always done regardless of green)
        height, width = threshold.shape
        y_levels = np.linspace(int(height * 0.05), int(height * 0.90), 10, dtype=int)
        cx_list = []
        prev_cx = width // 2

        for y in y_levels:
            y_start = max(0, y - 2)
            y_end = min(height, y + 3)
            slice_line = threshold[y_start:y_end, :]
            contours, _ = cv2.findContours(slice_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.MIN_LINE_AREA]
                if large_contours:
                    min_dist = float('inf')
                    best_cx = prev_cx
                    for cnt in large_contours:
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            test_cx = int(M["m10"] / M["m00"])
                            dist = abs(test_cx - prev_cx)
                            if dist < min_dist:
                                min_dist = dist
                                best_cx = test_cx

                    cx = best_cx

                    # Only adjust for junction if we detected one
                    if (junction_center is not None and 
                        abs(y - junction_center[1]) < self.JUNCTION_TOLERANCE_Y):
                        cx = junction_center[0]

                    cx_list.append(cx)
                    prev_cx = cx
                    cv2.circle(output_frame, (cx, y), 5, (0, 255, 0), -1)

        # 6. Calculate line parameters and PID
        if cx_list:
            self.cx_avg = sum(cx_list) // len(cx_list)
            frame_center = width // 2
            errors = [cx - frame_center for cx in cx_list]
            error = int(np.mean(errors))
            self.integral += error
            self.integral = max(min(self.integral, self.integral_max), self.integral_min)
            derivative = error - self.previous_error
            output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
            self.previous_error = error

            # Apply median filtering to cx values (less sensitive to outliers than mean)
            smoothed_cx = signal.medfilt(np.array(cx_list), 3)  # Window size 3
                
            self.k, self.d = np.polyfit(smoothed_cx, np.array(y_levels[:len(cx_list)]), 1)
            lower_func_point = int(self.k * 0 + self.d)
            upper_func_point = int(self.k * width + self.d)
            cv2.line(output_frame, (0, lower_func_point), (width, upper_func_point), (0, 165, 255), 2)  # Orange for fallback line
            self.radians = np.arctan(self.k)
                
        # 7. Logic for junctions and green dots
        if has_significant_green and junction_center:
            junction_x, junction_y = junction_center
            filtered_dots = [dot for dot in green_dots if dot[1] > junction_y]

            for dot in filtered_dots:
                cv2.circle(output_frame, dot, 5, (0, 0, 255), -1)

            if len(filtered_dots) == 2:
                print("Turn 180°!")
                print("Simulated: turn_around()")
                #self.motor.turn_around180()
            elif len(filtered_dots) == 1 and cx_list:
                cx_green, cy_green = filtered_dots[0]
                closest_idx = np.argmin(np.abs(np.array(y_levels[:len(cx_list)]) - cy_green))
                cx_at_green = cx_list[closest_idx]
                frame_center_y = height // 2

                if abs(junction_y - frame_center_y) < self.center_tolerance_y:
                    if cx_green < cx_at_green - 20:
                        print("Green dot left of the line: Turn left!")
                        print("Simulated: left()")
                        #self.motor.left90()
                    elif cx_green > cx_at_green + 20:
                        print("Green dot right of the line: Turn right!")
                        print("Simulated: right()")
                        #self.motor.right90()
                else: 
                    print("Not yet in the center of the junction")
                    #self.motor.run(width, self.cx_avg)
                    
            elif len(filtered_dots) == 0:
                print("No Green Dots below junction! Drive Forward!")
                print("Simulated: forward()")
                #self.motor.run(width, self.cx_avg)
        elif has_significant_green and not junction_center:
            print("Green detected but no junction found - continuing line following")
            #self.motor.run(width, self.cx_avg)
        elif cx_list:
            print("Standard line following - no green/junction")
            #self.motor.run(width, self.cx_avg)

        return threshold, output_frame, cx_list, green_mask

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        try:
            while True:
                time1 = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    print("Could not read frame.")
                    break

                threshold_img, visualized_frame, x_values, green_mask = self.process_frame(frame)

                print(f"Average cx: {self.cx_avg}\n")
                print(f"{np.degrees(self.radians):.2f} degrees")

                self.saver.save(visualized_frame, frame, self.radians)

                time2 = time.time()
                print(f"FPS: {1 / (time2 - time1):.2f}")

                cv2.imshow("Threshold", threshold_img)
                cv2.imshow("Green Mask", green_mask)
                cv2.imshow("Visualization", visualized_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")
            self.saver.close_file()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.saver.close_file()

if __name__ == "__main__":
    robot = LineFollow()
    robot.run()
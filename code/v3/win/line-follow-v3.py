import cv2
import numpy as np
import save_img
import time

class LineFollow:
    def __init__(self):
        # PID constants for line following
        self.Kp = 0.6
        self.Ki = 0.0
        self.Kd = 0.2
        self.previous_error = 0
        self.integral = 0

        # Minimum green dot area to filter noise
        self.MIN_GREEN_DOT_AREA = 400
        self.MIN_LINE_AREA = 100

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.saver = save_img.save_img()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

        self.turningflag = 0  # Add this if not present


    def process_frame(self, frame):
        # Convert to HSV to detect green
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([100, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        # Clean up the green mask using morphological operations
        kernel = np.ones((5, 5), np.uint8)  # Define a 5x5 kernel
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)  # Close gaps

        # Convert the original frame to grayscale for line detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mask out green areas in the grayscale image by setting them to white
        gray[green_mask > 0] = 255

        # Apply Gaussian blur to the grayscale image to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply a binary inverse threshold to highlight dark lines
        _, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)


        # Extract horizontal and vertical lines using morphology
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 5))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 150))
        horizontal = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Find junctions (intersections of horizontal and vertical lines)
        joints = cv2.bitwise_and(horizontal, vertical)

        # Find contours of the junctions and calculate centroid
        cnts, _ = cv2.findContours(joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        junction_center = None
        output_frame = frame.copy()
        if cnts:
            #draw the lines of the juction if they are found
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area > self.MIN_LINE_AREA:
                    cv2.drawContours(output_frame, [cnt], -1, (0, 255, 0), 3)
            largest_joint = max(cnts, key=cv2.contourArea)
            M = cv2.moments(largest_joint)
            if M["m00"] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                junction_center = (cx, cy)
                cv2.circle(output_frame, junction_center, 15, (36, 255, 12), -1)

        # PID control to follow the line based on sampled points
        height, width = threshold.shape
        y_levels = np.linspace(int(height * 0.05), int(height * 0.90), 15, dtype=int)
        cx_list = []
        prev_cx = width // 2  # Start from the center
        JUNCTION_TOLERANCE_Y = 25  # vertical tolerance in pixels
        JUNCTION_TOLERANCE_X = 25  # horizontal tolerance in pixels

        for y in y_levels:
            y_start = max(0, y - 2)
            y_end = min(height, y + 3)
            slice_line = threshold[y_start:y_end, :]
            contours, _ = cv2.findContours(slice_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cx = prev_cx  # Default to previous cx
            if contours:
                # Filter by area
                large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.MIN_LINE_AREA]
                if large_contours:
                    # Find contour whose centroid is closest to prev_cx
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
            # --- Force cx to junction center if within tolerance of the junction in both axes ---
            if (
                junction_center is not None and
                abs(y - junction_center[1]) < JUNCTION_TOLERANCE_Y and
                abs(cx - junction_center[0]) < JUNCTION_TOLERANCE_X
            ):
                cx = junction_center[0]
            cx_list.append(cx)
            prev_cx = cx  # Update for next slice
            cv2.circle(output_frame, (cx, y), 5, (0, 255, 0), -1)

        # Calculate PID output
        frame_center = width // 2
        errors = [cx - frame_center for cx in cx_list]
        error = int(np.mean(errors))
        self.integral += error
        derivative = error - self.previous_error
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.previous_error = error

        # Detect green dots and act
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_dots = []
        for cnt in green_contours:
            area = cv2.contourArea(cnt)
            if area > self.MIN_GREEN_DOT_AREA:
                M_green = cv2.moments(cnt)
                if M_green["m00"] != 0:
                    cx_green = int(M_green["m10"] / M_green["m00"])
                    cy_green = int(M_green["m01"] / M_green["m00"])
                    green_dots.append((cx_green, cy_green))


        if junction_center:
            junction_x, junction_y = junction_center
            filtered_dots = []
            for dot in green_dots:
                if dot[1] > junction_y:
                    filtered_dots.append(dot)
            if len(filtered_dots) > 0:
                for dot in filtered_dots: cv2.circle(output_frame, dot, 5, (0, 0, 255), -1)
            if len(filtered_dots) == 2 and self.turningflag == 0:
                print("Turn 180°!")
                # self.motor.turn_around()
                print("Simulated: turn_around()")
            elif len(filtered_dots) == 1 and self.turningflag == 0:
                cx_green, cy_green = filtered_dots[0]
                closest_idx = np.argmin(np.abs(np.array(y_levels) - cy_green))
                cx_at_green = cx_list[closest_idx]
                if cx_green < cx_at_green - 20:
                    print("Green dot left of the line: Turn left!")
                    # self.motor.left()
                    print("Simulated: left()")
                elif cx_green > cx_at_green + 20:
                    print("Green dot right of the line: Turn right!")
                    # self.motor.right()
                    print("Simulated: right()")
            elif len(filtered_dots) == 0:
                print("No Green Dots! Drive Forward!.")
                # self.motor.forward()
                print("Simulated: forward()")

        k, d = np.polyfit(np.array(cx_list), np.array(y_levels), 1)
        # y = k * x + d

        lower_func_point = int(k * 0 + d)
        upper_func_point = int(k * width + d)
        cv2.line(output_frame, (0, lower_func_point), (width, upper_func_point), (255, 0, 0), 2)


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

                cx_sum, valid_cx_count = 0, 0
                _, width, _ = frame.shape
                for cx in x_values:
                    if cx != width // 2:
                        cx_sum += cx
                        valid_cx_count += 1
                cx_avg = cx_sum // valid_cx_count if valid_cx_count != 0 else 0

                print(f"Average cx: {cx_avg}\n")

                # motors.run(self.motor, width, cx_avg)
                print(f"Simulated: motors.run(width={width}, cx_avg={cx_avg})")

                self.saver.save(visualized_frame, frame, cx_avg)

                time2 = time.time()
                print(f"FPS: {1 / (time2 - time1):.2f}")

                cv2.imshow("Threshold", threshold_img) #disable this if no gui needed
                cv2.imshow("Green Mask", green_mask)
                cv2.imshow("Visualization", visualized_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'): #disable this if no gui needed
                    break

        except KeyboardInterrupt:
            print("Interrupted by user")
            self.saver.close_file()

        finally:
            self.cap.release()
            cv2.destroyAllWindows() #disable this if no gui needed
            self.saver.close_file()

if __name__ == "__main__":
    robot = LineFollow()
    robot.run()

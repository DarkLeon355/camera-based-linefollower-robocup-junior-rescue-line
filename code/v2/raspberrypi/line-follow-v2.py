import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations
import save_img # Import custom module for saving images
import time # Import time for fps calculations

# PID parameters for line following control
Kp = 0.6  # Proportional gain
Ki = 0.0  # Integral gain
Kd = 0.2  # Derivative gain

previous_error = 0  # Store previous error for derivative calculation
integral = 0       # Store accumulated error for integral calculation
prev_cx_green = None
GREEN_JUMP_TOLERANCE = 50  # Adjust as needed (pixels)
MIN_GREEN_DOT_AREA = 200  # Minimum area for a green dot to be considered valid

def set_motor_speed(left_speed, right_speed):
    # Dummy function
    return 0

def process_frame(frame):
    global previous_error, integral
    global prev_cx_green  # To remember the last green dot x

    # Convert the frame to HSV color space for green detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define lower and upper threshold for green color in HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask where green colors are white and others are black
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

    height, width = threshold.shape  # Get image dimensions
    # Choose 20 horizontal slices from 10% to 90% of the image height
    y_levels = np.linspace(int(height * 0.05), int(height * 0.90), 15, dtype=int)

    cx_list = []  # List to store x-coordinates of line centroids
    output_frame = frame.copy()  # Copy of the frame for visualization

    for y in y_levels:
        y_start = max(0, y - 2)  # Start row for the slice
        y_end = min(height, y + 3)  # End row for the slice
        slice_line = threshold[y_start:y_end, :]  # Get the slice from the thresholded image

        # Find all contours in this slice
        contours, _ = cv2.findContours(slice_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx = None

        if len(contours) == 0:
            cx = width // 2  # Default to center if no line is found
        elif len(contours) == 1:
            # Only one line, use its centroid
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
            else:
                cx = width // 2
        else:
            # Multiple lines: choose the one closest to the green dot (if available)
            if 'cx_green' in locals():
                min_dist = width
                for cnt in contours:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        candidate_cx = int(M["m10"] / M["m00"])
                        dist = abs(candidate_cx - cx_green)
                        if dist < min_dist:
                            min_dist = dist
                            cx = candidate_cx
                if cx is None:
                    cx = width // 2
            else:
                # If no green dot, pick the largest contour (most likely the main line)
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                else:
                    cx = width // 2

        cx_list.append(cx)
        cv2.circle(output_frame, (cx, y), 5, (0, 255, 0), -1)

        # For the same slice, check for green areas (e.g., circles)
        slice_green = green_mask[y_start:y_end, :]
        contours, _ = cv2.findContours(slice_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:  # Ignore small contours
                M_green = cv2.moments(cnt)
                if M_green["m00"] != 0:
                    cx_green = int(M_green["m10"] / M_green["m00"])  # Green centroid x
                    cy_green = int(M_green["m01"] / M_green["m00"]) + y_start  # Green centroid y (adjusted)
                    # Draw a blue circle for detected green object
                    cv2.circle(output_frame, (cx_green, cy_green), 8, (255, 0, 0), 2)

        cv2.line(output_frame, (width//2, 0), (cx_list[0], y_levels[0]), (255,0,0), 5)
        for i in range(len(cx_list) - 1):
            cv2.line(output_frame, (cx_list[i], y_levels[i]), (cx_list[i+1], y_levels[i+1]), (255,0,0), 5)

   
    # PID Control for line following
    frame_center = width // 2  # Center x-coordinate of the frame
    errors = [cx - frame_center for cx in cx_list]  # List of errors for each slice
    error = int(np.mean(errors))  # Average error

    integral += error  # Accumulate error for integral term
    derivative = error - previous_error  # Change in error for derivative term
    output = (Kp * error) + (Ki * integral) + (Kd * derivative)  # PID output
    previous_error = error  # Update previous error

    base_speed = 50  # Base speed for motors
    left_speed = base_speed - output  # Adjust left motor speed
    right_speed = base_speed + output  # Adjust right motor speed
    set_motor_speed(left_speed, right_speed)  # Set motor speeds (dummy on Windows)

    # Draw a red vertical line at the center of the frame for reference
    cv2.line(output_frame, (frame_center, 0), (frame_center, height), (0, 0, 255), 2)

    # After finding all contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_dots = []
        for cnt in green_contours:
            area = cv2.contourArea(cnt)
            if area > MIN_GREEN_DOT_AREA:
                M_green = cv2.moments(cnt)
                if M_green["m00"] != 0:
                    cx_green = int(M_green["m10"] / M_green["m00"])
                    cy_green = int(M_green["m01"] / M_green["m00"])
                    green_dots.append((cx_green, cy_green, cnt, area))

        if len(green_dots) > 2:
            print("Detected more than 2 green dots, checking for the lowest one...")
            #detected more than 2 green dots, the ones with the lowest y coordinate are the ones we want
            #there are 3 cases now: there is one lowest green dot at the left hand side of the line, -> turn left
            #there is one lowest green dot at the right hand side of the line, -> turn right
            #there are two lowest green dots, one at the left hand side of the line and one at the right hand side of the line -> turn 180 degrees
            #in order to differtiate between those dots we have to sorty them by their y values
            green_dots.sort(key=lambda x: x[1])  # Sort by y-coordinate
            # now we have to check if two of those dots are on the same height, if so we have to turn by 180 degrees
            if green_dots[0][1] - green_dots[1][1] < 20 or green_dots[1][1] - green_dots[0][1] < 20:
                # the lowest two dots are on the same height which means we have to turn by 180 degrees
                print("Two green lowest dots at the same height, Turn 180 degrees!")
                prev_cx_green = None  # Reset tracking after a 180
            
            else: #this means there is only one lowest green dot
                #check if the lowest green dot is on the left or right side of the line
                cx_green, cy_green, cnt, area = green_dots[0]
                closest_idx = np.argmin(np.abs(np.array(y_levels) - cy_green))
                cx_at_green = cx_list[closest_idx]
                if cx_green < cx_at_green - 20:
                    print("Green dot left of the line at this height: Turn left!")
                    # Insert your left turn code here
                elif cx_green > cx_at_green + 20:
                    print("Green dot right of the line at this height: Turn right!")
                    # Insert your right turn code here



        # 180-degree turn if two or more valid green dots
        if len(green_dots) == 2:
            print("Turn 180 degrees!")
            prev_cx_green = None  # Reset tracking after a 180
        elif len(green_dots) == 1:
            cx_green, cy_green, cnt, area = green_dots[0]
            
            # Find the y in y_levels closest to cy_green
            closest_idx = np.argmin(np.abs(np.array(y_levels) - cy_green))
            cx_at_green = cx_list[closest_idx]

            if cx_green < cx_at_green - 20:
                print("Green dot left of the line at this height: Turn left!")
                # Insert your left turn code here
            elif cx_green > cx_at_green + 20:
                print("Green dot right of the line at this height: Turn right!")
                # Insert your right turn code here
            else:
                print("Green dot on the line at this height: Go straight or special action!")
                # Insert your straight/special action code here
        else:
            print("Intersection but no green Element, drive forward")

    
    # Return thresholded image, visualization frame, centroid list, and green mask
    return threshold, output_frame, cx_list, green_mask

def main():
    cap = cv2.VideoCapture(0)  # Open the default camera
    saver = save_img.save_img()  # Use a different name than the module
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            time.sleep(1)
            time1 = time.time()
            ret, frame = cap.read()  # Read a frame from the camera
            if not ret:
                print("Could not read frame.")
                break

            # Process the frame for line and green detection
            threshold_img, visualized_frame, x_values, green_mask = process_frame(frame)
            cx_add = 0
            cx_mid = 0
            _, width, _ = frame.shape
            for cx in x_values:
                if cx != width // 2: #dont count values where no line was found
                    cx_add += cx
                    cx_mid += 1
            if cx_mid != 0: cx_avg = cx_add // cx_mid #no division by zero
            else: cx_avg = 0

            print(f"Average cx: {cx_avg}\n")
            saver.save(visualized_frame, frame, cx_avg)  # Use the new variable
            time2 = time.time()
            print(f"FPS: {1 / (time2 - time1):.2f}")

    
    except KeyboardInterrupt:
        print("Interrupted by user")
        saver.close_file()

    finally:
        cap.release()  # Release the camera

if __name__ == "__main__":
    main()  # Run the main function if this script is executed

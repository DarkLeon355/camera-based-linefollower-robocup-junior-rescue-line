import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations

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
    # Dummy function for Windows (does nothing, just a placeholder)
    return 0

def process_frame(frame):
    global previous_error, integral
    global prev_cx_green  # To remember the last green dot x

    # Convert the frame to HSV color space for green detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define lower and upper bounds for green color in HSV
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
    y_levels = np.linspace(int(height * 0.05), int(height * 0.95), 20, dtype=int)

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

        # 180-degree turn if two or more valid green dots
        if len(green_dots) >= 2:
            print("Turn 180 degrees!")
            prev_cx_green = None  # Reset tracking after a 180
        elif len(green_dots) == 1:
            cx_green, cy_green, cnt, area = green_dots[0]
            # Prevent drastic jumps
            if prev_cx_green is not None and abs(cx_green - prev_cx_green) > GREEN_JUMP_TOLERANCE:
                print(f"Green dot jump too large (from {prev_cx_green} to {cx_green}), ignoring this frame.")
                # Optionally, you can skip action or use prev_cx_green instead
                cx_green = prev_cx_green
            else:
                prev_cx_green = cx_green  # Update only if not a drastic jump

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

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()  # Read a frame from the camera
            if not ret:
                break

            # Process the frame for line and green detection
            threshold_img, visualized_frame, x_values, green_mask = process_frame(frame)

            # Show the thresholded image
            cv2.imshow("Threshold", threshold_img)
            # Show the frame with detected lines and green objects
            cv2.imshow("Line & Green Visualization", visualized_frame)
            # Show the green mask
            cv2.imshow("Green Mask", green_mask)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        cap.release()  # Release the camera
        cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":
    main()  # Run the main function if this script is executed

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os



class LineDetector:
    def __init__(self):
        # Line following parameters
        self.cx_list = []
        self.cx_avg = 0
        self.min_line_area = 100
        self.max_distance_between_cx_points = 50
        self.min_green_area = 1000
        
        # Storage for results
        self.all_green = []
        self.junction_center = []
        self.line1 = None
        self.line2 = None
        self.centroid = None
        
        # Image processing parameters
        self.blur_kernel = (21, 21)
        self.canny_low = 50
        self.canny_high = 50
        self.hough_threshold = 80
        
    def find_intersection(self):
        """Find intersection between two Hough lines stored in self.line1 and self.line2"""
        if self.line1 is None or self.line2 is None:
            return None
            
        rho1, theta1 = self.line1[0]
        rho2, theta2 = self.line2[0]

        a1, b1 = np.cos(theta1), np.sin(theta1)
        a2, b2 = np.cos(theta2), np.sin(theta2)
        c1, c2 = rho1, rho2

        determinant = a1 * b2 - a2 * b1
        if abs(determinant) < 1e-5:
            return None

        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant

        return int(x), int(y)

    def find_centroid(self, edges):
        """Calculate the centroid of an edge image using moments"""
        moments = cv.moments(edges)
        
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            self.centroid = (cx, cy)
            return cx, cy
        else:
            self.centroid = None
            return None
    
    def draw_centroid(self, image, label=True):
        """Draw the centroid on the provided image"""
        if self.centroid is None:
            return image
            
        cx, cy = self.centroid
        result = image.copy()
        cv.circle(result, (cx, cy), 15, (0, 255, 0), -1)  # Green fill
        cv.circle(result, (cx, cy), 17, (255, 255, 255), 2)  # White outline
        
        if label:
            cv.putText(result, "Centroid", (cx + 20, cy), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                      
        return result
    
    def process_image(self, image):
        """Process the image to extract edges, centroid, and lines"""
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Create color copy for visualization
        color_img = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        
        # Edge detection
        imgBlur = cv.GaussianBlur(gray, self.blur_kernel, 3)
        cannyEdge = cv.Canny(imgBlur, self.canny_low, self.canny_high)
        
        # Find centroid of edges
        self.find_centroid(cannyEdge)
        
        # Create visualization images
        centroid_img = self.draw_centroid(color_img)
        result_img = color_img.copy()
        intersection_img = color_img.copy()
        
        # Line detection
        lines = cv.HoughLines(cannyEdge, 1, np.pi/180, self.hough_threshold)
        
        # If no lines detected, return early with just centroid information
        if lines is None or len(lines) == 0:
            return {
                'original': color_img,
                'edges': cannyEdge,
                'centroid': centroid_img,
                'has_lines': False,
                'centroid_point': self.centroid
            }
        
        # Classify lines as horizontal or vertical
        h_lines = []
        v_lines = []
        k = 3000  # Line extension parameter
        h_threshold = np.pi / 8
        
        for curline in lines:
            rho, theta = curline[0]
            theta = theta % np.pi

            dhat = np.array([[np.cos(theta)], [np.sin(theta)]])
            d = rho * dhat
            lhat = np.array([[-np.sin(theta)], [np.cos(theta)]])
            p1 = d + k * lhat
            p2 = d - k * lhat
            p1_point = p1.astype(int)
            p2_point = p2.astype(int)
            p1 = (p1_point[0][0], p1_point[1][0])
            p2 = (p2_point[0][0], p2_point[1][0])

            if theta < h_threshold or theta > np.pi - h_threshold:
                h_lines.append((p1, p2, rho, theta))
            elif np.pi / 2 - h_threshold < theta < np.pi / 2 + h_threshold:
                v_lines.append((p1, p2, rho, theta))
        
        # Draw lines
        for line in h_lines:
            p1, p2, _, _ = line
            cv.line(result_img, p1, p2, (255, 255, 255), 2)  # White

        for line in v_lines:
            p1, p2, _, _ = line
            cv.line(result_img, p1, p2, (255, 0, 0), 2)  # Blue
        
        # Find intersections
        height, width = gray.shape
        intersections = []
        
        for h_line in h_lines:
            for v_line in v_lines:
                h_rho, h_theta = h_line[2], h_line[3]
                v_rho, v_theta = v_line[2], v_line[3]

                self.line1 = np.array([[h_rho, h_theta]])
                self.line2 = np.array([[v_rho, v_theta]])
                
                intersection = self.find_intersection()

                if intersection is not None:
                    x, y = intersection
                    if 0 <= x < width and 0 <= y < height:
                        intersections.append(intersection)
    

        all_x = 0
        all_y = 0
        filtered_intersections = []
        #calculate the centroid of intersection

        # Filter out points that are close to each other (distance < 50 pixels)
        min_dist = 50
        for pt in intersections:
            if not filtered_intersections:
                filtered_intersections.append(pt)
            else:
                if all(np.hypot(pt[0] - fpt[0], pt[1] - fpt[1]) >= min_dist for fpt in filtered_intersections):
                    filtered_intersections.append(pt)
        intersections[:] = filtered_intersections

        # Draw intersections
        for point in intersections:
            cv.circle(intersection_img, point, 10, (0, 255, 0), -1)  # Green
            cv.circle(intersection_img, point, 12, (255, 255, 255), 2)  # White outline     

        if intersections:
            for point in intersections:
                all_x = all_x + point[0]
                all_y = all_y + point[1]

            self.junction_centroid = (int(all_x / len(intersections)), int(all_y / len(intersections)))
            cv.circle(intersection_img, self.junction_centroid, 15, (0, 0, 255), -1)  # Red fill
        
        # Draw centroid on edges visualization
        canny_rgb = cv.cvtColor(cannyEdge, cv.COLOR_GRAY2BGR)
        if self.centroid is not None:
            cx, cy = self.centroid
            cv.circle(canny_rgb, (cx, cy), 15, (0, 255, 0), -1)
            cv.circle(canny_rgb, (cx, cy), 17, (255, 255, 255), 2)
            
            # Calculate line from bottom center to centroid
            k = (cy - height) / (cx - width // 2)  # Note: bottom is y=height
            d = cy - k * cx
            #print the angle in degrees
            angle = np.degrees(np.arctan2(cy - height, cx - (width // 2)))
            print(f"Angle to centroid: {angle:.2f} degrees")
            
            # Draw the actual line
            cv.line(canny_rgb, (width // 2, height), (cx, cy), (255, 0, 0), 2)
            
            # If you want to draw individual points along the line:
            start_x = min(cx, width // 2)
            end_x = max(cx, width // 2)
            for x in range(start_x, end_x + 1):
                y = int(k * x + d)
                if 0 <= y < height:
                    cv.circle(canny_rgb, (x, y), 3, (0, 165, 255), -1)  # Orange dots
        
        # Return all relevant data and images
        return {
            'original': color_img,
            'edges': cannyEdge,
            'edges_with_centroid': canny_rgb,
            'centroid': centroid_img,
            'lines': result_img,
            'intersections': intersection_img,
            'has_lines': True,
            'h_lines': h_lines,
            'v_lines': v_lines,
            'intersections_list': intersections,
            'centroid_point': self.centroid
        }
    
    def analyze_image_file(self, image_path):
        """Process an image file and visualize the results"""
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: Could not find image at {image_path}")
            return
        
        # Load image
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        # Process image
        results = self.process_image(img)
        
        # Visualize results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(221)
        plt.title('Original Image')
        plt.imshow(cv.cvtColor(results['original'], cv.COLOR_BGR2RGB))
        
        plt.subplot(222)
        plt.title('Canny Edges with Centroid')
        if 'edges_with_centroid' in results:
            plt.imshow(cv.cvtColor(results['edges_with_centroid'], cv.COLOR_BGR2RGB))
        else:
            plt.imshow(results['edges'], cmap='gray')
        
        plt.subplot(223)
        if results['has_lines']:
            plt.title(f'Horizontal ({len(results["h_lines"])}) and Vertical ({len(results["v_lines"])}) Lines')
            plt.imshow(cv.cvtColor(results['lines'], cv.COLOR_BGR2RGB))
        else:
            plt.title('No Lines Detected')
            plt.imshow(cv.cvtColor(results['centroid'], cv.COLOR_BGR2RGB))
        
        plt.subplot(224)
        if results['has_lines']:
            plt.title(f'Intersection Points ({len(results["intersections_list"])}) and Centroid')
            plt.imshow(cv.cvtColor(results['intersections'], cv.COLOR_BGR2RGB))
        else:
            plt.title('Centroid of Edges')
            plt.imshow(cv.cvtColor(results['centroid'], cv.COLOR_BGR2RGB))
        
        

        plt.tight_layout()
        plt.show()
        
        return results
        
    def get_line_following_error(self, img=None, roi_height_percent=40):
        """
        Calculate line following error based on centroid position
        Returns: (success, error, steering_value)
        """
        if img is not None:
            # Process new image
            if len(img.shape) == 3:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            else:
                gray = img
                
            # Edge detection
            imgBlur = cv.GaussianBlur(gray, self.blur_kernel, 3)
            cannyEdge = cv.Canny(imgBlur, self.canny_low, self.canny_high)
            
            # Create ROI mask for bottom portion of image
            height, width = cannyEdge.shape
            roi_mask = np.zeros_like(cannyEdge)
            roi_height = int(height * roi_height_percent / 100)
            roi_mask[height - roi_height:height, :] = 255
            
            # Apply ROI mask
            roi_edges = cv.bitwise_and(cannyEdge, roi_mask)
            
            # Find centroid of ROI edges
            self.find_centroid(roi_edges)
        
        # Calculate steering error
        if self.centroid is not None:
            cx, cy = self.centroid
            if img is not None:
                center_x = width // 2
            else:
                # Use previous image dimensions
                center_x = gray.shape[1] // 2
                
            error = cx - center_x
            steering = -0.5 * error  # P controller
            
            return True, error, steering
        else:
            return False, 0, 0

# Example usage
if __name__ == "__main__":
    detector = LineDetector()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    imgPath = os.path.join(script_dir, 'junc4.png')
    results = detector.analyze_image_file(imgPath)
    
    # Example of using the line follower functionality
    if results and 'centroid_point' in results and results['centroid_point']:
        success, error, steering = detector.get_line_following_error()
        print(f"Line following: Error = {error}, Steering = {steering}")

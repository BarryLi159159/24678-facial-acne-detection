import cv2
import numpy as np

class AcneDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.red_params = {
            'top_percentage': 0.01,  # Top 0.5% red pixels
        }

    def show_debug_image(self, title, image):
        if self.debug:
            cv2.imshow(title, image)
            cv2.waitKey(0)

    def find_red_regions(self, a_channel):
        """Find red regions and create a binary mask for the top 0.5% redness."""
        # Get the threshold value for the top 0.5% redness
        threshold = np.percentile(a_channel, 100 - (self.red_params['top_percentage'] * 100))
        
        # Create a binary mask
        red_mask = (a_channel >= threshold).astype(np.uint8) * 255
        return red_mask

    def filter_circular_regions(self, red_mask):
        """Filter regions based on circularity."""
        # Find contours in the binary mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circular_regions = []
        for contour in contours:
            # Calculate area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue  # Avoid division by zero
            
            # Circularity = (4π × Area) / (Perimeter²)
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            
            # Filter regions with circularity close to 1
            if 0.7 <= circularity <= 1.2:  # Allow some margin for irregular shapes
                circular_regions.append(contour)
        
        return circular_regions

    def visualize_red_detection(self, image):
        """Detect red regions and filter by circularity."""
        # Convert to LAB color space and extract the A channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)

        # Normalize the A channel for better visualization
        a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)

        # Find red regions
        red_mask = self.find_red_regions(a)
        self.show_debug_image("Red Mask", red_mask)

        # Filter regions based on circularity
        circular_regions = self.filter_circular_regions(red_mask)

        # Create output images for visualization
        result_image = image.copy()
        for contour in circular_regions:
            # Draw a bounding circle around each detected region
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(result_image, center, radius, (0, 255, 0), 20)  # Green circle for acne

        # Visualize the redness map
        redness_colormap = cv2.applyColorMap(a_normalized, cv2.COLORMAP_JET)

        return {
            'original': image,
            'redness_colormap': redness_colormap,
            'red_mask': red_mask,
            'result': result_image,
        }


    def analyze_image(self, image_path):
        """Analyze the image and display results."""
        # Read the input image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to read the image")

        # Resize the image if it's too large
        max_dimension = 800
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        # Detect and visualize redness
        results = self.visualize_red_detection(image)

        # Display results
        for title, img in results.items():
            self.show_debug_image(title, img)

if __name__ == "__main__":
    try:
        detector = AcneDetector(debug=True)
        image_path = input("Enter the image path: ")
        detector.analyze_image(image_path)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

import cv2
import numpy as np

class AcneDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.red_params = {
            'top_percentage': 0.005,  # Take the top 0.1% most red points
            'min_distance': 50        # Minimum distance between two red points
        }

    def show_debug_image(self, title, image):
        if self.debug:
            cv2.imshow(title, image)
            cv2.waitKey(0)

    def find_red_peaks(self, a_channel):
        """Find the most red regions in the A channel."""
        # Get image dimensions
        height, width = a_channel.shape
        total_pixels = height * width

        # Calculate the number of top red pixels to select
        num_peaks = int(total_pixels * self.red_params['top_percentage'])

        # Find the top N most red pixel positions
        flat_indices = np.argpartition(a_channel.ravel(), -num_peaks)[-num_peaks:]
        row_indices, col_indices = np.unravel_index(flat_indices, a_channel.shape)

        # Combine coordinates and values
        coords_and_values = list(zip(row_indices, col_indices, a_channel[row_indices, col_indices]))

        # Sort by value in descending order
        coords_and_values.sort(key=lambda x: x[2], reverse=True)

        # Use non-maximum suppression to filter peaks
        selected_peaks = []
        for y, x, val in coords_and_values:
            # Check if the point is too close to any already selected peak
            too_close = False
            for selected_y, selected_x, _ in selected_peaks:
                distance = np.sqrt((y - selected_y)**2 + (x - selected_x)**2)
                if distance < self.red_params['min_distance']:
                    too_close = True
                    break

            if not too_close:
                selected_peaks.append((y, x, val))

        return selected_peaks

    def visualize_red_detection(self, image):
        """Detect red regions using the LAB color space."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Normalize the A channel for visualization
        a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)

        # Create a pseudo-color map
        a_colormap = cv2.applyColorMap(a_normalized, cv2.COLORMAP_JET)

        # Find red peaks
        red_peaks = self.find_red_peaks(a)

        # Create copies for visualization
        result_image = image.copy()
        colormap_with_marks = a_colormap.copy()

        # Heatmap visualization
        heatmap = np.zeros_like(a_normalized)

        # Mark detected red peaks on images
        for y, x, val in red_peaks:
            # Draw circles on the result image
            cv2.circle(result_image, (x, y), 5, (0, 0, 255), 2)  # Red circle
            cv2.circle(result_image, (x, y), 1, (0, 255, 0), -1) # Green center

            # Draw circles on the colormap
            cv2.circle(colormap_with_marks, (x, y), 5, (255, 255, 255), 2)  # White circle
            cv2.circle(colormap_with_marks, (x, y), 1, (0, 255, 0), -1)     # Green center

            # Add a Gaussian distribution to the heatmap
            y_coords, x_coords = np.ogrid[-y:a_normalized.shape[0]-y, -x:a_normalized.shape[1]-x]
            mask = x_coords*x_coords + y_coords*y_coords <= 100
            heatmap[mask] = np.maximum(heatmap[mask], val)

        # Normalize and colorize the heatmap
        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # Create a transparent overlay
        alpha = 0.3
        overlay = cv2.addWeighted(image, 1-alpha, heatmap_color, alpha, 0)

        results = {
            'original': image,
            'a_channel': a_normalized,
            'a_colormap': a_colormap,
            'colormap_with_marks': colormap_with_marks,
            'heatmap': heatmap_color,
            'overlay': overlay,
            'result': result_image
        }

        return results

    def analyze_image(self, image_path):
        """Analyze the image and display processing steps."""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to read the image")

        # Resize the image if it's too large
        height, width = image.shape[:2]
        max_dimension = 800
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        # Get the processing results
        results = self.visualize_red_detection(image)

        # Display all processing steps
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

import cv2
import numpy as np

class AcneDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.red_params = {
            'top_percentage': 0.005,  # Increase to top 1% for redness
        }

    def show_debug_image(self, title, image):
        if self.debug:
            cv2.imshow(title, image)
            cv2.waitKey(0)

    def find_red_regions(self, a_channel):
        """Find red regions and expand the mask using dilation."""
        threshold = np.percentile(a_channel, 100 - (self.red_params['top_percentage'] * 100))
        red_mask = (a_channel >= threshold).astype(np.uint8) * 255

        # Expand the red mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)  # Expands redness regions
        return red_mask


    def visualize_red_and_texture(self, image):
        """Detect redness and texture, and combine them."""
        # Convert to LAB and grayscale
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize A channel
        a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)

        # Detect red regions
        red_mask = self.find_red_regions(a)
        self.show_debug_image("Red Mask (Expanded)", red_mask)


        # Find contours for acne detection
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_image = image.copy()
        for contour in contours:
            # Filter based on size
            if 5 < cv2.contourArea(contour) < 3000:  # Adjust max size to capture larger acne
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(result_image, center, radius, (0, 255, 0), 2)  # Green circle
                

        redness_colormap = cv2.applyColorMap(a_normalized, cv2.COLORMAP_JET)

        return {
            'original': image,
            'redness_colormap': redness_colormap,
            'red_mask': red_mask,

            'result': result_image,
        }

    def analyze_image(self, image_path):
        """Analyze the image and display results."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to read the image")

        max_dimension = 800
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        results = self.visualize_red_and_texture(image)

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

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from mpl_toolkits.mplot3d import Axes3D

class AcneDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.red_params = {
            'top_percentage': 0.01,  # Top 1% for redness
        }

    def show_debug_image(self, title, image):
        if self.debug:
            cv2.imshow(title, image)
            cv2.waitKey(0)

    def visualize_a_channel(self, image):
        """Create and display a color map for the a-channel of the LAB color space."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)
        
        # Normalize the a-channel
        a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply a color map to the a-channel
        a_colormap = cv2.applyColorMap(a_normalized, cv2.COLORMAP_JET)
        
        # Show the color map
        self.show_debug_image("A-Channel Colormap", a_colormap)
        return a_normalized

    def find_red_regions(self, a_channel):
        """Find red regions and expand the mask using dilation."""
        threshold = np.percentile(a_channel, 100 - (self.red_params['top_percentage'] * 100))
        red_mask = (a_channel >= threshold).astype(np.uint8) * 255

        # Expand the red mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)  # Expands redness regions
        return red_mask

    def calculate_gradient(self, a_channel):
        """Calculate the gradient magnitude of the a-channel."""
        grad_x = cv2.Sobel(a_channel, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(a_channel, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        self.show_debug_image("Gradient Magnitude", gradient_magnitude.astype(np.uint8))
        return gradient_magnitude

    def calculate_laplacian(self, a_channel):
        """Calculate the Laplacian of the a-channel to emphasize regions with high curvature."""
        laplacian = cv2.Laplacian(a_channel, cv2.CV_64F, ksize=5)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
        self.show_debug_image("Laplacian", laplacian.astype(np.uint8))
        return laplacian

    def detect_peaks(self, red_mask, gradient, laplacian, min_distance=5):
        """Detect peaks based on red mask, high gradient, and high curvature."""
        combined_score = red_mask * gradient * laplacian
        combined_score_blurred = cv2.GaussianBlur(combined_score, (5, 5), sigmaX=2, sigmaY=2)
        y_coords, x_coords = np.where(combined_score_blurred > np.percentile(combined_score_blurred, 99))
        peaks = list(zip(x_coords, y_coords))
        return peaks

    def plot_peaks(self, image, peaks):
        """Plot the detected peaks on the image."""
        for x, y in peaks:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        self.show_debug_image("Detected Peaks", image)

    def plot_3d_map(self, a_channel, block_size=3):
        """Plot a 3D map where x and y are pixel locations and z is the intensity."""
        height, width = a_channel.shape
        downsampled_height = height // block_size
        downsampled_width = width // block_size
        downsampled_a = cv2.resize(a_channel, (downsampled_width, downsampled_height), interpolation=cv2.INTER_AREA)
        x, y = np.meshgrid(np.arange(downsampled_width), np.arange(downsampled_height))
        
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        axis.plot_surface(x, y, downsampled_a, cmap='jet')
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Intensity")
        plt.colorbar(axis.plot_surface(x, y, downsampled_a, cmap='jet'), ax=axis, label='Intensity')
        plt.show()

def select_image():
    """Open a file dialog to select an image."""
    Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
    image_path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not image_path:
        raise Exception("No image selected")
    return image_path

if __name__ == "__main__":
    try:
        image_path = select_image()
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to load image")
        
        detector = AcneDetector(debug=True)
        a_channel = detector.visualize_a_channel(image)
        red_mask = detector.find_red_regions(a_channel)
        detector.show_debug_image("Red Mask (Top 1% Redness)", red_mask)
        gradient_magnitude = detector.calculate_gradient(a_channel)
        laplacian = detector.calculate_laplacian(a_channel)
        peaks = detector.detect_peaks(red_mask, gradient_magnitude, laplacian)
        detector.plot_peaks(image, peaks)
        detector.plot_3d_map(a_channel, block_size=3)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
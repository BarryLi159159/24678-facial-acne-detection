import cv2
import numpy as np
import dlib
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MouthDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("/Users/li/Desktop/cmu文件/cv/project/24678-facial-acne-detection-main/shape_predictor_68_face_landmarks.dat")

    def show_debug_image(self, title, image):
        if self.debug:
            cv2.imshow(title, image)
            cv2.waitKey(0)

    def mask_out_mouth(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        for face in faces:
            landmarks = self.predictor(gray, face)
            
            # Mouth points
            mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            mouth_mask = np.zeros_like(gray)
            cv2.fillPoly(mouth_mask, [np.array(mouth_points, dtype=np.int32)], 255)

            # Fill the gap between the lips
            hull = cv2.convexHull(np.array(mouth_points, dtype=np.int32))
            cv2.fillConvexPoly(mouth_mask, hull, 255)

            # Dilate the mask to ensure the lips are fully covered
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=2)

            self.show_debug_image("Mouth Mask", mouth_mask)

            # Invert the mask to mask out the mouth
            inverted_mouth_mask = cv2.bitwise_not(mouth_mask)
            masked_image = cv2.bitwise_and(image, image, mask=inverted_mouth_mask)
            self.show_debug_image("Masked Image", masked_image)

        return masked_image

class AcneDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.red_params = {
            'top_percentage': 0.005,  # Increase to top 0.5% for redness
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

    def detect_peaks(self, laplacian, red_mask, min_distance=5):
        """Detect peaks based on the Laplacian (second derivative) and filter by red mask."""
        laplacian_blurred = cv2.GaussianBlur(laplacian, (5, 5), sigmaX=2, sigmaY=2)
        y_coords, x_coords = np.where(laplacian_blurred > np.percentile(laplacian_blurred, 85))
        peaks = [(x, y) for x, y in zip(x_coords, y_coords) if red_mask[y, x] > 0]
        return peaks

    def plot_peaks(self, image, peaks):
        """Plot the detected peaks on the image."""
        for x, y in peaks:
            cv2.circle(image, (x, y), 10, (0, 255, 0), 2)  # Draw circles with radius 10 and thickness 2
        self.show_debug_image("Detected Peaks", image)

    def plot_3d_map(self, a_channel, block_size=3):
        """Plot a 3D map where x and y are pixel locations and z is the intensity."""
        # Apply Gaussian blur to smooth the a-channel
        a_channel_smoothed = cv2.GaussianBlur(a_channel, (5, 5), sigmaX=2, sigmaY=2)
        
        height, width = a_channel_smoothed.shape
        downsampled_height = height // block_size
        downsampled_width = width // block_size
        downsampled_a = cv2.resize(a_channel_smoothed, (downsampled_width, downsampled_height), interpolation=cv2.INTER_AREA)
        x, y = np.meshgrid(np.arange(downsampled_width), np.arange(downsampled_height))
        
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        axis.plot_surface(x, y, downsampled_a, cmap='jet')
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Intensity")
        plt.colorbar(axis.plot_surface(x, y, downsampled_a, cmap='jet'), ax=axis, label='Intensity')
        plt.show()

    def plot_3d_laplacian_map(self, laplacian, block_size=3):
        """Plot a 3D map where x and y are pixel locations and z is the Laplacian intensity."""
        height, width = laplacian.shape
        downsampled_height = height // block_size
        downsampled_width = width // block_size
        downsampled_laplacian = cv2.resize(laplacian, (downsampled_width, downsampled_height), interpolation=cv2.INTER_AREA)
        x, y = np.meshgrid(np.arange(downsampled_width), np.arange(downsampled_height))
        
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        axis.plot_surface(x, y, downsampled_laplacian, cmap='jet')
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Laplacian Intensity")
        plt.colorbar(axis.plot_surface(x, y, downsampled_laplacian, cmap='jet'), ax=axis, label='Laplacian Intensity')
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
        
        mouth_detector = MouthDetector(debug=True)
        masked_image = mouth_detector.mask_out_mouth(image)
        
        acne_detector = AcneDetector(debug=True)
        a_channel = acne_detector.visualize_a_channel(masked_image)
        red_mask = acne_detector.find_red_regions(a_channel)
        acne_detector.show_debug_image("Red Mask (Top 0.5% Redness)", red_mask)
        gradient_magnitude = acne_detector.calculate_gradient(a_channel)
        laplacian = acne_detector.calculate_laplacian(a_channel)
        peaks = acne_detector.detect_peaks(laplacian, red_mask)
        acne_detector.plot_peaks(masked_image, peaks)
        acne_detector.plot_3d_map(a_channel, block_size=5)
        acne_detector.plot_3d_laplacian_map(laplacian, block_size=5)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
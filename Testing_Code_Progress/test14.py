import cv2
import numpy as np
import dlib
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FaceMasker:
    def __init__(self, debug=True):
        self.debug = debug
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def show_debug_image(self, title, image):
        if self.debug:
            cv2.imshow(title, image)
            cv2.waitKey(0)

    def mask_out_mouth_and_eyes(self, image):
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
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=3)

            # Eye points
            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            eye_mask = np.zeros_like(gray)
            cv2.fillPoly(eye_mask, [np.array(left_eye_points, dtype=np.int32)], 255)
            cv2.fillPoly(eye_mask, [np.array(right_eye_points, dtype=np.int32)], 255)

            # Dilate the eye masks to ensure the eyes are fully covered
            eye_mask = cv2.dilate(eye_mask, kernel, iterations=3)

            # Combine mouth and eye masks
            combined_mask = cv2.bitwise_or(mouth_mask, eye_mask)
            #self.show_debug_image("Mouth and Eye Mask", combined_mask)

            # Invert the mask to mask out the mouth and eyes
            inverted_combined_mask = cv2.bitwise_not(combined_mask)
            masked_image = cv2.bitwise_and(image, image, mask=inverted_combined_mask)
            #self.show_debug_image("Masked Image", masked_image)

        return masked_image

    def mask_face(self, image):
        """Mask out the background and keep only the face region."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        face_mask = np.zeros_like(gray)
        for face in faces:
            cv2.rectangle(face_mask, (face.left(), face.top()), (face.right(), face.bottom()), 255, -1)
        
        # Apply the face mask to the image
        masked_face = cv2.bitwise_and(image, image, mask=face_mask)
        self.show_debug_image("Masked Face", masked_face)
        return masked_face

class AcneDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.red_params = {
            'top_percentage': 0.001,  # Increase to top 0.1% for redness
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
        red_mask = cv2.dilate(red_mask, kernel, iterations=3)  # Expands redness regions
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

    def eye_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        face = face_cascade.detectMultiScale(gray, 1.3, 4)

        for (x,y,w,h) in face:
            new_gray = gray[y:y+int(h/2), x:x+w]
            new_color = image[y:y+int(h/2), x:x+w]
   
            # After determine the face area, find eyes' position within it
            eyes = eye_cascade.detectMultiScale(new_gray)
            
            # draw rectangles 
            for (eye_x,eye_y,eye_w,eye_h) in eyes:
                cv2.rectangle(new_color,(eye_x,eye_y),(eye_x+eye_w,eye_y+eye_h),(0,255,255),2)
                # cv2.rectangle(new_color,(ex-int(ew/2),ey-int(eh/2)),(ex+int(ew*1.5),ey+int(eh*1.5)),(0,255,255),2)

                new_area = gray[y+eye_y+int(eye_h/2):y+eye_y+eye_h+10, x+eye_x:x+eye_x+eye_w+10]
                lower_bound = 130   # Lower intensity
                upper_bound = 160  # Upper intensity 
                # Create a mask for the intensity range
                mask = cv2.inRange(new_area, lower_bound, upper_bound)
                filtered_image = cv2.bitwise_or(new_area, new_area, mask=mask)
                contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                adjusted_contours = []
                for contour in contours:
                    adjusted_contour = contour + [x+eye_x,y+eye_y+int(eye_h/2)]  # Add offset to each point
                    adjusted_contours.append(adjusted_contour)
                sorted_contours = sorted(adjusted_contours, key=cv2.contourArea, reverse=True)
                cv2.drawContours(image, [sorted_contours[0]], -1, (0, 255, 0), 2)  # Green color, 2-pixel thickness


        cv2.imshow('new',filtered_image)
        # display the image with detected eyes
        cv2.imshow('Eyes Detection',image)
        cv2.waitKey(0)

    def calculate_gradient(self, a_channel):
        """Calculate the gradient magnitude of the a-channel."""
        grad_x = cv2.Sobel(a_channel, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(a_channel, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        #self.show_debug_image("Gradient Magnitude", gradient_magnitude.astype(np.uint8))
        return gradient_magnitude

    def calculate_laplacian(self, a_channel):
        """Calculate the Laplacian of the a-channel to emphasize regions with high curvature."""
        laplacian = cv2.Laplacian(a_channel, cv2.CV_64F, ksize=5)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
        #self.show_debug_image("Laplacian", laplacian.astype(np.uint8))
        return laplacian

    def detect_peaks(self, laplacian, red_mask, min_distance=5):
        """Detect peaks based on the Laplacian (second derivative) and filter by red mask."""
        laplacian_blurred = cv2.GaussianBlur(laplacian, (5, 5), sigmaX=2, sigmaY=2)
        y_coords, x_coords = np.where(laplacian_blurred > np.percentile(laplacian_blurred, 99))
        peaks = [(x, y) for x, y in zip(x_coords, y_coords) if red_mask[y, x] > 0]
        return peaks

    def plot_contours(self, image, red_mask):
        """Plot contours around the detected red regions."""
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Draw contours with green color and thickness 2
        self.show_debug_image("Detected Contours", image)

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
        #plt.colorbar(axis.plot_surface(x, y, downsampled_a, cmap='jet'), ax=axis, label='Intensity')
        #plt.show()

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
        #plt.colorbar(axis.plot_surface(x, y, downsampled_laplacian, cmap='jet'), ax=axis, label='Laplacian Intensity')
        #plt.show()

    def score_skin_condition(self, red_mask, a_channel):
        """Score the face skin condition from 0-100 based on the detected contours and average redness."""
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        # Calculate the average redness of the top 1 percent redness
        top_redness_values = a_channel[red_mask > 0]
        average_redness = np.mean(top_redness_values)

        # Score calculation: more contours and higher average redness result in a lower score
        score = max(0, 100 - num_contours * 2 - int(average_redness / 500))  # Adjust the multipliers as needed
        return score

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
        
        face_masker = FaceMasker(debug=True)
        masked_face = face_masker.mask_face(image)
        masked_image = face_masker.mask_out_mouth_and_eyes(masked_face)
        
        acne_detector = AcneDetector(debug=True)
        a_channel = acne_detector.visualize_a_channel(masked_image)
        red_mask = acne_detector.find_red_regions(a_channel)
        acne_detector.show_debug_image("Red Mask (Top 0.1% Redness)", red_mask)
        gradient_magnitude = acne_detector.calculate_gradient(a_channel)
        laplacian = acne_detector.calculate_laplacian(a_channel)
        peaks = acne_detector.detect_peaks(laplacian, red_mask)
        acne_detector.plot_contours(masked_image, red_mask)
        acne_detector.plot_3d_map(a_channel, block_size=5)
        acne_detector.plot_3d_laplacian_map(laplacian, block_size=5)
        acne_detector.eye_detection(image)
        
        # Score the skin condition
        score = acne_detector.score_skin_condition(red_mask, a_channel)
        print(f"Skin Condition Score: {score}")

        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
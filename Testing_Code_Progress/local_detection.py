import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


class PimpleDetector:
    def __init__(self):
        self.src = None
        self.clone_img = None
        self.mouse_down = False
        self.contours = []
        self.pts = []

    def find_pimples(self, img):
        # Split the image into BGR channels
        bgr = cv2.split(img)
        bw = bgr[1]  # Use green channel
        pimples_count = 0

        # Apply adaptive threshold
        bw = cv2.adaptiveThreshold(bw, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        bw = cv2.dilate(bw, None, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 150:
                min_rect = cv2.boundingRect(contour)
                x, y, w, h = min_rect
                img_roi = img[y:y + h, x:x + w]

                # Convert to HSV for color checking
                img_roi_hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
                color = cv2.mean(img_roi_hsv)

                if color[0] < 10 and color[1] > 70 and color[2] > 50:
                    (center, radius) = cv2.minEnclosingCircle(contour)

                    if radius < 20:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        pimples_count += 1

        # Put the count on the image
        cv2.putText(img, str(pimples_count), (50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("pimples detector", img)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self.contours = []
            self.pts = []

        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
            if len(self.pts) > 2:
                # Create mask from points
                mask = np.zeros(self.clone_img.shape[:2], dtype=np.uint8)
                points = np.array([self.pts])
                cv2.fillPoly(mask, points, 255)

                # Apply mask to image
                masked = np.full(self.clone_img.shape, (255, 255, 255), dtype=np.uint8)
                np.copyto(masked, self.clone_img, where=mask[:, :, None].astype(bool))

                self.clone_img = self.src.copy()
                self.find_pimples(masked)

        if self.mouse_down:
            if len(self.pts) > 2:
                cv2.line(self.clone_img, (x, y),
                         (self.pts[-1][0], self.pts[-1][1]),
                         (0, 255, 0), 2)

            self.pts.append((x, y))
            cv2.imshow("pimples detector", self.clone_img)

    def select_image(self):
        # Create tkinter window and hide it
        root = tk.Tk()
        root.withdraw()

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )

        return file_path if file_path else None

    def run(self):
        # Select image using file dialog
        image_path = self.select_image()
        if not image_path:
            print("No image selected")
            return False

        self.src = cv2.imread(image_path)
        if self.src is None:
            print("Error: Could not read image")
            return False

        self.clone_img = self.src.copy()
        cv2.namedWindow("pimples detector")
        cv2.setMouseCallback("pimples detector", self.on_mouse)
        cv2.imshow("pimples detector", self.src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = PimpleDetector()
    detector.run()
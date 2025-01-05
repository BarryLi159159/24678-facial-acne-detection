import cv2
import numpy as np
import dlib
from tkinter import Tk
from tkinter.filedialog import askopenfilename


class FaceMasker:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def resize_image(self, image, target_width=500):
        h, w = image.shape[:2]
        aspect_ratio = float(h) / float(w)
        target_height = int(target_width * aspect_ratio)
        return cv2.resize(image, (target_width, target_height))

    def mask_out(self, image):
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
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            mouth_mask = cv2.dilate(mouth_mask, kernel, iterations=3)

            # Eye points
            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            eye_mask = np.zeros_like(gray)
            cv2.fillPoly(eye_mask, [np.array(left_eye_points, dtype=np.int32)], 255)
            cv2.fillPoly(eye_mask, [np.array(right_eye_points, dtype=np.int32)], 255)

            # Dilate the eye masks to ensure the eyes are fully covered
            eye_mask = cv2.dilate(eye_mask, kernel, iterations=3)

            # Eyebrows points
            left_eyebrow_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)]
            right_eyebrow_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)]
            eyebrow_mask = np.zeros_like(gray)
            cv2.fillPoly(eyebrow_mask, [np.array(left_eyebrow_points, dtype=np.int32)], 255)
            cv2.fillPoly(eyebrow_mask, [np.array(right_eyebrow_points, dtype=np.int32)], 255)

            # Dilate the eyebrow masks to ensure the eyes are fully covered
            eyebrow_mask = cv2.dilate(eyebrow_mask, kernel, iterations=2)

            # ear points
            left_ear_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 8)]
            right_ear_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(9, 17)]
            ear_mask = np.zeros_like(gray)
            cv2.fillPoly(ear_mask, [np.array(left_ear_points, dtype=np.int32)], 255)
            cv2.fillPoly(ear_mask, [np.array(right_ear_points, dtype=np.int32)], 255)

            # Dilate the eyebrow masks to ensure the eyes are fully covered
            ear_mask = cv2.dilate(ear_mask, kernel, iterations=2)

            # Combine mouth and eye masks
            combined_mask = cv2.bitwise_or(mouth_mask, eye_mask)
            combined_mask = cv2.bitwise_or(combined_mask, eyebrow_mask)
            combined_mask = cv2.bitwise_or(combined_mask, ear_mask)
            # Invert the mask to mask out the mouth and eyes
            inverted_combined_mask = cv2.bitwise_not(combined_mask)
            masked_image = cv2.bitwise_and(image, image, mask=inverted_combined_mask)
            cv2.imshow("Mask", masked_image)
        return masked_image

    def mask_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        face_mask = np.zeros_like(gray)

        for face in faces:
            forehead_extension = int((face.bottom() - face.top()) * 0.25)
            top = max(0, face.top() - forehead_extension)
            cv2.rectangle(face_mask, (face.left(), top), (face.right(), face.bottom()), 255, -1)

        masked_face = cv2.bitwise_and(image, image, mask=face_mask)
        return masked_face


class AcneDetector:
    def __init__(self):
        self.red_params = {
            'top_percentage': 0.001
        }

    def combined_detection(self, image, masked_image):
        """结合两种方法的痘痘检测"""
        detected_regions = []

        # 方法1: HSV色彩空间检测
        bgr = cv2.split(masked_image)
        bw = bgr[1]
        bw = cv2.adaptiveThreshold(bw, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        bw = cv2.dilate(bw, None, iterations=1)

        contours1, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours1:
            area = cv2.contourArea(contour)
            if 20 < area < 150:
                min_rect = cv2.boundingRect(contour)
                x, y, w, h = min_rect
                img_roi = masked_image[y:y + h, x:x + w]

                img_roi_hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
                color = cv2.mean(img_roi_hsv)

                if color[0] < 10 and color[1] > 70 and color[2] > 50:
                    (center, radius) = cv2.minEnclosingCircle(contour)
                    if radius < 20:
                        detected_regions.append(('color', (x, y, w, h)))

        # 方法2: LAB色彩空间检测
        lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)
        a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)

        threshold = np.percentile(a_normalized, 100 - (self.red_params['top_percentage'] * 100))
        red_mask = (a_normalized >= threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.dilate(red_mask, kernel, iterations=3)

        contours2, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours2:
            x, y, w, h = cv2.boundingRect(contour)
            detected_regions.append(('red', (x, y, w, h)))

        # 合并重叠区域
        merged_regions = self.merge_overlapping_regions(detected_regions)

        # 分类和标记
        self.classify_and_mark_regions(image, masked_image, merged_regions)

        return len(merged_regions)

    def merge_overlapping_regions(self, regions):
        if not regions:
            return []

        boxes = [(x, y, x + w, y + h) for method, (x, y, w, h) in regions]
        merged = []

        while boxes:
            current = boxes.pop(0)
            current_merged = False

            i = 0
            while i < len(boxes):
                if self.boxes_overlap(current, boxes[i]):
                    current = self.merge_boxes(current, boxes.pop(i))
                    current_merged = True
                else:
                    i += 1

            if current_merged:
                boxes.append(current)
            else:
                x1, y1, x2, y2 = current
                merged.append((x1, y1, x2 - x1, y2 - y1))

        return merged

    def boxes_overlap(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        iou = intersection / float(area1 + area2 - intersection)

        return iou > 0.3

    def merge_boxes(self, box1, box2):
        return (min(box1[0], box2[0]),
                min(box1[1], box2[1]),
                max(box1[2], box2[2]),
                max(box1[3], box2[3]))

    def calculate_simi(self, image1, image2):
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def classify_and_mark_regions(self, original_image, masked_image, regions):
        ground_truth_images = [
            cv2.imread('papule.jpg'),
            cv2.imread('pustule.jpg'),
            cv2.imread('nodule.jpg')
        ]

        counts = {'papule': 0, 'pustule': 0, 'nodule': 0}

        for x, y, w, h in regions:
            roi = masked_image[y:y + h, x:x + w]
            if roi.size == 0:
                continue

            similarities = []
            for i, ref_img in enumerate(ground_truth_images):
                if ref_img is not None:
                    try:
                        similarity = self.calculate_simi(roi, ref_img)
                        similarities.append((i, similarity))
                    except Exception as e:
                        print(f"Error calculating similarity: {str(e)}")
                        continue

            if similarities:
                best_match = max(similarities, key=lambda x: x[1])[0]

                if best_match == 0:
                    color = (255, 0, 0)  # Blue for papule
                    counts['papule'] += 1
                elif best_match == 1:
                    color = (0, 255, 0)  # Green for pustule
                    counts['pustule'] += 1
                else:
                    color = (0, 0, 255)  # Red for nodule
                    counts['nodule'] += 1

                cv2.rectangle(original_image, (x, y), (x + w, y + h), color, 2)

        y_offset = 30
        for acne_type, count in counts.items():
            cv2.putText(original_image, f"{acne_type}: {count}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30

    def score_skin_condition(self, red_mask, a_channel):
        """Score the skin condition from 0-100."""
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        # Calculate average redness
        top_redness_values = a_channel[red_mask > 0]
        average_redness = np.mean(top_redness_values) if len(top_redness_values) > 0 else 0

        # Score calculation
        score = max(0, 100 - num_contours * 2 - int(average_redness / 500))
        return score


def select_image():
    """Open a file dialog to select an image."""
    Tk().withdraw()
    image_path = askopenfilename(title="Select an image",
                                 filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                                            ("All files", "*.*")])
    if not image_path:
        raise Exception("No image selected")
    return image_path


if __name__ == "__main__":
    try:
        # 选择并加载图片
        image_path = select_image()
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to load image")

        # 面部处理
        face_masker = FaceMasker()
        image = face_masker.resize_image(image)
        original_image = image.copy()
        masked_face = face_masker.mask_face(image)
        masked_image = face_masker.mask_out(masked_face)

        # 痘痘检测和分类
        acne_detector = AcneDetector()
        total_detections = acne_detector.combined_detection(original_image, masked_image)

        # 计算并显示皮肤状况评分
        lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)
        a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
        red_mask = (a_normalized >= np.percentile(a_normalized, 99.9)).astype(np.uint8) * 255
        score = acne_detector.score_skin_condition(red_mask, a_normalized)

        cv2.putText(original_image, f"Skin Score: {score}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 保存和显示结果
        output_path = "detection_results.png"
        cv2.imwrite(output_path, original_image)
        print(f"Saved detection results to {output_path}")

        cv2.imshow("Detection Results", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
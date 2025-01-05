import cv2

def capture_image_from_camera(output_filename = 'image.jpg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Unable to open camera")

    print("press the space bar to take a photo and press esc to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture image")
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1)
        if key == 27:
            print("Closing camera")
            break
        elif key == 32:
            cv2.imwrite(output_filename, frame)
            print("Image saved")
            break

    cap.release()
    cv2.destroyAllWindows()
    return output_filename

import cv2
import numpy as np

# Load the image you want to use as the replacement background
background_image = cv2.imread('image.jpg')
background_resized = cv2.resize(background_image, (640, 480))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

camera = cv2.VideoCapture(1)
camera.set(3, 640)
camera.set(4, 480)

while True:
    status, frame = camera.read()

    if status:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Flip both the frame and the mask horizontally
        frame = cv2.flip(frame, 1)
        hsv = cv2.flip(hsv, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Adjusted HSV range for red
        lower_bound = np.array([0, 100, 100])
        upper_bound = np.array([10, 255, 255])


        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # Invert the mask to select the background
        mask_inv = cv2.bitwise_not(mask)

        # Extract the region of interest from the frame
        roi = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask_inv)

        # Resize the background to match the frame size
        background_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

        # Extract the region of interest from the background
        background_roi = cv2.bitwise_and(background_resized, background_resized, mask=mask)

        # Combine the foreground and background using alpha blending
        final_output = cv2.addWeighted(roi, 1, background_roi, 1, 0)
        output_file.write(final_output)

        cv2.imshow('frame', final_output)
        cv2.imshow('mask', mask)

        code = cv2.waitKey(1)
        if code == 32:
            break

camera.release()
output_file.release()
cv2.destroyAllWindows()

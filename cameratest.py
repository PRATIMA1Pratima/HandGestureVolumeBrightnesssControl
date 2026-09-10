import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Error: Could not read camera frame.")
        break

    cv2.imshow("Laptop Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
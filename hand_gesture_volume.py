import cv2
import mediapipe as mp
import math
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities


# ============================================================
# PROJECT INFORMATION
# ============================================================

PROJECT_NAME = "Hand Gesture Volume & Brightness Control"
DEVELOPER_NAME = "Developed by Pratima"
GITHUB_NAME = "GitHub: PRATIMA1Pratima"

TEXT_COLOR = (0, 0, 0)


# ============================================================
# AUDIO SETUP
# ============================================================

devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume


# ============================================================
# MEDIAPIPE TASKS SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# CAMERA SETUP
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
# FULLSCREEN WINDOW
# ============================================================

WINDOW_NAME = "Hand Gesture Volume & Brightness Control"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ============================================================
# TERMINAL INFORMATION
# ============================================================

print("=" * 60)
print(PROJECT_NAME)
print(DEVELOPER_NAME)
print(GITHUB_NAME)
print("=" * 60)
print("Camera started successfully!")
print()
print("Thumb + Index  -> Volume")
print("Thumb + Middle -> Brightness")
print()
print("Press Q to exit")
print("=" * 60)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(mp_image)


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        thumb = hand_landmarks[4]
        index = hand_landmarks[8]
        middle = hand_landmarks[12]

        thumb_x = int(thumb.x * w)
        thumb_y = int(thumb.y * h)

        index_x = int(index.x * w)
        index_y = int(index.y * h)

        middle_x = int(middle.x * w)
        middle_y = int(middle.y * h)


        # ====================================================
        # VOLUME CONTROL
        # ====================================================

        volume_distance = math.hypot(
            index_x - thumb_x,
            index_y - thumb_y
        )

        volume_distance = max(
            30,
            min(200, volume_distance)
        )

        volume_percent = int(
            ((volume_distance - 30) / 170) * 100
        )

        min_db = volume.GetVolumeRange()[0]
        max_db = volume.GetVolumeRange()[1]

        volume_db = min_db + (
            volume_percent / 100
        ) * (max_db - min_db)

        volume.SetMasterVolumeLevel(
            volume_db,
            None
        )


        # ====================================================
        # BRIGHTNESS CONTROL
        # ====================================================

        brightness_distance = math.hypot(
            middle_x - thumb_x,
            middle_y - thumb_y
        )

        brightness_distance = max(
            30,
            min(200, brightness_distance)
        )

        brightness_percent = int(
            ((brightness_distance - 30) / 170) * 100
        )

        try:
            sbc.set_brightness(
                brightness_percent
            )
        except Exception:
            pass


        # ====================================================
        # HAND CONNECTION LINES
        # ====================================================

        cv2.line(
            frame,
            (thumb_x, thumb_y),
            (index_x, index_y),
            (255, 0, 0),
            4
        )

        cv2.line(
            frame,
            (thumb_x, thumb_y),
            (middle_x, middle_y),
            (0, 255, 0),
            4
        )


        # ====================================================
        # HAND POINTS
        # ====================================================

        cv2.circle(
            frame,
            (thumb_x, thumb_y),
            12,
            (0, 255, 255),
            -1
        )

        cv2.circle(
            frame,
            (index_x, index_y),
            12,
            (255, 0, 0),
            -1
        )

        cv2.circle(
            frame,
            (middle_x, middle_y),
            12,
            (0, 255, 0),
            -1
        )


        # ====================================================
        # VOLUME BAR
        # ====================================================

        bar_x = 50
        bar_y = 170
        bar_width = 40
        bar_height = 350

        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (
                bar_x + bar_width,
                bar_y + bar_height
            ),
            (70, 70, 70),
            -1
        )

        filled_height = int(
            bar_height *
            volume_percent / 100
        )

        cv2.rectangle(
            frame,
            (
                bar_x,
                bar_y + bar_height - filled_height
            ),
            (
                bar_x + bar_width,
                bar_y + bar_height
            ),
            (255, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "VOLUME",
            (35, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"{volume_percent}%",
            (35, 555),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2
        )


        # ====================================================
        # BRIGHTNESS BAR
        # ====================================================

        bright_bar_x = w - 90
        bright_bar_y = 170
        bright_bar_width = 40
        bright_bar_height = 350

        cv2.rectangle(
            frame,
            (
                bright_bar_x,
                bright_bar_y
            ),
            (
                bright_bar_x + bright_bar_width,
                bright_bar_y + bright_bar_height
            ),
            (70, 70, 70),
            -1
        )

        bright_filled_height = int(
            bright_bar_height *
            brightness_percent / 100
        )

        cv2.rectangle(
            frame,
            (
                bright_bar_x,
                bright_bar_y +
                bright_bar_height -
                bright_filled_height
            ),
            (
                bright_bar_x +
                bright_bar_width,
                bright_bar_y +
                bright_bar_height
            ),
            (0, 255, 255),
            -1
        )

        cv2.putText(
            frame,
            "BRIGHTNESS",
            (w - 165, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            TEXT_COLOR,
            2
        )

        cv2.putText(
            frame,
            f"{brightness_percent}%",
            (w - 120, 555),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TEXT_COLOR,
            2
        )


    # ========================================================
    # PROJECT TITLE
    # ========================================================

    cv2.putText(
        frame,
        PROJECT_NAME,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        TEXT_COLOR,
        2
    )

    cv2.putText(
        frame,
        DEVELOPER_NAME,
        (30, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        2
    )

    cv2.putText(
        frame,
        GITHUB_NAME,
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        2
    )


    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "Thumb + Index = Volume",
        (30, h - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TEXT_COLOR,
        2
    )

    cv2.putText(
        frame,
        "Thumb + Middle = Brightness",
        (w - 360, h - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        TEXT_COLOR,
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print()
print("Program closed successfully.")
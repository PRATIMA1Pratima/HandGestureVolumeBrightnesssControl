import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
import threading
import wave
import subprocess
from datetime import datetime

from pycaw.pycaw import AudioUtilities
import screen_brightness_control as sbc
import soundcard as sc
import imageio_ffmpeg


# ============================================================
# SETTINGS
# ============================================================

WINDOW_NAME = "HAND GESTURE CONTROL"

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 760

RECORD_FPS = 20.0

AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2

PROJECT_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# AUDIO / VOLUME SETUP
# ============================================================

devices = AudioUtilities.GetSpeakers()
volume_control = devices.EndpointVolume

min_volume = volume_control.GetVolumeRange()[0]
max_volume = volume_control.GetVolumeRange()[1]


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(
    PROJECT_FOLDER,
    "hand_landmarker.task"
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(
    options
)


# ============================================================
# RECORDING VARIABLES
# ============================================================

recording = False

video_writer = None

audio_recording = False

audio_thread = None

audio_chunks = []

audio_lock = threading.Lock()

current_avi = None
current_wav = None
current_mp4 = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def text(
    img,
    message,
    position,
    size=0.6,
    thickness=1,
    color=(255, 255, 255)
):
    cv2.putText(
        img,
        message,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def map_value(
    value,
    in_min,
    in_max,
    out_min,
    out_max
):
    value = max(
        in_min,
        min(value, in_max)
    )

    return (
        (value - in_min) *
        (out_max - out_min) /
        (in_max - in_min)
    ) + out_min


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(value, maximum)
    )


# ============================================================
# SYSTEM AUDIO RECORDING
# ============================================================

def record_system_audio():
    global audio_chunks

    try:

        # Get default Windows speaker
        speaker = sc.default_speaker()

        print()
        print("System audio device:")
        print(speaker.name)

        # Create loopback microphone for speaker output
        microphone = sc.get_microphone(
            id=str(speaker.name),
            include_loopback=True
        )

        print("System audio capture started.")

        with microphone.recorder(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS
        ) as recorder:

            while audio_recording:

                data = recorder.record(
                    numframes=1024
                )

                with audio_lock:
                    audio_chunks.append(
                        data.copy()
                    )

    except Exception as e:

        print()
        print("SYSTEM AUDIO ERROR:")
        print(e)
        print()
        print(
            "Windows speaker loopback could not be accessed."
        )


# ============================================================
# SAVE SYSTEM AUDIO
# ============================================================

def save_system_audio():

    global audio_chunks

    with audio_lock:

        if len(audio_chunks) == 0:

            print("No system audio was captured.")

            return False

        audio = np.concatenate(
            audio_chunks,
            axis=0
        )

    # Convert floating point audio
    # to 16-bit PCM

    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    audio_int16 = (
        audio * 32767
    ).astype(np.int16)

    try:

        with wave.open(
            current_wav,
            "wb"
        ) as wf:

            wf.setnchannels(
                AUDIO_CHANNELS
            )

            wf.setsampwidth(2)

            wf.setframerate(
                AUDIO_SAMPLE_RATE
            )

            wf.writeframes(
                audio_int16.tobytes()
            )

        print(
            "System audio saved:",
            current_wav
        )

        return True

    except Exception as e:

        print(
            "Audio saving error:",
            e
        )

        return False


# ============================================================
# START RECORDING
# ============================================================

def start_recording():

    global recording
    global audio_recording
    global audio_thread
    global audio_chunks
    global video_writer

    global current_avi
    global current_wav
    global current_mp4

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    current_avi = os.path.join(
        PROJECT_FOLDER,
        f"hand_gesture_recording_{timestamp}.avi"
    )

    current_wav = os.path.join(
        PROJECT_FOLDER,
        f"hand_gesture_recording_{timestamp}.wav"
    )

    current_mp4 = os.path.join(
        PROJECT_FOLDER,
        f"hand_gesture_recording_{timestamp}.mp4"
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"MJPG"
    )

    video_writer = cv2.VideoWriter(
        current_avi,
        fourcc,
        RECORD_FPS,
        (
            CANVAS_WIDTH,
            CANVAS_HEIGHT
        )
    )

    if not video_writer.isOpened():

        print(
            "ERROR: Could not start video recording."
        )

        video_writer = None

        return

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    with audio_lock:
        audio_chunks = []

    audio_recording = True

    audio_thread = threading.Thread(
        target=record_system_audio,
        daemon=True
    )

    audio_thread.start()

    recording = True

    print()
    print("========================================")
    print("          RECORDING STARTED")
    print("========================================")
    print("Video: 20 FPS")
    print("Audio: System / YouTube")
    print("Speed: REAL-TIME 1x")
    print("========================================")
    print()


# ============================================================
# CONVERT TO MP4
# ============================================================

def convert_to_mp4():

    global current_avi
    global current_wav
    global current_mp4

    if not os.path.exists(current_avi):

        print("Video file not found.")

        return

    if not os.path.exists(current_wav):

        print("Audio file not found.")

        return

    print()
    print("Combining video + system audio...")
    print("Please wait...")

    try:

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [

            ffmpeg,

            "-y",

            # Video
            "-i",
            current_avi,

            # Audio
            "-i",
            current_wav,

            # Video codec
            "-c:v",
            "libx264",

            "-preset",
            "medium",

            "-crf",
            "23",

            "-pix_fmt",
            "yuv420p",

            # Audio codec
            "-c:a",
            "aac",

            "-b:a",
            "192k",

            # Explicitly maintain normal frame timing
            "-vsync",
            "cfr",

            # Use video/audio duration
            "-shortest",

            current_mp4
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:

            print()
            print("========================================")
            print("       RECORDING COMPLETED")
            print("========================================")
            print()
            print("Final MP4:")
            print(current_mp4)
            print()

            # Remove temporary files

            try:

                os.remove(current_avi)
                os.remove(current_wav)

            except Exception:
                pass

        else:

            print()
            print("FFmpeg conversion failed.")
            print()
            print(result.stderr)

    except Exception as e:

        print(
            "MP4 conversion error:",
            e
        )


# ============================================================
# STOP RECORDING
# ============================================================

def stop_recording():

    global recording
    global audio_recording
    global video_writer
    global audio_thread

    recording = False

    # Stop audio first
    audio_recording = False

    if audio_thread is not None:

        audio_thread.join(
            timeout=2
        )

    # Stop video
    if video_writer is not None:

        video_writer.release()

        video_writer = None

    print()
    print("Recording stopped.")

    # Save audio
    audio_saved = save_system_audio()

    if audio_saved:

        convert_to_mp4()

    else:

        print(
            "MP4 was not created because "
            "system audio was unavailable."
        )


# ============================================================
# CIRCULAR INDICATOR
# ============================================================

def draw_circle_indicator(
    canvas,
    center,
    radius,
    percentage,
    title
):

    x, y = center

    percentage = clamp(
        percentage,
        0,
        100
    )

    # Background
    cv2.circle(
        canvas,
        (x, y),
        radius,
        (45, 45, 45),
        12
    )

    # Progress
    angle = int(
        360 * percentage / 100
    )

    cv2.ellipse(
        canvas,
        (x, y),
        (radius, radius),
        -90,
        0,
        angle,
        (100, 220, 100),
        12,
        cv2.LINE_AA
    )

    text(
        canvas,
        f"{int(percentage)}%",
        (x - 38, y + 10),
        0.8,
        2,
        (255, 255, 255)
    )

    text(
        canvas,
        title,
        (x - 50, y + radius + 35),
        0.55,
        1,
        (180, 180, 180)
    )


# ============================================================
# PROGRESS BAR
# ============================================================

def draw_progress_bar(
    canvas,
    x,
    y,
    width,
    height,
    percentage
):

    percentage = clamp(
        percentage,
        0,
        100
    )

    cv2.rectangle(
        canvas,
        (x, y),
        (x + width, y + height),
        (45, 45, 45),
        -1
    )

    filled_width = int(
        width * percentage / 100
    )

    cv2.rectangle(
        canvas,
        (x, y),
        (
            x + filled_width,
            y + height
        ),
        (100, 220, 100),
        -1
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print(
        "ERROR: Could not open camera."
    )

    landmarker.close()

    raise SystemExit


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    CANVAS_WIDTH,
    CANVAS_HEIGHT
)


# ============================================================
# MAIN LOOP
# ============================================================

video_timestamp_ms = 0

volume_value = 0
brightness_value = 20

try:

    while True:

        loop_start = time.perf_counter()

        ret, frame = cap.read()

        if not ret:

            print(
                "Camera frame error."
            )

            break

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        video_timestamp_ms += int(
            1000 / RECORD_FPS
        )

        result = landmarker.detect_for_video(
            mp_image,
            video_timestamp_ms
        )

        hand_detected = False

        # ----------------------------------------------------
        # HAND
        # ----------------------------------------------------

        if result.hand_landmarks:

            hand_detected = True

            landmarks = result.hand_landmarks[0]

            h, w = frame.shape[:2]

            points = []

            for landmark in landmarks:

                px = int(
                    landmark.x * w
                )

                py = int(
                    landmark.y * h
                )

                points.append(
                    (px, py)
                )

                cv2.circle(
                    frame,
                    (px, py),
                    4,
                    (100, 255, 100),
                    -1
                )

            # Hand connections

            for connection in (
                mp.tasks.vision
                .HandLandmarksConnections
                .HAND_CONNECTIONS
            ):

                start = connection.start
                end = connection.end

                p1 = points[start]
                p2 = points[end]

                cv2.line(
                    frame,
                    p1,
                    p2,
                    (100, 220, 100),
                    2
                )

            thumb = points[4]
            index = points[8]
            middle = points[12]

            # ------------------------------------------------
            # VOLUME
            # ------------------------------------------------

            volume_distance = distance(
                thumb,
                index
            )

            volume_value = map_value(
                volume_distance,
                30,
                220,
                0,
                100
            )

            volume_value = clamp(
                volume_value,
                0,
                100
            )

            db_value = map_value(
                volume_value,
                0,
                100,
                min_volume,
                max_volume
            )

            try:

                volume_control.SetMasterVolumeLevel(
                    db_value,
                    None
                )

            except Exception:
                pass

            cv2.line(
                frame,
                thumb,
                index,
                (255, 100, 100),
                3
            )

            # ------------------------------------------------
            # BRIGHTNESS
            # ------------------------------------------------

            brightness_distance = distance(
                thumb,
                middle
            )

            brightness_value = map_value(
                brightness_distance,
                30,
                220,
                0,
                100
            )

            brightness_value = clamp(
                brightness_value,
                0,
                100
            )

            try:

                sbc.set_brightness(
                    int(brightness_value)
                )

            except Exception:
                pass

            cv2.line(
                frame,
                thumb,
                middle,
                (100, 200, 255),
                3
            )

        # ====================================================
        # UI CANVAS
        # ====================================================

        canvas = np.zeros(
            (
                CANVAS_HEIGHT,
                CANVAS_WIDTH,
                3
            ),
            dtype=np.uint8
        )

        # Header

        text(
            canvas,
            "HAND GESTURE CONTROL",
            (40, 55),
            1.0,
            2,
            (255, 255, 255)
        )

        text(
            canvas,
            "AI-powered volume & screen brightness control",
            (42, 82),
            0.5,
            1,
            (160, 160, 160)
        )

        # Recording indicator

        if recording:

            cv2.circle(
                canvas,
                (1110, 45),
                8,
                (80, 80, 255),
                -1
            )

            text(
                canvas,
                "REC",
                (1130, 52),
                0.55,
                2,
                (255, 255, 255)
            )

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        camera_width = 640
        camera_height = 480

        resized_frame = cv2.resize(
            frame,
            (
                camera_width,
                camera_height
            )
        )

        camera_x = 280
        camera_y = 130

        canvas[
            camera_y:
            camera_y + camera_height,

            camera_x:
            camera_x + camera_width
        ] = resized_frame

        cv2.rectangle(
            canvas,
            (camera_x, camera_y),
            (
                camera_x + camera_width,
                camera_y + camera_height
            ),
            (80, 80, 80),
            2
        )

        # ----------------------------------------------------
        # VOLUME PANEL
        # ----------------------------------------------------

        cv2.rectangle(
            canvas,
            (30, 130),
            (250, 610),
            (25, 25, 25),
            -1
        )

        text(
            canvas,
            "VOLUME",
            (80, 165),
            0.65,
            2,
            (255, 255, 255)
        )

        draw_circle_indicator(
            canvas,
            (140, 260),
            80,
            volume_value,
            "SYSTEM"
        )

        draw_progress_bar(
            canvas,
            60,
            390,
            160,
            15,
            volume_value
        )

        text(
            canvas,
            "Thumb + Index",
            (73, 445),
            0.48,
            1,
            (160, 160, 160)
        )

        # ----------------------------------------------------
        # BRIGHTNESS PANEL
        # ----------------------------------------------------

        cv2.rectangle(
            canvas,
            (950, 130),
            (1170, 610),
            (25, 25, 25),
            -1
        )

        text(
            canvas,
            "BRIGHTNESS",
            (975, 165),
            0.52,
            2,
            (255, 255, 255)
        )

        draw_circle_indicator(
            canvas,
            (1060, 260),
            80,
            brightness_value,
            "SCREEN"
        )

        draw_progress_bar(
            canvas,
            980,
            390,
            160,
            15,
            brightness_value
        )

        text(
            canvas,
            "Thumb + Middle",
            (985, 445),
            0.48,
            1,
            (160, 160, 160)
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if hand_detected:

            status = "HAND DETECTED"

        else:

            status = "SHOW YOUR HAND"

        text(
            canvas,
            status,
            (480, 650),
            0.6,
            2,
            (255, 255, 255)
        )

        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        text(
            canvas,
            "R  Record / Stop",
            (40, 720),
            0.48,
            1,
            (150, 150, 150)
        )

        text(
            canvas,
            "Q  Exit",
            (220, 720),
            0.48,
            1,
            (150, 150, 150)
        )

        text(
            canvas,
            "Developed by Pratima",
            (470, 720),
            0.48,
            1,
            (180, 180, 180)
        )

        text(
            canvas,
            "GitHub: PRATIMA1Pratima",
            (890, 720),
            0.42,
            1,
            (150, 150, 150)
        )

        # ====================================================
        # WRITE VIDEO FRAME
        # ====================================================

        if recording and video_writer is not None:

            video_writer.write(canvas)

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            canvas
        )

        key = cv2.waitKey(1) & 0xFF

        # Q = EXIT

        if key == ord("q"):

            break

        # R = RECORD / STOP

        elif key == ord("r"):

            if not recording:

                start_recording()

            else:

                stop_recording()

        # ----------------------------------------------------
        # REAL-TIME 1X PACING
        # ----------------------------------------------------

        if recording:

            elapsed = (
                time.perf_counter()
                - loop_start
            )

            target_frame_time = (
                1.0 / RECORD_FPS
            )

            remaining = (
                target_frame_time
                - elapsed
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


# ============================================================
# CLEANUP
# ============================================================

finally:

    if recording:

        stop_recording()

    cap.release()

    landmarker.close()

    cv2.destroyAllWindows()

    print()
    print("Program closed.")
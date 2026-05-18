import io
import os
import tempfile
from typing import Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError


Box = Tuple[int, int, int, int]
Point = Tuple[float, float]

LEFT_EYE_LANDMARKS = (33, 133, 160, 158, 144, 153)
RIGHT_EYE_LANDMARKS = (362, 263, 385, 380, 373, 390)
MOUTH_LANDMARKS = (61, 291, 0, 17, 78, 308)

MAX_FACES = 10
MIN_DETECTION_CONFIDENCE = 0.5


@st.cache_resource(show_spinner=False)
def get_face_mesh():
    """Create the offline MediaPipe Face Mesh detector once per app session."""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=MAX_FACES,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )


def detect_faces(image: Image.Image) -> List[Sequence[Point]]:
    """Detect faces with MediaPipe Face Mesh and return normalized landmarks."""
    rgb_image = image.convert("RGB")
    image_array = np.asarray(rgb_image)

    # Normalize through OpenCV so MediaPipe receives a contiguous RGB array.
    bgr_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
    results = get_face_mesh().process(rgb_array)

    if not results.multi_face_landmarks:
        return []

    return [
        [(landmark.x, landmark.y) for landmark in face_landmarks.landmark]
        for face_landmarks in results.multi_face_landmarks
    ]


def _clamp(value: float, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(maximum, round(value))))


def _box_from_landmarks(
    landmarks: Sequence[Point],
    indices: Iterable[int],
    image_width: int,
    image_height: int,
    padding_pct: float,
) -> Box:
    points = [
        (landmarks[index][0] * image_width, landmarks[index][1] * image_height)
        for index in indices
    ]

    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    width = max_x - min_x
    height = max_y - min_y
    pad_x = width * padding_pct
    pad_y = height * padding_pct

    left = _clamp(min_x - pad_x, 0, image_width)
    top = _clamp(min_y - pad_y, 0, image_height)
    right = _clamp(max_x + pad_x, 0, image_width)
    bottom = _clamp(max_y + pad_y, 0, image_height)

    return left, top, max(0, right - left), max(0, bottom - top)


def get_eye_mouth_boxes(
    landmarks: Sequence[Point],
    image_width: int,
    image_height: int,
    padding_pct: float,
) -> List[Box]:
    """Extract eye and mouth bounding boxes from a single face landmark set."""
    landmark_groups = (
        LEFT_EYE_LANDMARKS,
        RIGHT_EYE_LANDMARKS,
        MOUTH_LANDMARKS,
    )

    boxes = [
        _box_from_landmarks(
            landmarks=landmarks,
            indices=indices,
            image_width=image_width,
            image_height=image_height,
            padding_pct=padding_pct,
        )
        for indices in landmark_groups
    ]

    return [box for box in boxes if box[2] > 0 and box[3] > 0]


def get_redaction_boxes(
    landmarks: Sequence[Point],
    image_width: int,
    image_height: int,
    padding_pct: float,
) -> List[Box]:
    """Backward-compatible alias for eye and mouth redaction boxes."""
    return get_eye_mouth_boxes(landmarks, image_width, image_height, padding_pct)


def apply_redaction(image: Image.Image, boxes: Sequence[Box], opacity: float) -> Image.Image:
    """Overlay semi-transparent black rectangles over the supplied regions."""
    base_image = image.convert("RGBA")
    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = int(255 * opacity)

    for x, y, width, height in boxes:
        draw.rectangle((x, y, x + width, y + height), fill=(0, 0, 0, alpha))

    return Image.alpha_composite(base_image, overlay)


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Serialize an image to PNG bytes without writing to disk."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def process_image(image: Image.Image, opacity: float, padding_percent: int):
    """Run face detection and redaction in-memory."""
    faces = detect_faces(image)
    image_width, image_height = image.size
    padding_pct = padding_percent / 100

    boxes: List[Box] = []
    for landmarks in faces:
        boxes.extend(
            get_redaction_boxes(
                landmarks=landmarks,
                image_width=image_width,
                image_height=image_height,
                padding_pct=padding_pct,
            )
        )

    redacted_image = apply_redaction(image, boxes, opacity) if boxes else image.convert("RGBA")
    return redacted_image, len(faces), len(boxes)


def configure_page() -> None:
    st.set_page_config(
        page_title="Local Face PII Redaction",
        page_icon=":shield:",
        layout="wide",
    )


def render_sidebar_controls() -> Tuple[float, int]:
    st.sidebar.header("Redaction Controls")
    opacity = st.sidebar.slider(
        "Box Opacity",
        min_value=0.5,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Controls how strongly the black redaction boxes obscure the image.",
    )
    padding = st.sidebar.slider(
        "Box Padding (%)",
        min_value=0,
        max_value=30,
        value=15,
        step=1,
        help="Expands each eye and mouth box beyond the detected landmarks.",
    )
    return opacity, padding


def load_uploaded_image(uploaded_file) -> Image.Image:
    try:
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    return image.convert("RGB")


def main() -> None:
    configure_page()

    st.title("Local Face PII Redaction Tool")
    st.write(
        "Upload an image to redact eye and mouth regions locally with MediaPipe Face Mesh."
    )

    opacity, padding = render_sidebar_controls()

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=("jpg", "jpeg", "png"),
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload a JPG, JPEG, or PNG image to begin.")
        return

    try:
        original_image = load_uploaded_image(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.subheader("Original Image")
    st.image(original_image, use_container_width=True)

    if not st.button("Process Image", type="primary"):
        return

    with st.spinner("Detecting faces and applying local redactions..."):
        try:
            redacted_image, face_count, box_count = process_image(
                image=original_image,
                opacity=opacity,
                padding_percent=padding,
            )
        except Exception as exc:
            st.error(f"Image processing failed: {exc}")
            return

    if face_count == 0:
        st.warning("No faces were detected. The output image is unchanged.")
    else:
        st.success(
            f"Detected {face_count} face{'s' if face_count != 1 else ''} "
            f"and applied {box_count} redaction boxes."
        )

    original_column, redacted_column = st.columns(2)
    with original_column:
        st.subheader("Original")
        st.image(original_image, use_container_width=True)

    with redacted_column:
        st.subheader("Redacted")
        st.image(redacted_image, use_container_width=True)

    st.download_button(
        label="Download Redacted Image",
        data=image_to_png_bytes(redacted_image),
        file_name="redacted_image.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()

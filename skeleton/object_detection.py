import warnings
import torch
import cv2
from PIL import Image
import numpy as np
import os
import sys
import contextlib
import tempfile

# ==============================================================
# Hard-silence context: captures even subprocess output
# ==============================================================

@contextlib.contextmanager
def hard_silence():
    """
    Suppress *all* output (stdout/stderr) including from subprocesses,
    torch.hub, and pip. Works on Windows, macOS, Linux.
    """
    # Save original file descriptors
    sys_stdout = sys.stdout
    sys_stderr = sys.stderr

    # Open a null device to redirect everything
    devnull = open(os.devnull, 'w')

    # Duplicate file descriptors for stdout and stderr
    fd_stdout = os.dup(1)
    fd_stderr = os.dup(2)

    try:
        # Redirect Python-level output
        sys.stdout = devnull
        sys.stderr = devnull

        # Redirect OS-level file descriptors (for subprocesses)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

        # Suppress pip logging completely
        os.environ['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
        os.environ['PIP_NO_WARN_SCRIPT_LOCATION'] = '0'
        os.environ['PYTHONWARNINGS'] = 'ignore'

        yield
    finally:
        # Restore Python-level stdout/stderr
        sys.stdout = sys_stdout
        sys.stderr = sys_stderr

        # Restore OS-level file descriptors
        os.dup2(fd_stdout, 1)
        os.dup2(fd_stderr, 2)

        os.close(fd_stdout)
        os.close(fd_stderr)
        devnull.close()


# ==============================================================
# Model loading (now fully silent)
# ==============================================================

def load_model(model_name='yolov5s', force_reload=False):
    """
    Load a YOLOv5 model quietly from Ultralytics or local cache.
    No console noise at all (pip / torch suppressed).
    """
    try:
        with hard_silence():
            model = torch.hub.load(
                'ultralytics/yolov5',
                model_name,
                force_reload=force_reload,
                verbose=False
            )
    except Exception:
        with hard_silence():
            model = torch.hub.load(
                'ultralytics_yolov5_master',
                model_name,
                source='local',
                verbose=False
            )
    return model


# ==============================================================
# Core detection pipeline
# ==============================================================

def load_image(image_path):
    """Load an image using PIL."""
    return Image.open(image_path)


def perform_inference(model, image):
    """Run inference on the input image using the given YOLOv5 model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model(image)


def extract_predictions(results):
    """
    Extract object detection results as a pandas DataFrame.
    Prints only object names and confidences (clean output).
    """
    df = results.pandas().xyxy[0]
    if not df.empty:
        print("\n✅ Objects detected:")
        # TODO: Print the object names and confidences in the
        # format: " - <name>: <confidence>"
        for _, row in df.iterrows():
            try:
                name = row.get('name', row.get('label', 'unkown'))
                conf = float(row.get('confidence', row.get('conf', 0.0)))
                print(f" - {name}: {conf:.2f}")
            except Exception:
                continue
    else:
        print("\n⚠️ No objects detected.")
    return df


def convert_to_opencv(image):
    """Convert a PIL Image to an OpenCV-compatible BGR numpy array."""
    img_cv = np.array(image)
    return cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)


def draw_bounding_boxes(image_cv, predictions):
    """Draw bounding boxes and labels on the image using OpenCV."""
    if predictions.empty:
        return image_cv

    # TODO: Students should implement drawing bounding boxes and labels
    for _, row in predictions.iterrows():
        try:
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            label = f"{row.get('name', 'obj')} {float(row.get('confidence', 0.0)):.2f}"
            cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image_cv, label, (xmin, max(mint:=ymin-6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        except Exception:
            continue
    return image_cv


def display_with_opencv(image_cv, window_name='Detected Objects'):
    """Display an image with OpenCV until a key is pressed."""
    cv2.imshow(window_name, image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==============================================================
# New: print_bounding_boxes
# ==============================================================

def print_bounding_boxes(predictions):
    """
    Print bounding box coordinates for each detection.

    Expected input: a pandas DataFrame with columns ['xmin','ymin','xmax','ymax','name','confidence'].
    Output format (one line per detection):
      <label>: <xmin>,<ymin>,<xmax>,<ymax> (conf=<confidence>)
    """
    # If empty or None, still produce a clear message
    if predictions is None or getattr(predictions, "empty", True):
        print("\n⚠️ No bounding boxes to print.")
        return

    print("\n📦 Bounding boxes:")
    for _, row in predictions.iterrows():
        try:
            # TODO: Get the coordinates, label, and confidence from the row
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            label = row.get('name', row.get('label', 'object'))
            conf = float(row.get('confidence', row.get('conf', 0.0)))
            print(f"{label}: {x1},{y1},{x2},{y2} (conf={conf:.2f})")
        except Exception:
            # Fallback: print the raw row if formatting fails
            print(f"{row.to_dict()}")


# ==============================================================
# Main pipeline
# ==============================================================

def run_object_detection(img_path='objects.jpg', model_name='yolov5s'):
    """Main pipeline to perform object detection."""
    print("🔍 Loading model...")
    model = load_model(model_name)

    print("🖼️ Loading image...")
    img = load_image(img_path)

    print("🚀 Performing object detection...")
    results = perform_inference(model, img)

    predictions = extract_predictions(results)

    # Print bounding boxes (newly added feature)
    print_bounding_boxes(predictions)

    img_cv = convert_to_opencv(img)
    img_with_boxes = draw_bounding_boxes(img_cv, predictions)
    display_with_opencv(img_with_boxes)


# ==============================================================
# Entry point
# ==============================================================

if __name__ == '__main__':
    run_object_detection('../datasets/objects/objects.jpg', 'yolov5s')

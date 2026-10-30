import os
import cv2
import argparse
from deepface import DeepFace

# ================= CONFIGURATION =================
DEFAULT_DB_PATH = "./database"  # Default database path
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
# =================================================

def process_and_enroll_single_image(image_path, person_name, output_dir=DEFAULT_DB_PATH):
    """
    Takes a single input image, detects the primary face using DeepFace,
    crops it using the exact bounding box, and saves it to the database.
    """
    person_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image from path: {image_path}")
        return False

    h, w, _ = img.shape

    try:
        detected_faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend='opencv',
            enforce_detection=False,
            align=False
        )
    except Exception as e:
        print(f"[ERROR] DeepFace extraction failed for {image_path}: {e}")
        detected_faces = []

    if len(detected_faces) == 0:
        print(f"[WARNING] No face detected in '{image_path}'. Skipping.")
        return False

    # Select the most prominent face (largest area)
    max_area = 0
    best_box = None
    for face_obj in detected_faces:
        facial_area = face_obj.get("facial_area", {})
        fx = facial_area.get("x", 0)
        fy = facial_area.get("y", 0)
        fw = facial_area.get("w", 0)
        fh = facial_area.get("h", 0)
        
        area = fw * fh
        if area > max_area:
            max_area = area
            best_box = (fx, fy, fw, fh)

    if best_box is None:
        print(f"[WARNING] No valid face bounding box found in '{image_path}'. Skipping.")
        return False

    fx, fy, f_w, f_h = best_box

    # Crop using the exact DeepFace bounding box coordinates
    crop_x1 = max(0, fx)
    crop_y1 = max(0, fy)
    crop_x2 = min(w, fx + f_w)
    crop_y2 = min(h, fy + f_h)

    face_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

    if face_crop.size > 0:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        timestamp = int(cv2.getTickCount())
        output_filename = os.path.join(person_dir, f"{person_name}_{timestamp}.jpg")
        cv2.imwrite(output_filename, face_crop)
        print(f"[INFO] Successfully enrolled: {output_filename}")
        return True
    else:
        print(f"[ERROR] Face crop resulted in an empty image region for '{image_path}'.")
        return False

def process_input_path(input_path, person_name, output_dir=DEFAULT_DB_PATH):
    """
    Handles either a single image file or a batch directory of images.
    """
    if os.path.isfile(input_path):
        print(f"[INFO] Processing single image file...")
        process_and_enroll_single_image(input_path, person_name, output_dir)
        
    elif os.path.isdir(input_path):
        print(f"[INFO] Processing batch directory: '{input_path}'...")
        image_files = [
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if f.lower().endswith(SUPPORTED_EXTENSIONS)
        ]
        
        if not image_files:
            print(f"[WARNING] No supported image files found in directory: {input_path}")
            return

        success_count = 0
        for img_file in image_files:
            if process_and_enroll_single_image(img_file, person_name, output_dir):
                success_count += 1
                
        print(f"[INFO] Batch complete. Successfully enrolled {success_count} out of {len(image_files)} images for '{person_name}'.")
    else:
        print(f"[ERROR] Input path '{input_path}' is neither a valid file nor a directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enroll single face images or batch process a directory of images into the DeepFace database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to a raw input image file OR a directory containing multiple images for batch processing."
    )
    parser.add_argument(
        "--name", 
        type=str, 
        required=True, 
        help="Name of the person (creates or uses this subfolder inside the database)."
    )
    parser.add_argument(
        "--db", 
        type=str, 
        default=DEFAULT_DB_PATH, 
        help="Path to the local face database root directory."
    )
    
    args = parser.parse_args()
    process_input_path(args.input, args.name, args.db)
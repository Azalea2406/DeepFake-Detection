import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm
import sys

def process_frames(input_dir, output_dir, target_size=(224, 224)):
    """Process frames with comprehensive error handling"""
    try:
        # Initialize MTCNN detector
        detector = MTCNN()
        
        # Convert to absolute paths
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)
        
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        # Verify input directory structure
        required_folders = ['real', 'fake']
        for folder in required_folders:
            folder_path = os.path.join(input_dir, folder)
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Required folder not found: {folder_path}")
            print(f"Found {folder} folder: {folder_path}")

        # Process both categories
        for category in required_folders:
            category_path = os.path.join(input_dir, category)
            output_category_path = os.path.join(output_dir, category)
            
            os.makedirs(output_category_path, exist_ok=True)
            
            # Get frame files with error handling
            try:
                frame_files = [f for f in os.listdir(category_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            except Exception as e:
                print(f"Error reading {category} files: {str(e)}")
                continue
                
            if not frame_files:
                print(f"Warning: No images found in {category} folder")
                continue
                
            print(f"\nProcessing {len(frame_files)} {category} frames...")
            
            for frame_file in tqdm(frame_files, desc=category):
                try:
                    frame_path = os.path.join(category_path, frame_file)
                    image = cv2.imread(frame_path)
                    if image is None:
                        print(f"\nWarning: Could not read image {frame_path}")
                        continue
                        
                    # Convert to RGB after checking if image was loaded
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    faces = detector.detect_faces(image)
                    
                    if not faces:
                        print(f"\nWarning: No faces detected in {frame_file}")
                        continue
                        
                    for i, face in enumerate(faces):
                        x, y, w, h = face['box']
                        # Ensure coordinates are positive
                        x, y = max(0, x), max(0, y)
                        # Ensure box stays within image boundaries
                        w = min(image.shape[1] - x, w)
                        h = min(image.shape[0] - y, h)
                        
                        padding = int(0.1 * min(w, h))
                        x, y = max(0, x - padding), max(0, y - padding)
                        w = min(image.shape[1] - x, w + 2*padding)
                        h = min(image.shape[0] - y, h + 2*padding)
                        
                        face_crop = image[y:y+h, x:x+w]
                        face_resized = cv2.resize(face_crop, target_size)
                        
                        output_filename = f"{os.path.splitext(frame_file)[0]}_face{i}.jpg"
                        cv2.imwrite(os.path.join(output_category_path, output_filename),
                                   cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
                
                except Exception as e:
                    print(f"\nError processing {frame_file}: {str(e)}")
                    continue

    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Use absolute paths to avoid confusion
    desktop = os.path.join(os.environ['USERPROFILE'], 'OneDrive', 'Desktop')
    input_directory = os.path.join(desktop, 'dataset_frames')
    output_directory = os.path.join(desktop, 'dataset_faces')
    
    print("Starting face cropping process...")
    process_frames(input_directory, output_directory)
    print("\nProcessing completed!")
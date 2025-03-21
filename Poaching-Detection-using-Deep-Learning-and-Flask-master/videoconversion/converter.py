import os
import cv2

# Function to extract frames from a video
def extract_frames(video_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)

    # Extract frames until there are none left
    success, image = vidcap.read()
    count = 0
    while success:
        # Save the current frame as an image file
        image_file = os.path.join(output_dir, f"frame{count:04d}.jpg")
        cv2.imwrite(image_file, image)

        # Read the next frame
        success, image = vidcap.read()
        count += 1

    # Release the video file and close all windows
    vidcap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ")
    output_dir = input("Enter the path to the output directory: ")
    extract_frames(video_path, output_dir)

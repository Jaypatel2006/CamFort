import cv2
import os
import datetime
import time

def create_directory(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def get_video_filename(label_folder):
    """Generate unique filename with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(label_folder, f"gesture_{timestamp}.avi")

def draw_text(frame, text, position, color=(0, 255, 0), scale=0.7, thickness=2):
    """Helper function to draw text on frame"""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def main():
    # Create main data directory
    create_directory("data")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  # Frames per second
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    print("ASL Gesture Recording System")
    print("=" * 40)
    print("Instructions:")
    print("- Enter gesture label when prompted")
    print("- Press SPACE to start/stop recording")
    print("- Press 'q' to quit")
    print("- Press 'n' to change gesture label")
    print("=" * 40)
    
    # Get initial gesture label
    gesture_label = input("Enter gesture label (e.g., 'Hello', 'ThankYou'): ").strip()
    if not gesture_label:
        print("No label provided. Exiting...")
        cap.release()
        return
    
    # Create label folder
    label_folder = os.path.join("data", gesture_label)
    create_directory(label_folder)
    
    # Initialize recording state
    is_recording = False
    out = None
    recording_start_time = None
    video_count = len([f for f in os.listdir(label_folder) if f.endswith('.avi')])
    
    print(f"\nReady to record gestures for label: '{gesture_label}'")
    print(f"Existing videos in folder: {video_count}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Create overlay for status information
        overlay = frame.copy()
        
        # Add semi-transparent rectangle for text background
        cv2.rectangle(overlay, (10, 10), (frame_width - 10, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Display current label
        draw_text(frame, f"Label: {gesture_label}", (20, 40), (255, 255, 255))
        
        # Display recording status
        if is_recording:
            # Calculate recording duration
            recording_duration = time.time() - recording_start_time
            status_text = f"RECORDING - {recording_duration:.1f}s"
            status_color = (0, 0, 255)  # Red for recording
            
            # Add recording indicator (red circle)
            cv2.circle(frame, (frame_width - 40, 40), 15, (0, 0, 255), -1)
        else:
            status_text = "READY (Press SPACE to record)"
            status_color = (0, 255, 0)  # Green for ready
        
        draw_text(frame, status_text, (20, 70), status_color)
        
        # Display video count
        draw_text(frame, f"Videos saved: {video_count}", (20, 100), (255, 255, 255))
        
        # Display instructions at the bottom
        instructions = "SPACE: Record | Q: Quit | N: New Label"
        draw_text(frame, instructions, (20, frame_height - 20), (200, 200, 200), 0.5, 1)
        
        # Write frame to video if recording
        if is_recording and out is not None:
            out.write(frame)
        
        # Display the frame
        cv2.imshow('ASL Gesture Recording', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            print("\nQuitting...")
            if is_recording and out is not None:
                out.release()
                print(f"Saved recording before quitting")
            break
            
        elif key == ord(' '):  # Toggle recording
            if not is_recording:
                # Start recording
                video_filename = get_video_filename(label_folder)
                out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
                is_recording = True
                recording_start_time = time.time()
                print(f"\nStarted recording: {os.path.basename(video_filename)}")
            else:
                # Stop recording
                if out is not None:
                    out.release()
                    recording_duration = time.time() - recording_start_time
                    print(f"Stopped recording. Duration: {recording_duration:.1f} seconds")
                    video_count += 1
                is_recording = False
                out = None
                
        elif key == ord('n'):  # Change gesture label
            # Stop any ongoing recording
            if is_recording and out is not None:
                out.release()
                print("Stopped recording due to label change")
                is_recording = False
                out = None
            
            # Get new label
            print("\n" + "=" * 40)
            new_label = input("Enter new gesture label: ").strip()
            if new_label:
                gesture_label = new_label
                label_folder = os.path.join("data", gesture_label)
                create_directory(label_folder)
                video_count = len([f for f in os.listdir(label_folder) if f.endswith('.avi')])
                print(f"Changed to label: '{gesture_label}'")
                print(f"Existing videos in folder: {video_count}")
            else:
                print("No label provided. Keeping current label.")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 40)
    print("Recording session ended")
    print(f"Total videos saved for '{gesture_label}': {video_count}")
    
    # Show all recorded gestures summary
    print("\nSummary of all recorded gestures:")
    for label in os.listdir("data"):
        label_path = os.path.join("data", label)
        if os.path.isdir(label_path):
            video_files = [f for f in os.listdir(label_path) if f.endswith('.avi')]
            print(f"  {label}: {len(video_files)} videos")

if __name__ == "__main__":
    main()
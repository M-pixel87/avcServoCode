import jetson.inference
import jetson.utils
import time
import serial
import cv2  # Import OpenCV for key handling

pigsfly= 0
# Initialize serial communication (replace '/dev/ttyACM0' with your serial port)
ser = serial.Serial('/dev/ttyACM0', 9600)

# Initialize the object detection network
net = jetson.inference.detectNet(model="/home/uafs/Downloads/jetson-inference/python/training/detection/ssd/models/test_aone/ssd-mobilenet.onnx",
                                 labels="/home/uafs/Downloads/jetson-inference/python/training/detection/ssd/models/test_aone/labels.txt",
                                 input_blob="input_0",
                                 output_cvg="scores",
                                 output_bbox="boxes",
                                 threshold=0.5)

# Initialize camera input
camera = jetson.utils.videoSource("/dev/video0")  # Camera input

# Initialize display output
display = jetson.utils.videoOutput()  # Display output

# Define action values (you can modify these based on your setup)
zwii = 2  # Move right
eins = 1   # Move left

# Main loop
while True:
    # Capture frame from camera
    img = camera.Capture()

    # Run object detection
    detections = net.Detect(img)

    # Render image to display
    display.Render(img)

    # Get image width and height
    width = img.width
    height = img.height

    # If detections are found, handle them
    if detections:
        for detect in detections:
            # Get object details
            ID = detect.ClassID
            top = int(detect.Top)
            left = int(detect.Left)
            bottom = int(detect.Bottom)
            right = int(detect.Right)
            item = net.GetClassDesc(ID)
            w = right - left  # Object width
            objx = left + (w / 2)  # Object center (x)

            # Print object details (debugging)
            print(f"Object: {item}, Off center by: {objx - width / 2}, Width of: {w}")

            # Check for specific object (blue_bucket in this case)
            if item == 'blue_bucket':
                # Calculate error in pan (how far the object is from the center)
                errorPan = objx - width / 2

                # If the object is significantly off-center, move towards it
                if abs(errorPan) > 50 and pigsfly==0 :
                    if errorPan > 0:
                        # Move right
                        ser.write(f"{zwii}\n".encode())
                        print(f"AI alignment action, sent UART value: {zwii}")
                    elif errorPan < 0:
                        # Move left
                        ser.write(f"{eins}\n".encode())
                        print(f"AI alignment action, sent UART value: {eins}")
    
    # Update display status (FPS of the object detection)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

    # Wait for 'q' key to exit (OpenCV used here)
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up and close serial port
ser.close()
camera.Close()

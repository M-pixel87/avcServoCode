import jetson.inference
import jetson.utils
import time
import serial
import cv2  

pigsfly= 0
ser = serial.Serial('/dev/ttyACM0', 9600)

net = jetson.inference.detectNet(model="/home/uafs/Downloads/jetson-inference/python/training/detection/ssd/models/test_aone/ssd-mobilenet.onnx",
                                 labels="/home/uafs/Downloads/jetson-inference/python/training/detection/ssd/models/test_aone/labels.txt",
                                 input_blob="input_0",
                                 output_cvg="scores",
                                 output_bbox="boxes",
                                 threshold=0.5)

camera = jetson.utils.videoSource("/dev/video0") 
display = jetson.utils.videoOutput()  
zwii = 2  
eins = 1   

while True:
    img = camera.Capture()
    detections = net.Detect(img)
    display.Render(img)
    width = img.width
    height = img.height
    if detections:
        for detect in detections:
            ID = detect.ClassID
            top = int(detect.Top)
            left = int(detect.Left)
            bottom = int(detect.Bottom)
            right = int(detect.Right)
            item = net.GetClassDesc(ID)
            w = right - left 
            objx = left + (w / 2) 

            print(f"Object: {item}, Off center by: {objx - width / 2}, Width of: {w}")

            if item == 'blue_bucket':
                errorPan = objx - width / 2
                if abs(errorPan) > 50 and pigsfly==0 :
                    if errorPan > 0:
                        ser.write(f"{zwii}\n".encode())
                        print(f"AI alignment action, sent UART value: {zwii}")
                    elif errorPan < 0:
                        ser.write(f"{eins}\n".encode())
                        print(f"AI alignment action, sent UART value: {eins}")
    
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))

    if cv2.waitKey(1) == ord('q'):
        break
ser.close()
camera.Close()

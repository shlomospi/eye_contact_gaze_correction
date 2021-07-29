import cv2
import argparse
parser = argparse.ArgumentParser(description="vu")
parser.add_argument("-n", '--name',
                        type=str,
                        default='basicvideo',
                        help='video name')
args = parser.parse_args()


cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(args.name+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
recording = False

while True:
    ret, frame = cap.read()
    if ret:

        cv2.imshow('frame', frame)

        if recording:
            writer.write(frame)

        k = cv2.waitKey(10)
        if k == ord('r'):
            if recording:
                print("stopped recording")
                recording = False
            else:
                print("Started recording")
                recording = True
        elif k == ord('q'):
            print("stopped recording. quitting")
            break
    else:
        break
cap.release()
writer.release()
cv2.destroyAllWindows()


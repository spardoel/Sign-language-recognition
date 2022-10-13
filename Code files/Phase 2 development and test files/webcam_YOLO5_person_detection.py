import cv2
import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5n")

font = cv2.FONT_HERSHEY_SIMPLEX  # The font used when displaying text on the image frame


class VideoCamera(object):
    def __init__(self):
        # create the Open CV video capture object
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):

        # Read the next frame from the camera
        _, fr = self.video.read()

        # Use the YOLO model to identify objects in the frame
        results = model(fr)

        # convert the results to pandas
        dataframe = results.pandas()

        # create an intermediate variable to hold the detected objects (for clarity)
        detected_objects = dataframe.xyxy[0].name

        # loop through each detected opbect
        for idx, object in enumerate(detected_objects):

            #  if the object is classified as a person
            if object == "person":

                # print the label to the frame
                cv2.putText(
                    fr, dataframe.xyxy[0].name[idx], (25, 25), font, 1, (255, 255, 0), 2
                )

                # extract the integer coordinates of the bounding box
                x1 = round(dataframe.xyxy[0].xmin[idx])
                y1 = round(dataframe.xyxy[0].ymin[idx])

                x2 = round(dataframe.xyxy[0].xmax[idx])
                y2 = round(dataframe.xyxy[0].ymax[idx])

                # Draw the bounding box around the object
                cv2.rectangle(fr, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # return the frame with the text and bounding box
        return fr


def main(camera):
    while True:

        frame = camera.get_frame()

        cv2.imshow("Sign language recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# run the main function and pass in the camera object
main(VideoCamera())

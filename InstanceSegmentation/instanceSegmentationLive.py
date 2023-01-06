import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2

capture = cv2.VideoCapture(0)

segment_video = instanceSegmentation()
segment_video.load_model("pointrend_resnet50.pkl", confidence=0.7, detection_speed = "rapid")
segment_video.process_camera(capture,  show_bboxes = True, frames_per_second=15, check_fps=True, show_frames= True,
frame_name= "frame", output_video_name="output_video.mp4")


"""
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2

capture = cv2.VideoCapture(0)

segment_video = instanceSegmentation()
segment_video.load_model("pointrend_resnet50.pkl")
segment_video.process_camera(capture,  show_bboxes = True, frames_per_second= 5, extract_segmented_objects=True, extract_from_box=True,
save_extracted_objects=True, show_frames= True,frame_name= "frame", output_video_name="output_video.mp4")"""
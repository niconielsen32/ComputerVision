import pixellib
from pixellib.semantic import semantic_segmentation

segment_image = semantic_segmentation()
segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
segment_image.segmentAsAde20k("sample.jpg", overlay = True, output_image_name="image_new.jpg")


"""
import pixellib
from pixellib.semantic import semantic_segmentation

segment_video = semantic_segmentation()
segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
segment_video.process_video_ade20k("sample_video.mp4", overlay = True, frames_per_second= 15, output_video_name="output_video.mp4")  """
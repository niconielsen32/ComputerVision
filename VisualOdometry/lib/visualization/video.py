import cv2
import numpy as np

from lib.visualization.image import put_text, draw_matches

def play_trip(l_frames, r_frames=None, lat_lon=None, timestamps=None, color_mode=False, waite_time=100, win_name="Trip"):
    l_r_mode = r_frames is not None

    if not l_r_mode:
        r_frames = [None]*len(l_frames)

    frame_count = 0
    for i, frame_step in enumerate(zip(l_frames, r_frames)):
        img_l, img_r = frame_step

        if not color_mode:
            img_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
            if img_r is not None:
                img_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)


        if img_r is not None:
            img_l = put_text(img_l, "top_center", "Left")
            img_r = put_text(img_r, "top_center", "Right")
            show_image = np.vstack([img_l, img_r])
        else:
            show_image = img_l
        show_image = put_text(show_image, "top_left", "Press ESC to stop")
        show_image = put_text(show_image, "top_right", f"Frame: {frame_count}/{len(l_frames)}")

        if timestamps is not None:
            time = timestamps[i]
            show_image = put_text(show_image, "bottom_right", f"{time}")


        if lat_lon is not None:
            lat, lon = lat_lon[i]
            show_image = put_text(show_image, "bottom_left", f"{lat}, {lon}")

        cv2.imshow(win_name, show_image)

        key = cv2.waitKey(waite_time)
        if key == 27:  # ESC
            break
        frame_count += 1
    cv2.destroyWindow(win_name)


def draw_matches_frame(img1, kp1, img2, kp2, matches):
    """
    Need to be call for each frame
    """
    matches = sorted(matches, key=lambda x: x.distance)
    vis_img = draw_matches(img1, kp1, img2, kp2, matches)
    cv2.imshow('Matches', vis_img)
    cv2.waitKey(100)
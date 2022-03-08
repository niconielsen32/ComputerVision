import cv2
import numpy as np


def put_text(image, org, text, color=(0, 0, 255), fontScale=0.7, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    if not isinstance(org, tuple):
        (label_width, label_height), baseline = cv2.getTextSize(text, font, fontScale, thickness)
        org_w = 0
        org_h = 0

        h, w, *_ = image.shape

        place_h, place_w = org.split("_")

        if place_h == "top":
            org_h = label_height
        elif place_h == "bottom":
            org_h = h
        elif place_h == "center":
            org_h = h // 2 + label_height // 2

        if place_w == "left":
            org_w = 0
        elif place_w == "right":
            org_w = w - label_width
        elif place_w == "center":
            org_w = w // 2 - label_width // 2

        org = (org_w, org_h)

    image = cv2.putText(image, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image


def draw_matches(img1, kp1, img2, kp2, matches):
    matches = sorted(matches, key=lambda x: x.distance)
    vis_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return vis_img


def show_images(images, window_name='Image', image_title=None):
    if len(images.shape) == 2:
        images = [images]

    for i, image in enumerate(images):
        image_c = image.copy()

        if image_c.dtype != np.uint8:
            if image_c.max() < 1.:
                image_c = image_c * 255
            image_c = image_c.astype(np.uint8)

        if len(image.shape) == 2:
            image_c = cv2.cvtColor(image_c, cv2.COLOR_GRAY2BGR)

        if image_title is None:
            image_title_show = f"{i}"
        else:
            image_title_show = image_title

        image_c = put_text(image_c, "top_center", image_title_show)
        cv2.imshow(window_name, image_c)
        cv2.waitKey(0)


def draw_face_boxs(image, faces, fontScale=0.7, text_thickness=1, line_thickness=1):
    imge_draw = image.copy()
    for i, (v, u, w, h) in enumerate(faces):
        put_text(imge_draw[u:u + h, v:v + w], "top_left", f"{i}", (0, 0, 255), fontScale=fontScale,
                 thickness=text_thickness)
        cv2.rectangle(imge_draw, (v, u), (v + w, u + h), (0, 0, 255), line_thickness)
    return imge_draw


def create_face_collage(image, faces, fontScale=0.5, text_thickness=1, face_size=(100, 100)):
    faces_sub = []
    for i, (v, u, w, h) in enumerate(faces):
        faces_sub.append(cv2.resize(image[u:u + h, v:v + w], dsize=face_size))

    rows = int(np.ceil(np.sqrt(len(faces_sub))))
    cols = int(np.ceil(len(faces_sub) / rows))

    sub_faces = np.zeros(shape=(face_size[0] * rows, face_size[1] * cols, 3), dtype=np.uint8)
    for i, face_sub in enumerate(faces_sub):
        c, r = i % cols, i // cols
        put_text(face_sub, "top_center", f"{i}", (0, 0, 255), fontScale=fontScale, thickness=text_thickness)
        sub_faces[face_size[0] * r:face_size[0] * (r + 1), face_size[1] * c:face_size[1] * (c + 1)] = face_sub
    return sub_faces


def choose_face(image, faces, name):
    """
    Helps with choosing the right face in a image given the name of the person

    Parameters
    ----------
    image (ndarray): The iamge with the faces
    faces (list): List with the faces. In [[v, u, w, h], ...] format
    name (str): The name of the person to choose the face of

    Returns
    -------
    faces (list): List with the face of the person. In [[v, u, w, h]] format
    """
    cv2.imshow("Image", draw_face_boxs(image, faces))
    cv2.imshow("Choose face", create_face_collage(image, faces))

    print(f'Choose face of {name}. Pres one of {list(range(len(faces)))}')

    choice = int(cv2.waitKey(0)) - 48

    print(f"Using face with index: {choice} for {name}")
    cv2.destroyWindow("Image")
    cv2.destroyWindow("Choose face")

    faces = [faces[choice]]
    return faces

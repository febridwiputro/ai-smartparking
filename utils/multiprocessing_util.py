import cv2

def put_queue_none(q):
    if q is None:
        return

    try:
        q.put(None)
    except Exception as e:
        print(f"Error at putting `None` to queue: {e}")


def clear_queue(q, close=True):
    if q is None:
        return

    try:
        while not q.empty():
            try:
                q.get_nowait()
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)

    if close:
        try:
            q.close()
            q.join_thread()
        except Exception as e:
            print(e)

def check_floor(cam_idx):
    cam_map = {
        0: (2, "IN"), 1: (2, "OUT"),
        2: (3, "IN"), 3: (3, "OUT"),
        4: (4, "IN"), 5: (4, "OUT"),
        6: (5, "IN"), 7: (5, "OUT")
    }
    return cam_map.get(cam_idx, (0, ""))

def resize_image(image, max_width, max_height):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scale = max_width / width if width > height else max_height / height
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return image

def show_cam(text, image, max_width=1080, max_height=720):
    res_img = resize_image(image, max_width, max_height)
    cv2.imshow(text, res_img)
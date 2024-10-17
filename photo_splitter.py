import dlib
import cv2

image_path = input('image path: ')
image = cv2.imread(image_path)

if image is None:
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()

faces = detector(gray)

if not faces:
    exit()

for i, face in enumerate(faces):
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())

    if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        continue

    cropped_image = image[y:y + h, x:x + w]

    if cropped_image.size == 0:
        continue

    cropped_image_path = f'/Users/isaacjieu/Documents/projects/genesis/faces/face_{i + 1}.jpg'
    cv2.imwrite(cropped_image_path, cropped_image)

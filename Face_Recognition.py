import os, glob
import cv2
import face_recognition
import numpy as np

# Load known faces and their encodings from subfolders
known_encodings, known_names = [], []
for person in os.listdir("./Images"):
    folder = os.path.join("./Images", person)
    if not os.path.isdir(folder):
        continue
    for img_file in glob.glob(os.path.join(folder, "*.jpg")):
        bgr_img = cv2.imread(img_file)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb_img)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(person)

# Load the test image
test_bgr = cv2.imread("Test_Face.jpg")
if test_bgr is None:
    raise FileNotFoundError("Test_Face.jpg not found.")
test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)

# Detect faces in the test image (using CNN model)
locations = face_recognition.face_locations(test_rgb, model="cnn")
encodings = face_recognition.face_encodings(test_rgb, locations)
print("Faces detected:", len(locations))

# Image dimensions for padding around faces
img_h, img_w, _ = test_bgr.shape
pad = 20  # pixels of padding

# Draw rectangles and labels for each detected face
for (top, right, bottom, left), enc in zip(locations, encodings):
    # Add padding and ensure values stay within image bounds
    top = max(0, top - pad)
    right = min(img_w, right + pad)
    bottom = min(img_h, bottom + pad)
    left = max(0, left - pad)

    # Compare face to known faces
    matches = face_recognition.compare_faces(known_encodings, enc)
    name = "Unknown"
    dists = face_recognition.face_distance(known_encodings, enc)
    if len(dists) > 0:
        best_match = np.argmin(dists)
        if matches[best_match]:
            name = known_names[best_match]

    # Draw rectangle around the face
    cv2.rectangle(test_bgr, (left, top), (right, bottom), (0, 255, 0), 3)
    # Draw filled label box
    cv2.rectangle(test_bgr, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
    # Put the name text on the image
    cv2.putText(test_bgr, name, (left + 5, bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Display the final result
cv2.imshow("Face Recognition Result", test_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

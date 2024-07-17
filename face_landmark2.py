import cv2
import face_recognition

image = cv2.imread('../Image/Me.jpg')
results = face_recognition.face_landmarks(image)
face_location = face_recognition.face_locations(image)
print(results)
chin = results[0]['chin']
right_eyebrow = results[0]['right_eyebrow']
left_eyebrew = results[0]['left_eyebrow']
right_eye = results[0]['right_eye']
left_eye = results[0]['left_eye']
nose_tip = results[0]['nose_tip']
bottom_lip = results[0]['bottom_lip']

while True:
    for (top, right, bottom, left) in face_location:
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 1)

    for chins in chin:
        # Draw a box around the face
        cv2.circle(image, chins, 2, (0, 255, 0), 1)

    for r_eb in right_eyebrow:
        cv2.circle(image, r_eb, 2, (0, 255, 0), 1)

    for l_eb in left_eyebrew:
        cv2.circle(image, l_eb, 2, (0, 255, 0), 1)

    for l_eb in left_eyebrew:
        cv2.circle(image, l_eb, 2, (0, 255, 0), 1)

    for r_e in right_eye:
        cv2.circle(image, r_e, 2, (0, 255, 0), 1)

    for l_e in left_eye:
        cv2.circle(image, l_e, 2, (0, 255, 0), 1)

    for l_e in nose_tip:
        cv2.circle(image, l_e, 2, (0, 255, 0), 1)

    for b_l in bottom_lip:
        cv2.circle(image, b_l, 2, (0, 255, 0), 1)

    cv2.imshow('img', image)
    cv2.waitKey(5)

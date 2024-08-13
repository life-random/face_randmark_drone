"""
작성자 : 최병권
최초 작성일 : 2024 08 13
최종 편집일 : 2024 08 13

요약
    현재 페이지는 페이스 랜드마크을 이용한 얼굴인식 및 추적 AI 드론의 페이스 관련 모듈을 정리한 문서입니다

기능
    - load_known_faces(self, face_file_src):
        얼굴 파일의 데이터화 및 관련 리스트 생성
    - detect_and_match_faces(self, img):
        저장된 얼굴 데이터와 드론이 인식한 얼굴 데이터 비교 및 분류
    - track_face(self, img):
        얼굴 위치 데이터를 해석하여 드론 위치 조정을 위한 데이터 생성
"""

from djitellopy import tello
import cv2
import cvzone
import face_recognition
import numpy as np

from cvzone.PIDModule import PID
from cvzone.PlotModule import LivePlot

class FaceTracking:
    def __init__(self, face_file_src, frame_size=(640, 480)):
        self.wi, self.hi = frame_size  # 프레임 크기 설정 (너비, 높이)

        # PID 제어 초기화
        self.xPID = PID([0.22, 0, 0.1], self.wi // 2)
        self.yPID = PID([0.27, 0, 0.1], self.hi // 2, axis=1)
        self.zPID = PID([0.005, 0, 0.003], 12000, limit=[-20, 15])

        # PID 값의 변화를 실시간으로 보여주는 그래프 초기화
        self.myPlotX = LivePlot(yLimit=[-100, 100], char='X')
        self.myPlotY = LivePlot(yLimit=[-100, 100], char='Y')
        self.myPlotZ = LivePlot(yLimit=[-100, 100], char='Z')

        # 알려진 얼굴을 로드하여 인코딩
        self.known_face_encodings, self.known_face_names = self.load_known_faces(face_file_src)

        # 얼굴 인식에 필요한 초기 변수 설정
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def load_known_faces(self, face_file_src):
        # 알려진 얼굴의 이미지를 로드하고 인코딩하여 반환
        Target_image = face_recognition.load_image_file(face_file_src)
        Target_face_encoding = face_recognition.face_encodings(Target_image)[0]

        return [Target_face_encoding], ["Target"]

    def detect_and_match_faces(self, img):
        # 얼굴을 감지하고 알려진 얼굴과 매칭
        if self.process_this_frame:
            self.face_locations = face_recognition.face_locations(img)
            self.face_encodings = face_recognition.face_encodings(img, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

    def track_face(self, img):
        # 감지된 얼굴을 기반으로 드론 제어값을 계산하고 화면에 출력
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            cx = left + ((right - left) // 2)
            cy = top + ((bottom - top) // 2)
            area = (right - left) * (bottom - top)

            xVal = int(self.xPID.update(cx))
            yVal = int(self.yPID.update(cy))
            zVal = int(self.zPID.update(area))

            print(f'PID Outputs: X={xVal}, Y={yVal}, Z={zVal}')

            # 얼굴 주변에 사각형을 그려주고, PID 값에 따라 드론 제어
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 1)

            imgPlotX = self.myPlotX.update(xVal)
            imgPlotY = self.myPlotY.update(yVal)
            imgPlotZ = self.myPlotZ.update(zVal)

            img = self.xPID.draw(img, [cx, cy])
            img = self.yPID.draw(img, [cx, cy])
            imgStacked = cvzone.stackImages([img, imgPlotX, imgPlotY, imgPlotZ], 2, 0.75)

            return imgStacked

        return cvzone.stackImages([img], 1, 0.75)

if __name__ == "__main__":
    face_tracker = FaceTracking("./faces/choi.jpg")

    me = tello.Tello()
    me.connect()
    print(me.get_battery())
    me.streamoff()
    me.streamon()

    while True:
        img = me.get_frame_read().frame
        img = cv2.resize(img, (640, 480))

        face_tracker.detect_and_match_faces(img)
        imgStacked = face_tracker.track_face(img)

        cv2.imshow("Image Stacked", imgStacked)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

import os
import path
import time
import cv2 as cv

#Input;
def _input(text):
    data = input(text)
    return data

#Check Student Name;
def check_name_valid(name):
    name = name.lower()
    lst = [i for i in range(97, 123)]
    lst.append(32)
    for i in name:
        if ord(i) not in lst : return False
    return True

#Check Student Id;
def check_id_valid(id, path_file):
    id = id.upper()
    file = open(path_file, "r")
    data = file.read()
    data = data.split("\n")
    data = data[:-1]
    lst_id = []

    for i in data:
        i = i.split("/")
        lst_id.append(i[0])

    if len(id) != 8: return False
    if id in lst_id: return False
    else:
        if id[:2] != 'SE' and id[:2] != 'se' and id[:2] != 'Se' and id[:2] != 'sE': return False
        try:
            id = int(id[2: ])
            return True
        except:
            return False

#Creat Data To Attendance;
def data_to_attendance(id):

    os.makedirs(path.path_train + "/" + id)
    os.makedirs(path.path_test + "/" + id)

    face_detection = cv.CascadeClassifier(path.path_haar)
    capture = cv.VideoCapture(0)

    lst_face = []
    for i in range(10):

        isTrue, frame = capture.read()
        faces = face_detection.detectMultiScale(frame)
        if len(faces) != 1:
            print("Only one face in the frame")
        else:
            (x, y, w, h) = faces[0]
            face = frame[y: h+y, x: w+x]
            face = cv.resize(face, (64, 64))
            lst_face.append(face)

        cv.imshow("Webcam", frame)
        time.sleep(0.5)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()

    number_data = len(lst_face)
    number_train = int(number_data / 100 * 80)

    for i in range(number_data):
        if i < number_train:
            cv.imwrite(path.path_train + "/" + id + "/" + str(i) + ".jpg", lst_face[i])
        else:
            cv.imwrite(path.path_test + "/" + id + "/" + str(i) + ".jpg", lst_face[i])

    return (path.path_train + "/" + id, path.path_test + "/" + id)


#Write students to file text;
def write_students_file(path_file, student_id, student_name):

    file = open(path_file, "a")
    file.writelines(student_id + "/" + student_name + "\n")
    file.close()




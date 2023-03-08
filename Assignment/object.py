import lib
import path
import data
import model

#Class;
class Student_Attendance():

    def __init__(self) -> None:

        self.name = None
        self.id = None

        self.path_train = None
        self.path_test = None

    def create_information(self):

        while True:
            self.name = lib._input("Enter your name: ")
            if lib.check_name_valid(self.name) == True: break
        while True:
            self.id = lib._input("Enter your id: ")
            if lib.check_id_valid(self.id, path.path_students_in_class) == True: break
        self.id = self.id.upper()
        lib.write_students_file(path.path_students_in_class, self.id, self.name)

    def creat_data_to_attendance(self):

        self.path_train, self.path_test = lib.data_to_attendance(self.id)

        data.augment_images(self.path_train)
        data.augment_images(self.path_test)

        print("Done")

#Tạo Dữ liệu;
# A = Student_Attendance()
# A.create_information()
# A.creat_data_to_attendance()






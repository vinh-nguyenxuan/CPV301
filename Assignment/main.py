from object import *
from model import *

def menu():
    print("========== CHECK ATTENDANCE ==========")
    print("1. Add Student.")
    print("2. Student Information.")
    print("3. Run model.")
    print("4. Out")
    print("======================================")

def check_input():
    while True:
        inp = input("Enter your choice: ")
        try:
            inp = int(inp)
            if inp in [1, 2, 3, 4]: return inp
        except:
            print("Your Selection Is Not Available!")

def main():
    while True:
        menu()
        choice = check_input()
        if choice == 1:
            student = Student_Attendance()
            student.create_information()
            student.creat_data_to_attendance()
            training()
            print("Done!")
        if choice == 2:
            print(1)
        if choice == 3:
            run()
        if choice == 4:
            print("Thank You!")


if __name__ == "__main__":
    main()
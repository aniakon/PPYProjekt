"""
Program allows the user to answer questions by showing a certain number of fingers to the camera. App class creates the
whole program and uses previously created machine learning models.

Functionalities:
- user can sign in, the program remembers all the answers as well as questions that were answered, so that no question is asked
twice to the same user.
- questions are randomized
- user can turn on the camera whenever ready and take a photo of the answer
- user can draw a different question (repeated randomization)
- questions are drawn until all were answered by the certain user
- user can save the picture taken for the last answer
- user can decide to either see last taken photos on the screen or hide them
- history of user's answers is visible on the screen
"""

import random
import atexit
from tkinter import messagebox

import customtkinter as ctk
import cv2 as cv
import json
import tensorflow as tf
import numpy as np
from PIL import Image

class App(ctk.CTk):
    def __init__(self):
        """
        Constructor. Creates the main screen, sets size, initializes finger recognision model, loads previously prepared
        questions and turns on the first window to allow for users' login.
        """
        super().__init__()
        self.geometry("1070x500")
        self.minsize(1070, 600)
        self.title("Odpowiadanie na pytania")
        self.mainOpened = False
        self.model = tf.keras.models.load_model("hand_model_finetuned.keras")
        self.class_names = ['0', '1', '2', '3', '4', '5']
        self.last_photo = None
        with open("questions.txt", 'r', encoding='utf-8') as f:
            self.questions = [linia.strip() for linia in f.readlines()]

        atexit.register(self.saveAnswers)
        self.firstWindow()


    def firstWindow(self):
        """
        Function allows for logging in.
        If a user just signed out, their data is saved.
        If a given login is already in database, then it loads the data (answers, indexes to be used) to variables.
        If a given login is not in the database, it creates an empty data dictionary and fills indexes to be used with every index.
        After accepting the login, it turns on the main window of the program.
        """
        if self.mainOpened:
            self.saveAnswers()
            self.last_photo = None
            self.mainFrame.grid_forget()

        dialog = ctk.CTkInputDialog(text="Podaj imię i nazwisko:", title="Nowy użytkownik")
        text = dialog.get_input()
        if text:
            self.name = text
            self.getAnswers()
            if self.name not in self.jsonAnswers:
                self.indexes = [i for i in range(len(self.questions))] # wszystkie indeksy do wykorzystania
                self.answers = {}
            else:
                self.indexes = self.jsonAnswers[self.name]["indeksy"] # albo te które pozostały
                self.answers = self.jsonAnswers[self.name]["odpowiedzi"]
            self.mainWindow()

    def mainWindow(self):
        """
        Creates the whole structure of the program with linking functions to certain buttons and loading history of answers from database.
        Left side of the screen is meant for the interaction with the user (opening camera, drawing a question, saving a photo, showing photos).
        Right side of the screen is meant for displaying the history of user's answers with a button allowing for signing out.
        """
        self.mainFrame = ctk.CTkFrame(self, width=1100)
        self.mainFrame.grid_columnconfigure(0, minsize=500)
        self.mainFrame.grid_columnconfigure(1, minsize=500)
        self.mainFrame.grid_rowconfigure(0, minsize=600)
        # -------- lewa strona ----------
        frameLeft = ctk.CTkFrame(self.mainFrame, width=500, height=500)
        frameLeft.grid(row=0, column=0, sticky="nsew")
        frameLeft.grid_propagate(False)

        innerFrame = ctk.CTkFrame(frameLeft, width=500, height=500)
        innerFrame.pack(expand=True)
        innerFrame.pack_propagate(False)
        innerFrame.grid_columnconfigure(0, minsize=500)

        label = ctk.CTkLabel(innerFrame, text=f"Cześć {self.name}! ", font=ctk.CTkFont(size=20, weight="bold"))
        label.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        insLable = ctk.CTkLabel(innerFrame,
                                text="Aby odpowiedzieć na pytanie otwórz kamerę, \nna palcach jednej ręki pokaż liczbę i wciśnij q.",
                                font=ctk.CTkFont(size=10, slant="italic"))
        insLable.grid(row=1, column=0, padx=3, pady=3, sticky="ew")

        questionDesc = ctk.CTkLabel(innerFrame, text="Wylosowano pytanie: ")
        questionDesc.grid(row=2, column=0, padx=5, pady=5, sticky="ew")

        self.buttonCam = ctk.CTkButton(innerFrame, text="Otwórz kamerę, aby odpowiedzieć",
                                       command=lambda: self.openCamera())
        self.question = ctk.CTkLabel(innerFrame, text=self.drawQuestion(), font=ctk.CTkFont(size=15), wraplength=450,
                                     justify="center")
        self.question.grid(row=3, column=0, padx=10, pady=10, sticky="n")
        self.buttonCam.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        self.buttonCam = ctk.CTkButton(innerFrame, text="Wylosuj inne pytanie",
                                  command=lambda: self.question.configure(text=self.drawQuestion()))
        self.buttonCam.grid(row=5, column=0, padx=10, pady=10, sticky="ew")

        buttonPhoto = ctk.CTkButton(innerFrame, text="Zapisz ostatnie zdjęcie", command=lambda: self.saveLastPhoto())
        buttonPhoto.grid(row=6, column=0, padx=10, pady=10, sticky="ew")

        self.photoLabel = ctk.CTkLabel(innerFrame)
        self.showLast = ctk.BooleanVar(value=False)
        showLastPhotoCheck = ctk.CTkCheckBox(innerFrame, text = "Pokaż ostatnie zdjęcie", variable=self.showLast, command=lambda: self.updatePhoto())
        showLastPhotoCheck.grid(row=7, column=0, padx=10, pady=10, sticky="ew")


        self.photoLabel.grid(row=8, column=0, padx=10, pady=10, sticky="ew")
        self.photoLabel.configure(image=None, text="")

        # -------- prawa strona ---------
        frameRight = ctk.CTkFrame(self.mainFrame, width=550)
        frameRight.grid_columnconfigure(0, minsize=500)
        frameRight.grid(row=0, column=1, sticky="nsew")

        logOutButton = ctk.CTkButton(frameRight, text="Wyloguj się", command=lambda: self.firstWindow())
        logOutButton.grid(row=0, column=0, sticky="ne")

        hisLabel = ctk.CTkLabel(frameRight, text="Historia pytań", font=ctk.CTkFont(size=15, weight="bold"))
        hisLabel.grid(row=1, column=0, padx=10, pady=10, sticky="n")

        self.hisFrame = ctk.CTkScrollableFrame(frameRight, height=400, width=550)
        self.hisFrame.grid(row=2, column=0, sticky="nsew")
        self.hisRows = []
        for ans in self.answers:
            txt = ans + " -> " + str(self.answers[ans])
            self.addHisRow(txt)

        self.mainFrame.grid(row=0, column=0, sticky="nsew")
        self.mainOpened = True


    def openCamera(self):
        """
        Function that open a camera. When user types q, then last frame is saved (changing colors, dimenentions),
        predictions are made. Predictions are added to history log and answers. Photo to be shown in main window gui is changed.
        """
        self.video_cap = cv.VideoCapture(0)
        if not self.video_cap.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # wyświetlanie w szarości
            cv.imshow('frame', gray)
            if cv.waitKey(1) == ord('q'):
                self.last_photo = frame.copy() # zapisane zdjęcie
                self.last_photo_res = cv.resize(self.last_photo, (128, 128))
                self.last_photo_res = cv.cvtColor(self.last_photo_res, cv.COLOR_BGR2RGB) # z bgr (opencv) -> rgb (keras)
                self.last_photo_res = self.last_photo_res.astype('uint8') # piksele [0,255]
                self.last_photo_res = np.expand_dims(self.last_photo_res, axis=0) # wymiar batch (1,128,128,3)

                pred = self.model.predict(self.last_photo_res)
                predicted_class = self.class_names[np.argmax(pred)]

                self.addHisRow(self.questions[self.ind] + " -> " + predicted_class)
                self.indexes.remove(self.ind) # aby już nie powtarzać
                q = self.questions[self.ind]
                self.answers[q] = int(predicted_class)
                self.prevInd = self.ind
                self.question.configure(text=self.drawQuestion())
                self.updatePhoto()
                break

        self.video_cap.release()
        cv.destroyAllWindows()

    def addHisRow(self, answer):
        """
        Function to add an answer to the history frame. Used both with loading all answers from history,
        but also new answers predicted by gesture recognition model.
        Args:
            answer (str): The answer to be displayed.
        """
        row = ctk.CTkLabel(self.hisFrame, text = answer, anchor="w")
        row.pack(fill='x', padx=2) #rozciągnięte na cały x
        self.hisRows.append(row)

    def getAnswers(self):
        """
        Function to load all user answers from the JSON database file into memory.

        Reads the 'answers.json' file and stores its content in a dictionary, which maps usernames to their
        saved answers and remaining question indexes.
        """
        with open("answers.json", "r", encoding="utf-8") as answers:
            self.jsonAnswers = json.load(answers)

    def drawQuestion(self):
        """
        Function to draw a random question from the question base that the user hasn't answered yet.
        If there is no questions left for a user, disables the ability to open the camera.
        Returns:
            str: question text or "brak pytań"
        """
        if len(self.indexes) == 0:
            self.buttonCam.configure(state=ctk.DISABLED)
            return "Brak pytań"
        self.ind = random.choice(self.indexes)
        return self.questions[self.ind]

    def saveAnswers(self):
        """
        Function to save all user answers from memory to a JSON file.
        """
        self.jsonAnswers[self.name] = {
            "odpowiedzi": self.answers,
            "indeksy": self.indexes
        }
        with open("answers.json", "w", encoding="utf-8") as file:
            json.dump(self.jsonAnswers, file, ensure_ascii=False, indent=4)

    def saveLastPhoto(self):
        """
        Function to save the last photo (in the grayscale) from the history in the photoAnswers directory with the name:
        {nameOfUser}_{numerOfQuestion}.jpg.
        Returns an info box with either information about successful saving or a warning that there was no photo to be saved.
        """
        if (self.last_photo is not None):
            name = f"photoAnswers/{self.name}_{self.prevInd}.jpg"
            gray = cv.cvtColor(self.last_photo, cv.COLOR_BGR2GRAY)
            cv.imwrite(name, gray)
            messagebox.showinfo("Informacja", f"Zdjęcie zapisano jako {name}")
        else:
            messagebox.showwarning("Błąd", "Brak zdjęcia do zapisania.")

    def updatePhoto(self):
        """
        Function to display (or not) the last image in greyscale, based on the option chosen by the user.
        """
        if self.showLast.get():
            if (self.last_photo is not None):
                img = Image.fromarray(self.last_photo).convert("L").resize((200, 150)) # na PIL image i greyscale
                self.photo = ctk.CTkImage(img, size=(200, 150))
                self.photoLabel.configure(image=self.photo, text="")
                self.photoLabel.image = self.photo # zapisanie w atrybucie image z photolabel, aby zachowane było w pamięci
                self.photoLabel.grid()
        else:
            self.photoLabel.grid_remove()

app = App()
app.mainloop()
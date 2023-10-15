import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
import face_recognition
from kivy.core.window import Window
import cv2
import shutil
import numpy as np
from tkinter import filedialog
import os
import csv
from datetime import datetime
from kivy.clock import Clock
from kivy.graphics.texture import Texture

kivy.require('1.11.1')

# Define las rutas de los archivos y directorios
script_dir = os.path.dirname(__file__)
attendance_csv = os.path.join(script_dir, 'bd', 'attendance_log.csv')
users_folder = 'Fotos'
users_csv = os.path.join(script_dir, 'bd', 'users.csv')
predictor_path = os.path.join(script_dir, 'SysAsistencia', 'shape_predictor_68_face_landmarks.dat')

# Otros elementos globales
known_face_encodings = []
known_face_names = []
students = known_face_names.copy()
attendance_times = {}
camera = None

# Inicializar la captura de video
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 840)  # Ancho
video_capture.set(4, 680)  # Alto

Window.size = (300, 400)

class NotificationPopup(Popup):
    def __init__(self, message, **kwargs):
        super(NotificationPopup, self).__init__(**kwargs)
        self.title = 'Notificación'
        self.size_hint = (None, None)
        self.size = (300, 200)
        self.background_color = (1, 1, 1, 1)  # Establece el fondo a blanco (no transparente)
        
        content_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        content = Label(text=message, halign='center', valign='middle', text_size=(300, None), markup=True)
        close_button = Button(text="Cerrar", size_hint_y=None, height=40)
        close_button.bind(on_press=self.dismiss)
        
        content_layout.add_widget(content)
        content_layout.add_widget(close_button)
        self.content = content_layout
        

class MyApp(App):
    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        self.attendance_times = {}  # Inicializar el diccionario de tiempos de asistencia
        self.attendance_csv = attendance_csv
        # Asegurarse de que la carpeta "bd" exista al inicio
        bd_folder = os.path.join(script_dir, 'bd')
        if not os.path.exists(bd_folder):
            os.makedirs(bd_folder)
            
        self.load_users()
        self.notification_popup = None  # Inicializar la notificación

    def capture_user_image(self):
        # Capturar un cuadro de la cámara
        if self.camera.texture:
            w, h = self.camera.texture.size
            buf = self.camera.texture.pixels
            frame = np.frombuffer(buf, 'uint8').reshape((h, w, 4))
            return frame
        return None

    def build(self):
        root_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Crea una cámara para mostrar la vista de la cámara
        self.camera = Camera(play=True)
        self.camera.resolution = (840, 680)  # Establece la resolución de la cámara
        self.camera.size = (840, 680)
        root_layout.add_widget(self.camera)

        # Agregar un botón para registrar un nuevo usuario
        register_button = Button(text="Registrar Usuario", size_hint_y=None, height=50)
        # Agregar un botón para mostrar la asistencia registrada
        show_attendance_button = Button(text="Mostrar Asistencia", size_hint_y=None, height=50)
        # Agregar un botón para limpiar los datos
        clear_data_button = Button(text="Limpiar Datos", size_hint_y=None, height=50)

        show_attendance_button.bind(on_press=lambda instance: self.show_attendance())
        register_button.bind(on_press=lambda instance: self.get_user_name())
        # Vincula el botón de limpieza con la función de limpiar datos
        clear_data_button.bind(on_press=lambda instance: self.clear_data())

        # Etiqueta para mostrar mensajes
        self.message_label = Label(
            text="",
            color=(0, 1, 0, 1),
            size_hint_y=None,
            height=50,
            size_hint_x=None,
            width=200,  # Establece el ancho máximo en 200 píxeles
            text_size=(200, None)  # Texto se ajusta al ancho máximo y salta a la siguiente línea si es necesario
        )

        root_layout.add_widget(register_button)
        root_layout.add_widget(show_attendance_button)
        root_layout.add_widget(clear_data_button)
        root_layout.add_widget(self.message_label)

        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # Actualizar la vista de la cámara

        return root_layout

    def load_users(self):
        csv_file_name = os.path.join(script_dir, 'bd', 'users.csv')
        if os.path.exists(csv_file_name):
            with open(csv_file_name, 'r') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    user_name, image_path = row
                    known_face_names.append(user_name)
                    image = face_recognition.load_image_file(image_path)
                    user_encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(user_encoding)

    def get_user_name(self):
        layout = BoxLayout(orientation='vertical')
        label = Label(text='Enter Para Escribir Nombre:')
        username_input = TextInput()
        save_button = Button(text='Guardar')

        layout.add_widget(label)
        layout.add_widget(username_input)
        layout.add_widget(save_button)

        popup = Popup(title='Nombre de Estudiante', content=layout, size_hint=(None, None), size=(300, 200))

        def save_user_name(instance):
            user_name = username_input.text
            if user_name:
                self.open_image_selector(user_name)
                popup.dismiss()

        save_button.bind(on_press=save_user_name)
        popup.open()

    def open_image_selector(self, user_name):
        layout = BoxLayout(orientation='vertical')
        label = Label(text=f'Selecciona Una Imagen Para {user_name}:')
        select_button = Button(text='Seleccionar Imagen')

        layout.add_widget(label)
        layout.add_widget(select_button)

        popup = Popup(title='Seleccionar Imagen', content=layout, size_hint=(None, None), size=(300, 200))

        def select_image():
            user_image_path = filedialog.askopenfilename(title=f'Selecciona Una Imagen Para {user_name}')
            if user_image_path:
                user_image_filename = os.path.join(users_folder, f"{user_name}.jpg")

                # Asegurarse de que el directorio exista, si no, créalo
                os.makedirs(os.path.dirname(user_image_filename), exist_ok=True)

                shutil.copy(user_image_path, user_image_filename)

                with open(os.path.join(script_dir, 'bd', 'users.csv'), 'a', newline='') as csv_file:
                    lnwriter = csv.writer(csv_file)
                    lnwriter.writerow([user_name, user_image_filename])

                known_face_names.append(user_name)
                user_image = face_recognition.load_image_file(user_image_filename)
                user_encoding = face_recognition.face_encodings(user_image)[0]
                known_face_encodings.append(user_encoding)

                students.append(user_name)
                print(f"Imagen seleccionada para {user_name} Registrado Exitosamente")
                popup.dismiss()

        select_button.bind(on_press=lambda x: select_image())
        popup.open()

    def register_attendance(self, user_name):
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")

        if user_name not in self.attendance_times or (datetime.now() - self.attendance_times[user_name]).seconds > 10:
            message = f"Asistencia de {user_name} registrada correctamente a las {current_time}."
            self.show_notification(message)

            attendance_data = [user_name, "Asistió", current_date, current_time]

            with open(self.attendance_csv, 'a', newline='') as attendance_file:
                lnwriter = csv.writer(attendance_file)
                lnwriter.writerow(attendance_data)

            self.attendance_times[user_name] = datetime.now()  # Actualizar el tiempo de asistencia

    def show_notification(self, message):
        if self.notification_popup:
            self.notification_popup.dismiss()  # Cerrar notificación anterior si existe
        self.notification_popup = NotificationPopup(message)
        self.notification_popup.open()

    def update_frame(self, dt):
        if self.camera.texture:
            frame = self.capture_user_image()
            if frame is not None:
                # Asegurarse de que el marco esté en el formato correcto (BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Realizar la detección facial
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, faces)

                for (top, right, bottom, left), face_encoding in zip(faces, face_encodings):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.7)
                    name = "Unknown"
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    if name in students:
                        students.remove(name)
                        self.register_attendance(name)
                # Rotar el marco 180 grados para corregir la orientación
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                frame = cv2.flip(frame, 1)
                # Convierte el marco a un formato compatible con Kivy y muestra la vista en la cámara
                h, w, _ = frame.shape
                buf = frame.tostring()
                texture = Texture.create(size=(w, h), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.camera.texture = texture

    def show_attendance(self):
        attendance_data = []

        try:
            # Intenta abrir y leer el archivo CSV con los datos de asistencia
            with open(self.attendance_csv, 'r') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    attendance_data.append(row)
        except FileNotFoundError:
            message = "La tabla de asistencia no existe."
            self.show_notification(message)
            return

        layout = BoxLayout(orientation='vertical')
        popup = Popup(title='Asistencia Registrada', content=layout, size_hint=(None, None), size=(300, 300))

        # Crea una cuadrícula para mostrar los datos
        grid = GridLayout(cols=4, spacing=5, size_hint=(None, None), width=290)
        grid.add_widget(Label(text="Nombre"))
        grid.add_widget(Label(text="Estado"))
        grid.add_widget(Label(text="Fecha"))
        grid.add_widget(Label(text="Hora"))

        for data in attendance_data:
            for item in data:
                grid.add_widget(Label(text=item, size_hint_x=None, width=75))

        # Agrega la cuadrícula a un desplazamiento para ver todos los datos
        scroll_view = ScrollView()
        scroll_view.add_widget(grid)
        layout.add_widget(scroll_view)

        close_button = Button(text="Cerrar")
        close_button.bind(on_press=popup.dismiss)
        layout.add_widget(close_button)

        popup.open()




    def clear_data(self):
        # Lógica para limpiar los datos
        known_face_encodings.clear()
        known_face_names.clear()
        students.clear()
        self.attendance_times.clear()

        # Elimina los archivos CSV
        if os.path.exists(self.attendance_csv):
            os.remove(self.attendance_csv)
        if os.path.exists(users_csv):
            os.remove(users_csv)

        # Elimina la carpeta de fotos de usuarios y su contenido si existe
        users_photos_folder = os.path.join(os.path.dirname(__file__), users_folder)
        if os.path.exists(users_photos_folder):
            shutil.rmtree(users_photos_folder)
            message = "Datos y archivos CSV limpiados con éxito"
        else:
            message = "La carpeta de fotos de usuarios no existe."

        # Elimina la carpeta "bd" si está vacía
        bd_folder = os.path.join(script_dir, 'bd')
        if os.path.exists(bd_folder) and not os.listdir(bd_folder):
            os.rmdir(bd_folder)

        self.show_notification(message)


if __name__ == '__main__':
    MyApp().run()


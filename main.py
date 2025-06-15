import sys
import os
import traceback
import cv2
import numpy as np
import shutil
from collections import defaultdict
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
      QWidget, QComboBox, QDialog, QMessageBox, QInputDialog, QScrollArea, QGridLayout, QCheckBox, QSplashScreen
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from operations import (add_operation, subtract_operation, equalize_histogram, detect_edges_canny,
                         detect_edges_sobel, predefined, prepare_data, perform_pca_analysis, 
                         neutralize_images, copy_data, enhance_text_with_morphology, bayes_classification, run_bayes, text_detection, clahe)
from datetime import date, datetime
from functools import wraps



class RotateImageDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rotate Image")
        self.setFixedSize(600, 600)  # Stały rozmiar okna
        self.image_path = image_path
        self.current_angle = 0

        # Wczytanie obrazu
        self.original_image = cv2.imread(image_path)
        self.display_image = self.original_image.copy()

        # Inicjalizacja elementów interfejsu
        self.layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setFixedSize(550, 450)  # Stały rozmiar QLabel
        self.image_label.setAlignment(Qt.AlignCenter)  # Wyśrodkowanie obrazu
        self.update_image_display()

        # Przyciski obracania
        self.button_layout = QHBoxLayout()
        self.rotate_left_button = QPushButton("Rotate Left (-90°)")
        self.rotate_right_button = QPushButton("Rotate Right (+90°)")
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")

        self.rotate_left_button.clicked.connect(self.rotate_left)
        self.rotate_right_button.clicked.connect(self.rotate_right)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        self.button_layout.addWidget(self.rotate_left_button)
        self.button_layout.addWidget(self.rotate_right_button)
        self.button_layout.addWidget(self.ok_button)
        self.button_layout.addWidget(self.cancel_button)

        self.layout.addWidget(self.image_label)
        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

    def rotate_left(self):
        self.current_angle -= 90
        self.update_image()

    def rotate_right(self):
        self.current_angle += 90
        self.update_image()

    def update_image(self):
        # Obrót obrazu o bieżący kąt
        h, w = self.original_image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.current_angle, 1.0)
        self.display_image = cv2.warpAffine(self.original_image, rotation_matrix, (w, h))
        self.update_image_display()

    def get_rotation_angle(self):
        """Zwraca bieżący kąt obrotu"""
        return self.current_angle

    def update_image_display(self):
        # Aktualizacja obrazu w QLabel z dopasowaniem do stałego rozmiaru
        rgb_image = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Skalowanie obrazu do QLabel
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def get_rotated_image(self):
        return self.display_image


#obsluga loading screena
class LoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ładowanie")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)  # Usunięto Qt.WindowStaysOnTopHint
        self.setModal(True)

        # Ustawienia stylu - cienka czarna ramka
        self.setStyleSheet("background-color: white; border: 2px solid black;")

        # Komunikat ładowania
        layout = QVBoxLayout()
        label = QLabel("Proszę czekać, trwa ładowanie...")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)
        self.resize(300, 100)


def show_loading(func):
    """
    Dekorator, który wyświetla dialog ładowania przed wywołaniem funkcji i zamyka go po zakończeniu.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Tworzymy dialog ładowania
        loading_dialog = LoadingDialog(self)
        loading_dialog.show()  # Wyświetl dialog ładowania

        # Zablokuj zdarzenia GUI do momentu zamknięcia dialogu
        QApplication.processEvents()

        try:
            result = func(self, *args, **kwargs)  # Wywołanie oryginalnej funkcji
        except Exception as e:
            print(f"Błąd podczas ładowania: {e}")
            raise  # Rzuć wyjątek dalej po zalogowaniu
        finally:
            loading_dialog.close()  # Zamknięcie dialogu po zakończeniu
            QApplication.processEvents()  # Upewnij się, że zdarzenia są obsłużone
        return result

    return wrapper

class ImageWindow(QDialog):
    """
    Okno dialogowe do wyświetlania wyników operacji na obrazach.
    Umożliwia obracanie wyświetlanego obrazu i zmianę trybu wyświetlania.
    """
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Operation Result")  # Ustawienie tytułu okna
        self.setGeometry(100, 100, 800, 600)  # Domyślny rozmiar okna

        # Główny layout dla okna
        self.layout = QVBoxLayout(self)

        # QLabel do wyświetlania obrazu
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)  # Wyśrodkowanie obrazu w QLabel
        self.layout.addWidget(self.image_label)

        # Przechowywanie oryginalnego obrazu w formacie NumPy
        self.original_image = image
        self.scale_factor = 1.0  # Domyślny współczynnik skalowania
        self.rotation_angle = 0  # Domyślny kąt obrotu
        self.display_mode = "RGB" if len(image.shape) == 3 else "Grayscale"  # Domyślny tryb

        # Konwersja obrazu NumPy na QPixmap i wyświetlenie
        self.update_image()

        # Dodanie przycisków do obracania obrazu i zmiany trybu wyświetlania
        self.add_controls()

        self.setLayout(self.layout)

    def add_controls(self):
        """Dodanie przycisków kontrolnych do obracania obrazu i zmiany trybu wyświetlania."""
        controls_layout = QHBoxLayout()

        # Przycisk obracania w lewo
        rotate_left_button = QPushButton("Rotate Left")
        rotate_left_button.clicked.connect(self.rotate_left)
        controls_layout.addWidget(rotate_left_button)

        # Przycisk obracania w prawo
        rotate_right_button = QPushButton("Rotate Right")
        rotate_right_button.clicked.connect(self.rotate_right)
        controls_layout.addWidget(rotate_right_button)

        # Przycisk zmiany trybu wyświetlania
        toggle_mode_button = QPushButton("Toggle Mode")
        toggle_mode_button.clicked.connect(self.toggle_mode)
        controls_layout.addWidget(toggle_mode_button)

        self.layout.addLayout(controls_layout)

    def update_image(self):
        """Aktualizacja obrazu w QLabel w zależności od aktualnych transformacji."""
        # Obracanie obrazu
        rotated_image = self.rotate_image(self.original_image, self.rotation_angle)

        # Konwersja obrazu NumPy na QPixmap
        if self.display_mode == "Grayscale":
            if len(rotated_image.shape) == 3:  # Konwersja do skali szarości
                rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
            height, width = rotated_image.shape
            qimage = QImage(rotated_image.data, width, height, QImage.Format_Grayscale8)
        else:  # Tryb RGB
            if len(rotated_image.shape) == 2:  # Jeśli obraz jest w skali szarości
                rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2BGR)
            height, width, channels = rotated_image.shape
            qimage = QImage(rotated_image.data, width, height, channels * width, QImage.Format_BGR888)

        pixmap = QPixmap.fromImage(qimage)

        # Skalowanie obrazu do rozmiaru QLabel
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def rotate_image(self, image, angle):
        """Obracanie obrazu o zadany kąt."""
        center = (image.shape[1] // 2, image.shape[0] // 2)  # Środek obrazu
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # Macierz transformacji
        rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))  # Obrót obrazu
        return rotated

    def rotate_left(self):
        """Obracanie obrazu o 90 stopni w lewo."""
        self.rotation_angle += 90
        self.update_image()

    def rotate_right(self):
        """Obracanie obrazu o 90 stopni w prawo."""
        self.rotation_angle -= 90
        self.update_image()

    def toggle_mode(self):
        """Przełączanie trybu wyświetlania między RGB a Grayscale."""
        if self.display_mode == "RGB":
            self.display_mode = "Grayscale"
        else:
            self.display_mode = "RGB"
        self.update_image()

    def resizeEvent(self, event):
        """Automatyczna aktualizacja obrazu przy zmianie rozmiaru QLabel."""
        self.update_image()
        super().resizeEvent(event)



class MainWindow(QMainWindow):
    """
    Główne okno aplikacji, które obsługuje wyświetlanie obrazów,
    wybór kategorii, konfiguracji oraz wykonywanie operacji na obrazach.
    """
    def __init__(self):
        super().__init__()

        # Słownik funkcji operacji na obrazach
        self.operation_functions = {
            "Addition":  add_operation,  # Dodawanie obrazów
            "Subtraction": subtract_operation,  # Odejmowanie obrazów
            "Histogram Equalization": equalize_histogram,  # Wyrównanie histogramu
            "Edge Detection Canny": detect_edges_canny,  # Wykrywanie krawędzi
            "Edge Detection Sobel": detect_edges_sobel,
            "Predefined": predefined,  # Wykonywanie wstępnie zdefiniowanych operacji
            "Prepare Data": prepare_data, # funkcja przygotowująca dane do działania innych algorytmów
            "PCA": perform_pca_analysis,
            "Erosion and Dilation": enhance_text_with_morphology,
            "Neutralization": neutralize_images,
            "Bayes Classification": run_bayes,  # Nowa operacja
            "Text Detection": text_detection,
            "CLAHE": clahe
        }

        self.setWindowTitle("Main Window")  # Tytuł głównego okna
        self.setGeometry(100, 100, 850, 800)  # Rozmiar okna

        # Ustawienia głównego layoutu
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.central_widget.setLayout(self.layout)

        # Układ dla wyboru folderu i kategorii
        self.path_category_layout = QHBoxLayout()
        self.layout.addLayout(self.path_category_layout)

        # Przycisk do ładowania folderu
        self.load_button = QPushButton("Load Folder")
        self.load_button.clicked.connect(self.load_folder)  # Połączenie przycisku z funkcją ładowania folderu
        self.load_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.path_category_layout.addWidget(self.load_button)

        # Menu wyboru kategorii
        self.category_menu = QComboBox()
        self.category_menu.addItem("Select Category")
        self.category_menu.currentIndexChanged.connect(self.change_category)  # Połączenie zmiany kategorii z funkcją
        self.category_menu.setStyleSheet("font-size: 14px; padding: 5px;")
        self.path_category_layout.addWidget(self.category_menu)

        # Układ dla konfiguracji i operacji
        self.config_operations_layout = QHBoxLayout()
        self.layout.addLayout(self.config_operations_layout)

        

        # Menu wyboru operacji
        self.operations_menu = QComboBox()
        self.operations_menu.addItems(["Addition", "Subtraction", "Histogram Equalization", 
                                       "Edge Detection Canny", "Edge Detection Sobel", "Erosion and Dilation",  "PCA",
                                         "Neutralization", "Bayes Classification", "Text Detection", "CLAHE"])
        self.operations_menu.setStyleSheet("font-size: 14px; padding: 5px;")
        self.config_operations_layout.addWidget(self.operations_menu)

        # Menu wyboru konfiguracji
        self.configurations_menu = QComboBox()
        self.configurations_menu.addItem("Select Direction")
        #self.configurations_menu.addItem("ALL")
        self.configurations_menu.currentIndexChanged.connect(self.change_category)  # Połączenie zmiany kategorii z funkcją
        self.configurations_menu.setStyleSheet("font-size: 14px; padding: 5px;")
        self.config_operations_layout.addWidget(self.configurations_menu)

        # Układ dla przycisków uruchamiania
        self.run_buttons_layout = QHBoxLayout()
        self.layout.addLayout(self.run_buttons_layout)

        # Przycisk uruchamiania operacji
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_operation)  # Połączenie z funkcją uruchamiania operacji
        self.run_button.setStyleSheet(
            """
            font-size: 16px; 
            padding: 10px; 
            background-color: #5cb85c; 
            color: white; 
            border-radius: 5px;
            """
        )
        self.run_buttons_layout.addWidget(self.run_button)

        # Przycisk uruchamiania operacji wstępnie zdefiniowanej
        self.run_predefined_button = QPushButton("Run Predefined")
        self.run_predefined_button.clicked.connect(self.run_predefined_operation)  # Połączenie z funkcją uruchamiania wstępnej operacji
        self.run_predefined_button.setStyleSheet(
            """
            font-size: 16px; 
            padding: 10px; 
            background-color: #f0ad4e; 
            color: white; 
            border-radius: 5px;
            """
        )
        self.run_buttons_layout.addWidget(self.run_predefined_button)



        # Układ dla przycisków obsługi checkbox
        self.checkbox_layout = QHBoxLayout()
        self.layout.addLayout(self.checkbox_layout)

        self.checkbox_clear_button = QPushButton("Clear All")
        self.checkbox_clear_button.clicked.connect(self.clear_checkboxes)  # Połączenie z funkcją uruchamiania operacji
        self.checkbox_clear_button.setStyleSheet(
            """
            font-size: 16px; 
            padding: 10px; 
            background-color: blue; 
            color: white; 
            border-radius: 5px;
            """
        )
        self.checkbox_layout.addWidget(self.checkbox_clear_button)

        self.checkbox_all_button = QPushButton("All Images")
        self.checkbox_all_button.clicked.connect(self.select_all_checkboxes)  # Połączenie z funkcją uruchamiania operacji
        self.checkbox_all_button.setStyleSheet(
            """
            font-size: 16px; 
            padding: 10px; 
            background-color: blue; 
            color: white; 
            border-radius: 5px;
            """
        )
        self.checkbox_layout.addWidget(self.checkbox_all_button)

        # Inicjalizacja danych
        self.image_paths = []  # Ścieżki do obrazów
        self.image_categories = defaultdict(list)  # Kategorie obrazów
        self.image_directions = defaultdict(list)  # Kierunki w nazwach obrazów
        self.configuration_photos = []  # Zdjęcia spełniające wybraną konfigurację
        self.current_index = 0  # Aktualny indeks wyświetlanego obrazu
        self.current_category = None  # Aktualnie wybrana kategoria
        self.temporary_results_folder=r"temporary_results"

        # Tworzenie scrollowalnego obszaru
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_widget)
        self.grid_layout = QGridLayout(self.scroll_widget)

        self.selected_images = []  # Lista zaznaczonych obrazów

        # Dodanie scroll area do głównego layoutu
        self.layout.addWidget(self.scroll_area)

    @show_loading 
    def select_all_checkboxes(self, *args):
        """
        Zaznacza wszystkie obrazy dla aktualnej kategorii i konfiguracji.
        """
        try:
            self.selected_images.clear()
            selected_category = self.category_menu.currentText()
            selected_configuration=self.configurations_menu.currentText()
            # Upewniamy się, że operujemy na aktualnie wyświetlanych obrazach (filtrowanych)
            if selected_category in self.image_categories and selected_configuration in self.image_directions:
                # Lista obrazów przefiltrowanych według kategorii i konfiguracji
                category_images = set(self.image_categories[selected_category])
                configuration_images = set(self.image_directions[selected_configuration])
                filtered_images = list(category_images & configuration_images)  # Przecięcie zbiorów
                
                # Reset zaznaczeń i ustawienie nowej listy
                
                self.selected_images.extend(filtered_images)
                
                # Opcjonalne odświeżenie menu, jeśli potrzebne
                if selected_category in self.image_categories:
                    category_index = self.category_menu.findText(selected_category)
                    if category_index != -1:  # Upewniamy się, że element istnieje
                        self.category_menu.setCurrentIndex(category_index)

                if selected_configuration in self.image_directions:
                    configuration_index = self.configurations_menu.findText(selected_configuration)
                    if configuration_index != -1:  # Upewniamy się, że element istnieje
                        self.configurations_menu.setCurrentIndex(configuration_index)
                
                self.current_category=selected_category
                self.current_configuration=selected_configuration
                # Odśwież widok
                self.refresh_image_grid()
            else:
                print("Brak obrazów do zaznaczenia w bieżącej kategorii i konfiguracji.")
        except Exception as e:
            print(f"Nieoczekiwany błąd: {e}")


    @show_loading 
    def clear_checkboxes(self, *args):
        self.selected_images.clear()
        if self.current_category in self.image_categories:
            category_index = self.category_menu.findText(self.current_category)
            if category_index != -1:  # Upewniamy się, że element istnieje
                self.category_menu.setCurrentIndex(category_index)

        if self.current_configuration in self.image_directions:
            configuration_index = self.configurations_menu.findText(self.current_configuration)
            if configuration_index != -1:  # Upewniamy się, że element istnieje
                self.configurations_menu.setCurrentIndex(configuration_index)
        self.refresh_image_grid()

    @show_loading    
    def load_folder(self, *args):
        """
        Funkcja do ładowania folderu z obrazami.
        Pobiera ścieżki do wszystkich obrazów w wybranym folderze i kategoryzuje je.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz folder ze zdjęciami")  # Wybranie folderu z obrazami
        print(f"Folder path: {folder_path}")
        print(f"Temporary_folder_path: {self.temporary_results_folder}")

        if folder_path:
            # Pobieranie ścieżek do obrazów w folderze
            
             # prepare_data(folder_path, self.temporary_results_folder)
            self.load_folder_known(folder_path, False)
            choice = QMessageBox.question(
                self,
                "Wyrównanie homografii",
                "Czy wyrównać zdjęcia z wybranego folderu?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if choice == QMessageBox.Yes:
                prepare_data(folder_path, self.temporary_results_folder)
            else:
                self.load_folder_known(folder_path, False)

            self.image_paths = [
                os.path.join(self.temporary_results_folder, f)
                for f in os.listdir(self.temporary_results_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            ]
            

            if self.image_paths:
                self.categorize_images()  # Kategoryzowanie obrazów na podstawie nazw
                self.update_category_menu()  # Aktualizacja menu kategorii
                # if self.image_categories:
                #     self.category_menu.setCurrentIndex(1)  # Domyślne ustawienie pierwszej kategorii
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)  # Ustawienie ikony ostrzeżenia
                msg.setWindowTitle("Błąd wyrownania oświetlenia")  # Tytuł okna dialogowego
                msg.setText("Brak obrazów w folderze")  # Treść komunikatu
                msg.setStandardButtons(QMessageBox.Ok)  # Przyciski do wybrania (w tym przypadku tylko "OK")
                msg.exec_()  # Wywołanie wyświetlania okna dialogowego
                # Jeśli nie znaleziono obrazów, wyświetlenie odpowiedniego komunikatu
                # self.image_label.setText("No images in the selected folder.")
                # self.file_name_label.setText("")


    @show_loading
    def load_folder_known(self, folder_path, prepare = False, *args):
        """
        Funkcja do ładowania folderu z obrazami.
        Pobiera ścieżki do wszystkich obrazów w wybranym folderze i kategoryzuje je.
        """
        print(f"Folder path: {folder_path}")
        print(f"Temporary_folder_path: {self.temporary_results_folder}")

        if folder_path:
            # Pobieranie ścieżek do obrazów w folderze
            if prepare:
                prepare_data(folder_path, self.temporary_results_folder)
            else:
                copy_data(folder_path, self.temporary_results_folder)
            self.image_paths = [
                os.path.join(self.temporary_results_folder, f)
                for f in os.listdir(self.temporary_results_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            ]
            

            if self.image_paths:
                self.categorize_images()  # Kategoryzowanie obrazów na podstawie nazw
                self.update_category_menu()  # Aktualizacja menu kategorii
                if self.image_categories:
                    self.category_menu.setCurrentIndex(0)  # Domyślne ustawienie pierwszej kategorii
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)  # Ustawienie ikony ostrzeżenia
                msg.setWindowTitle("Błąd wyrownania oświetlenia")  # Tytuł okna dialogowego
                msg.setText("Brak obrazów w folderze")  # Treść komunikatu
                msg.setStandardButtons(QMessageBox.Ok)  # Przyciski do wybrania (w tym przypadku tylko "OK")
                msg.exec_()  # Wywołanie wyświetlania okna dialogowego
                # Jeśli nie znaleziono obrazów, wyświetlenie odpowiedniego komunikatu
                # self.image_label.setText("No images in the selected folder.")
                # self.file_name_label.setText("")

    @show_loading
    def refresh_results(self, *args):
        """
        Odświeża listę obrazów w folderze `temporary_results` i aktualizuje widok.
        """
        folder_path = self.temporary_results_folder
        new_image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

        # Dodanie tylko nowych obrazów do listy
        for image_path in new_image_paths:
            if image_path not in self.image_paths:
                self.image_paths.append(image_path)

        print(f"Images after refresh: {self.image_paths}")  # Debugowanie

        # Kategoryzuj obrazy na nowo i odśwież widok

        self.categorize_images()
        self.refresh_image_grid()

        

    def categorize_images(self):
        """
        Funkcja kategoryzuje obrazy na podstawie nazw plików.
        Wykorzystuje słowniki do przyporządkowania obrazów do kategorii i kierunków.
        """
        self.image_categories.clear()  # Czyszczenie poprzednich danych
        self.image_directions.clear()

        for category in [ "ALL", "WHITE", "IR", "UV", "result"]:
            if category not in self.image_categories:
                self.image_categories[category] = []
        
        for direction in ["ALL","NE", "NW", "SE", "SW", "all_diods", "backlight", "4 Directions", "all_no_backlight" ]:
            if direction not in self.image_directions:
                self.image_directions[direction] = []

        for image_path in self.image_paths:
            file_name = os.path.basename(image_path)  # Pobieranie nazwy pliku
            if "WHITE" in file_name.upper():
                self.image_categories["WHITE"].append(image_path)
            elif "IR" in file_name.upper():
                self.image_categories["IR"].append(image_path)
            elif "UV" in file_name.upper():
                self.image_categories["UV"].append(image_path)
            elif "result" in file_name:
                self.image_categories["result"].append(image_path)
            self.image_categories["ALL"].append(image_path)  # Każdy obraz jest w kategorii "ALL"

            # Przypisywanie obrazów do kierunków na podstawie fragmentów nazw
            if "NE" in file_name.upper():
                self.image_directions["NE"].append(image_path)
                self.image_directions["4 Directions"].append(image_path)
            elif "NW" in file_name.upper():
                self.image_directions["NW"].append(image_path)
                self.image_directions["4 Directions"].append(image_path)
            elif "SE" in file_name.upper():
                self.image_directions["SE"].append(image_path)
                self.image_directions["4 Directions"].append(image_path)
            elif "SW" in file_name.upper():
                self.image_directions["SW"].append(image_path)
                self.image_directions["4 Directions"].append(image_path)
            elif "all_no" in file_name.lower():
                self.image_directions["all_no_backlight"].append(image_path)
                print(f"Added {file_name} to all_no_backlight")
            elif "all" in file_name.lower():
                self.image_directions["all_diods"].append(image_path)
                print(f"Added {file_name} to all")
            elif "backlight" in file_name.lower():
                self.image_directions["backlight"].append(image_path)
                print(f"Added {file_name} to backlight")
            self.image_directions["ALL"].append(image_path)
            

        # Uzupełnianie pustych kategorii
        

    def update_category_menu(self):
        """
        Aktualizacja menu kategorii na podstawie dostępnych obrazów.
        """
        self.category_menu.clear()
        self.category_menu.addItems(self.image_categories.keys())
        self.configurations_menu.clear()
        self.configurations_menu.addItems(self.image_directions.keys())

    @show_loading
    def change_category(self, *args):
        """
        Zmiana wyświetlanej kategorii obrazów.
        """
        
        selected_category = self.category_menu.currentText()
        selected_configuration=self.configurations_menu.currentText()

        # Sprawdzamy, czy wybrana kategoria i konfiguracja istnieją w danych
        if selected_category in self.image_categories and selected_configuration in self.image_directions:
            # Lista obrazów dla wybranej kategorii
            category_images = set(self.image_categories[selected_category])
            
            # Lista obrazów dla wybranego kierunku
            direction_images = set(self.image_directions[selected_configuration])
            
            # Znajdowanie wspólnych obrazów dla kategorii i kierunku
            filtered_images = list(category_images & direction_images)
            
            if filtered_images:
                self.current_category = selected_category
                self.current_configuration = selected_configuration
                
                # Aktualizacja listy obrazów
                self.image_paths = filtered_images
                
                # Reset indeksu obrazów
                self.current_index = 0
                
                # Odświeżenie widoku obrazów
                self.refresh_image_grid()
            else:
                self.category_menu.setCurrentIndex(0)
                print("Brak obrazów spełniających wybrane kryteria.")
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)  # Ustawienie ikony ostrzeżenia
                msg.setWindowTitle("Błąd wyświetlenia wyników")  # Tytuł okna dialogowego
                msg.setText("Brak obrazów spełniających wybrane kryteria")  # Treść komunikatu
                msg.setStandardButtons(QMessageBox.Ok)  # Przyciski do wybrania (w tym przypadku tylko "OK")
                msg.exec_()  # Wywołanie wyświetlania okna dialogowego
                    


    def display_all_images(self):
        """
        Wyświetla wszystkie obrazy w scrollowalnym obszarze jako siatkę miniatur z centrowanym układem
        i ramkami wokół każdego segmentu siatki.
        """
        for i, image_path in enumerate(self.image_paths):
            row = i // 3  # 3 obrazy na wiersz
            col = i % 3

            # Tworzenie głównego widżetu kontenerowego z ramką
            container = QWidget(self)
            container_layout = QVBoxLayout()
            container_layout.setAlignment(Qt.AlignCenter)  # Wyśrodkowanie elementów w pionie
            container.setLayout(container_layout)
            container.setStyleSheet(
                "border: 1px solid gray; border-radius: 5px; margin: 10px; padding: 5px; background-color: #f9f9f9;"
            )  # Styl ramki i tła

            # Tworzenie QLabel dla miniatury
            label = QLabel(self)
            pixmap = QPixmap(image_path)
            label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            label.setAlignment(Qt.AlignCenter)  # Wyśrodkowanie miniatury
            label.setStyleSheet("border: 1px solid black; margin-bottom: 10px;")  # Ramka wokół miniatury
            container_layout.addWidget(label)

            # Tworzenie QLabel dla nazwy pliku
            file_name_label = QLabel(os.path.basename(image_path), self)
            file_name_label.setAlignment(Qt.AlignCenter)  # Wyśrodkowanie tekstu
            file_name_label.setStyleSheet("margin-bottom: 5px; font-size: 12px; color: #333;")  # Styl dla nazwy pliku
            container_layout.addWidget(file_name_label)

            # Tworzenie checkboxa
            checkbox = QCheckBox("Select", self)
            checkbox.setChecked(image_path in self.selected_images)  # Zaznacz, jeśli jest w liście zaznaczonych
            checkbox.stateChanged.connect(lambda state, path=image_path: self.update_selected_images(state, path))
            checkbox.setStyleSheet("margin-top: 5px;")  # Styl dla checkboxa
            container_layout.addWidget(checkbox)

            # Dodanie widżetu kontenerowego do siatki
            self.grid_layout.addWidget(container, row, col)

    @show_loading 
    def refresh_image_grid(self, *args):
        """
        Odświeża siatkę obrazów w scrollowalnym obszarze na podstawie `self.image_paths`.
        Zapewnia stały rozmiar komórek i wyrównanie obrazów do góry siatki.
        """
        miniatura_x=250
        miniatura_y=300
        current_size = self.size()
        setki = (current_size.width() // 100) * 100
        dyszki = (current_size.width() // 10) * 10
        # Obliczenie, ile razy dzielnik mieści się w setkach
        # Usuń istniejące widgety z siatki
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Liczba kolumn w siatce
        columns = dyszki // miniatura_x # Liczba kolumn w siatce
        rows = (len(self.image_paths) + columns - 1) // columns  # Liczba wierszy potrzebnych dla obrazów

        # Ustawienia układu siatki
        self.grid_layout.setSpacing(10)  # Odstępy między elementami
        self.grid_layout.setAlignment(Qt.AlignTop)  # Wyrównanie siatki do góry
        self.image_paths=sorted(self.image_paths)
        # Iteracja po obrazach
        for i, image_path in enumerate(self.image_paths):
            row = i // columns
            col = i % columns

            # Tworzenie kontenerowego widżetu dla każdego elementu
            container = QWidget(self)
            container_layout = QVBoxLayout()
            container_layout.setAlignment(Qt.AlignCenter)
            container.setLayout(container_layout)
            container.setFixedSize(miniatura_x, miniatura_y)  # Ustawienie stałego rozmiaru kontenera
            container.setStyleSheet(
                "border: 1px solid gray; border-radius: 5px; margin: 10px; padding: 5px; background-color: #f9f9f9;"
            )

            # Tworzenie QLabel dla miniatury
            label = QLabel(self)
            pixmap = QPixmap(image_path)
            label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 1px solid black; margin-bottom: 10px;")
            container_layout.addWidget(label)

            # Tworzenie QLabel dla nazwy pliku
            file_name_label = QLabel(os.path.basename(image_path), self)
            file_name_label.setAlignment(Qt.AlignCenter)
            file_name_label.setStyleSheet("margin-bottom: 5px; font-size: 12px; color: #333;")
            container_layout.addWidget(file_name_label)

            # Tworzenie checkboxa
            checkbox = QCheckBox("Select", self)
            checkbox.setChecked(image_path in self.selected_images)
            checkbox.stateChanged.connect(lambda state, path=image_path: self.update_selected_images(state, path))
            checkbox.setStyleSheet("margin-top: 5px;")
            container_layout.addWidget(checkbox)

            # Dodanie widżetu kontenerowego do siatki
            self.grid_layout.addWidget(container, row, col)

        # Wypełnij siatkę pustymi komórkami (opcjonalnie)
        total_cells = rows * columns
        for i in range(len(self.image_paths), total_cells):
            row = i // columns
            col = i % columns

            # Dodaj pusty widget
            spacer = QWidget(self)
            spacer.setFixedSize(miniatura_x, miniatura_y)  # Ustawienie stałego rozmiaru dla pustych komórek
            self.grid_layout.addWidget(spacer, row, col)



    def update_selected_images(self, state, image_path):
        """
        Aktualizuje listę zaznaczonych obrazów na podstawie stanu checkboxa.
        """
        if state == Qt.Checked and image_path not in self.selected_images:
            self.selected_images.append(image_path)
        elif state == Qt.Unchecked and image_path in self.selected_images:
            self.selected_images.remove(image_path)

        print(f"Selected images: {self.selected_images}")  # Debugowanie, pokazuje zaznaczone obrazy

    def resizeEvent(self, event):
        """
        Automatyczne dopasowanie obrazu podczas zmiany rozmiaru okna.
        """
        self.refresh_image_grid()

    @show_loading 
    def run_operation(self, *args):
        """
        Uruchamianie wybranej operacji na obrazie lub zestawie obrazów.
        """
        result = []  # Przechowywanie wyniku operacji
        self.configuration_photos.clear()
        operation = self.operations_menu.currentText()  # Pobranie wybranej operacji
        #configuration = self.configurations_menu.currentText()  # Pobranie konfiguracji
        #light = self.category_menu.currentText()  # Pobranie wybranej kategorii światła
        #self.sort_configuration(configuration, light)  # Filtrowanie zdjęć na podstawie konfiguracji i światła
        #print(f"Selected operation: {operation}, configuration: {configuration}, light: {light}")
        
        operation_function = self.operation_functions.get(operation)  # Pobranie funkcji dla operacji

        if operation == "PCA":
            num_components = 0
            # # Wyświetl okno wyboru zdjęć
            # files, _ = QFileDialog.getOpenFileNames(
            #     self,
            #     "Wybierz zdjęcia do PCA",
            #     self.temporary_results_folder,
            #     "Images (*.png *.jpg *.jpeg *.bmp)"
            # )
            # if self.selected_images is not None:
            #     self.configuration_photos = self.selected_images  # Zastąp listę zdjęć nowymi wybranymi

            # Wyświetl okno wyboru liczby głównych składowych
            num_components, ok = QInputDialog.getInt(
                self,
                "Wybór liczby głównych składowych",
                "Podaj liczbę głównych składowych (1-7):",
                value=1,
                min=1,
                max=7
            )
            if not ok:  # Jeśli użytkownik anulował
                print("Operacja PCA anulowana.")
                return

            # Oblicz PCA i wyświetl miniatury składowych
            try:
                if len(self.selected_images) < 1:
                    QMessageBox.information(self, "Error", f"Select at least 1 images to perform {operation}. You have selected {len(self.selected_images)} images.")
                elif len(self.selected_images)<num_components:
                    QMessageBox.information(self, "Error", f"Error during PCA operation: n_components={num_components} must be between 1 and number of images= {len(self.selected_images)}")
                else:
                    
                    pca_results = operation_function(self.selected_images, num_components)
                    if isinstance(pca_results, list) and len(pca_results) == num_components:
                        selected_component = self.show_pca_components(pca_results)  # Wyświetl miniatury i pozwól wybrać
                        if selected_component is None:
                            print("Użytkownik anulował wybór składowej PCA.") 
                            self.refresh_results()
                            return
                        result = pca_results[selected_component]  # Wybierz składową do wyświetlenia
                        self.refresh_results()
                    else:
                        raise ValueError("Nieprawidłowe wyniki PCA.")
            except Exception as e:
                print(f"Error during PCA operation: {e}")
                return  
        elif operation == "Neutralization":
            # Wybór folderu ze zdjęciami białej kartki
            white_sheets_folder = QFileDialog.getExistingDirectory(self, "Wybierz folder ze zdjęciami białej kartki")
            print(f"Wybrany folder: {white_sheets_folder}")  # DEBUG: log do konsoli
            
            if not white_sheets_folder or white_sheets_folder.strip() == "":
                QMessageBox.warning(self, "Błąd", "Nie wybrano folderu ze zdjęciami białej kartki.")
                return

            # Folder na macierze kalibracyjne
            calibration_folder = os.path.join(self.temporary_results_folder, "calibration_matrices")
            # Folder na wynikowe obrazy
            output_folder = self.temporary_results_folder

                # Utwórz folder na wynikowe obrazy, jeśli nie istnieje
            if not os.path.exists(output_folder+"2"):
                os.makedirs(output_folder+"2")
            new_output_folder = output_folder+"2"

            try:
                print(white_sheets_folder, calibration_folder, output_folder)
                print(f"operation_function: {operation_function}")
                new_folder = operation_function(white_sheets_folder, calibration_folder, output_folder, new_output_folder)
                  # Przykład funkcji ustawiającej folder w GUI
                # QMessageBox.information(self, "Sukces")

                reply = QMessageBox.question(
                self,
                "Zmiana folderu",
                f"Neutralizacja zakończona. Wyniki zapisane w folderze: {new_output_folder}\n\n"
                "Czy chcesz zmienić wyświetlany folder na ten z neutralizacją?",
                QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self.load_folder_known(new_folder)
            except Exception as e:
                QMessageBox.critical(self, "Błąd", "Wystąpił błąd, wybierz folder ze zdjęciami białego wzorca.")
                print(f"Wystąpił problem: {str(e)}")
                tb = traceback.format_exc()
                print("Traceback:")
                print(tb)
            return

        elif operation == "Addition":
            try:
                result = operation_function(self.selected_images)  # Wykonanie operacji

                if isinstance(result, np.ndarray):  # Jeśli wynik to obraz
                    self.show_result_in_new_window(result)  # Wyświetlenie wyniku w nowym oknie
                    os.makedirs(f"results_{date.today()}", exist_ok=True)  # Tworzenie folderu na wyniki
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    image_path=f"{self.temporary_results_folder}/result_{operation}_{timestamp}.png"
                    cv2.imwrite(f"results_{date.today()}/result_{operation}_{timestamp}.png", result)  # Zapis wyniku
                    cv2.imwrite(image_path, result)  # Zapis wyniku
                    self.image_categories["result"].append(image_path)
                    print(f"image categories result {self.image_categories[4]}")
                    
                else:
                    print(f"Operation result '{operation}': {result}")
                self.refresh_results()
            except Exception as e:
                print(f"Error during operation '{operation}': {e}")
            else:
                print(f"Operation '{operation}' is not supported.")
        elif operation == "Erosion and Dilation":
            iterations_e = 0
            iterations_d = 0

            # Pobranie liczby iteracji erozji
            iterations_e, ok = QInputDialog.getInt(
                self,
                "Number of erosion iterations",
                "Select the number of erosion iterations: (1-5):",
                value=1,
                min=1,
                max=5
            )
            if not ok:  # Jeśli użytkownik anulował
                print("Operation canceled.")
                return

            # Pobranie liczby iteracji dylatacji
            iterations_d, ok = QInputDialog.getInt(
                self,
                "Number of dilation iterations",
                "Select the number of dilation iterations: (1-5):",
                value=1,
                min=1,
                max=5
            )
            if not ok:  # Jeśli użytkownik anulował
                print("Operation canceled.")
                return

            try:
                # Sprawdzenie, czy wybrano dokładnie jeden obraz
                if len(self.selected_images) != 1:
                    QMessageBox.information(
                        self,
                        "Error",
                        f"Select 1 image to perform {operation}. You have selected {len(self.selected_images)} images."
                    )
                else:

                    # Wykonanie operacji
                    result = operation_function(self.selected_images, iterations_d, iterations_e)

                    # Sprawdzenie, czy wynik to obraz (np. macierz numpy)
                    if isinstance(result, np.ndarray):
                        # Wyświetlenie wyniku w nowym oknie
                        self.show_result_in_new_window(result)  # Wyświetlenie wyniku w nowym oknie
                        os.makedirs(f"results_{date.today()}", exist_ok=True)  # Tworzenie folderu na wyniki
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        image_path=f"{self.temporary_results_folder}/result_erosion_dilation_{timestamp}.png"
                        cv2.imwrite(f"results_{date.today()}/result_{operation}_{timestamp}.png", result)  # Zapis wyniku
                        cv2.imwrite(image_path, result)  # Zapis wyniku
                        self.image_categories["result"].append(image_path)
                        print(f"Image saved to: {image_path}")
                    else:
                        print(f"Operation result '{operation}': {result}")

                # Odświeżenie wyników
                self.refresh_results()

            except Exception as e:
                tb = traceback.format_exc()
                print(f"Error during operation '{operation}': {e}")
                print("Traceback:")
                print(tb)


        elif operation == "Subtraction":
            try:
                if len(self.selected_images) != 2:
                    QMessageBox.information(self, "Error", f"Select 2 images to perform {operation}. You have selected {len(self.selected_images)} images.")
                else:
                    result = operation_function(self.selected_images)  # Wykonanie operacji

                    if isinstance(result, np.ndarray):  # Jeśli wynik to obraz
                        self.show_result_in_new_window(result)  # Wyświetlenie wyniku w nowym oknie
                        os.makedirs(f"results_{date.today()}", exist_ok=True)  # Tworzenie folderu na wyniki
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        image_path=f"{self.temporary_results_folder}/result_{operation}_{timestamp}.png"
                        cv2.imwrite(f"results_{date.today()}/result_{operation}_{timestamp}.png", result)  # Zapis wyniku
                        cv2.imwrite(image_path, result)  # Zapis wyniku
                        self.image_categories["result"].append(image_path)
                    else:
                        print(f"Operation result '{operation}': {result}")
                    self.refresh_results()
            except Exception as e:
                print(f"Error during operation '{operation}': {e}")
            else:
                print(f"Operation '{operation}' is not supported.")

        elif operation == "Histogram Equalization" or operation == "Edge Detection Canny"  or operation == "Edge Detection Sobel" or operation== "CLAHE":
            try:
                if len(self.selected_images) != 1:
                    QMessageBox.information(self, "Error", f"Select 1 image to perform {operation}. You have selected {len(self.selected_images)} images.")
                else:
                    result = operation_function(self.selected_images)  # Wykonanie operacji

                    if isinstance(result, np.ndarray):  # Jeśli wynik to obraz
                        self.show_result_in_new_window(result)  # Wyświetlenie wyniku w nowym oknie
                        os.makedirs(f"results_{date.today()}", exist_ok=True)  # Tworzenie folderu na wyniki
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        image_path=f"{self.temporary_results_folder}/result_{operation}_{timestamp}.png"
                        cv2.imwrite(f"results_{date.today()}/result_{operation}_{timestamp}.png", result)  # Zapis wyniku
                        cv2.imwrite(image_path, result)  # Zapis wyniku
                        self.image_categories["result"].append(image_path)
                    else:
                        print(f"Operation result '{operation}': {result}")
                    self.refresh_results()
            except Exception as e:
                print(f"Error during operation '{operation}': {e}")
            else:
                print(f"Operation '{operation}' is not supported.")

        elif operation == "Text Detection":
            try:
                if len(self.selected_images) != 1:
                    QMessageBox.information(self, "Error", f"Select 1 image to perform {operation}. You have selected {len(self.selected_images)} images.")
                else:
                    # Wyświetlenie okna dialogowego z obracaniem obrazu
                    image_path = self.selected_images[0]
                    rotate_dialog = RotateImageDialog(image_path, self)
                    if rotate_dialog.exec_() == QDialog.Accepted:
                        # Pobranie obróconego obrazu
                        rotated_image = rotate_dialog.get_rotated_image()
                        rotation_angle = rotate_dialog.get_rotation_angle()

                        # Zapis tymczasowego obróconego obrazu
                        temp_path = f"{self.temporary_results_folder}/temp_rotated_image.png"
                        cv2.imwrite(temp_path, rotated_image)

                        # Wywołanie funkcji operacji z nowym obrazem
                        result, wyniki = operation_function([temp_path])

                        # Przywrócenie obrazu do pierwotnej orientacji
                        if rotation_angle != 0:
                            h, w = result.shape[:2]
                            center = (w // 2, h // 2)
                            inverse_rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
                            result = cv2.warpAffine(result, inverse_rotation_matrix, (w, h))

                        # Wyświetlenie wykrytego tekstu w oknie dialogowym
                        detected_text = "\n".join([f"{i + 1}. {text} (Prawdopodobieństwo: {prob:.2f})" for i, (_, text, prob) in enumerate(wyniki)])
                        QMessageBox.information(self, "Detected Text", f"Wykryty tekst:\n{detected_text}")

                        if isinstance(result, np.ndarray):  # Jeśli wynik to obraz
                            self.show_result_in_new_window(result)  # Wyświetlenie wyniku w nowym oknie
                            os.makedirs(f"results_{date.today()}", exist_ok=True)  # Tworzenie folderu na wyniki
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            image_path = f"results_{date.today()}/result_{operation}_{timestamp}.png"
                            cv2.imwrite(image_path, result)  # Zapis wyniku
                            self.image_categories["result"].append(image_path)
                        else:
                            print(f"Operation result '{operation}': {result}")
                        self.refresh_results()
            except Exception as e:
                print(f"Error during operation '{operation}': {e}")
                tb = traceback.format_exc()
                print(f"Error during operation '{operation}': {e}")
                print("Traceback:")
                print(tb)




        elif operation == "Bayes Classification":
            try:
                print("Bayes")
                if len(self.selected_images) != 3:
                    QMessageBox.information(self, "Error", f"Select 3 images to perform {operation}. You have selected {len(self.selected_images)} images.")
                else:

                    result = operation_function(self.selected_images)  # Wykonanie operacji

                    if isinstance(result, np.ndarray):  # Jeśli wynik to obraz
                        # self.show_result_in_new_window(result)  # Wyświetlenie wyniku w nowym oknie
                        os.makedirs(f"results_{date.today()}", exist_ok=True)  # Tworzenie folderu na wyniki
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        image_path=f"{self.temporary_results_folder}/result_{operation}_{timestamp}.png"
                        cv2.imwrite(f"results_{date.today()}/result_{operation}_{timestamp}.png", result)  # Zapis wyniku
                        cv2.imwrite(image_path, result)  # Zapis wyniku
                        self.image_categories["result"].append(image_path)
                        print(image_path)
                        result_image = cv2.imread(image_path)
                        self.show_result_in_new_window(result_image)
                    else:
                        print(f"Operation result '{operation}': {result}")
                    self.refresh_results()
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Error during operation '{operation}': {e}")
                print("Traceback:")
                print(tb)
            else:
                print(f"Operation '{operation}' is not supported.")

        self.category_menu.setCurrentIndex(0)
        self.configurations_menu.setCurrentIndex(0)

        # if operation_function:
        #     try:
        #         if operation != "PCA" and operation != "Neutralization":
        #             result = operation_function(self.configuration_photos)  # Wykonanie operacji

        #         if isinstance(result, np.ndarray):  # Jeśli wynik to obraz
        #             self.show_result_in_new_window(result)  # Wyświetlenie wyniku w nowym oknie
        #             os.makedirs(f"results_{date.today()}", exist_ok=True)  # Tworzenie folderu na wyniki
        #             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        #             cv2.imwrite(f"results_{date.today()}/result_{operation}_{timestamp}.png", result)  # Zapis wyniku
        #             cv2.imwrite(f"{self.temporary_results_folder}/result_{operation}_{timestamp}.png", result)  # Zapis wyniku
        #         else:
        #             print(f"Operation result '{operation}': {result}")
        #         self.refresh_results()
        #     except Exception as e:
        #         print(f"Error during operation '{operation}': {e}")
        # else:
        #     print(f"Operation '{operation}' is not supported.")
    @show_loading 
    def show_pca_components(self, components, *args):
        """
        Wyświetla wszystkie główne składowe PCA w formie siatki miniatur, pozwala użytkownikowi wybrać jedną,
        podglądać dowolne składowe i ostatecznie zatwierdzić wybór.

        Args:
            components (list): Lista obrazów reprezentujących składowe PCA.

        Returns:
            None
        """
        selected_result = None

        while True:
            dialog = QDialog(self)
            dialog.setWindowTitle("Wybór głównej składowej PCA")
            dialog.resize(800, 600)  # Ustawienie większego rozmiaru okna dialogowego

            layout = QVBoxLayout()

            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)  # Dopasowanie scrolla do rozmiaru zawartości
            scroll_widget = QWidget()

            grid_layout = QGridLayout()
            scroll_widget.setLayout(grid_layout)

            # Dodaj miniatury do układu siatki (3 obrazy w wierszu)
            for idx, component in enumerate(components):
                # Utwórz label z obrazem
                label = QLabel()
                pixmap = QPixmap.fromImage(QImage(
                    component.data, component.shape[1], component.shape[0], component.strides[0], QImage.Format_Grayscale8
                ))
                label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))  # Większe miniatury (200x200)

                # Przyciski wyboru
                button = QPushButton(f"Składowa {idx + 1}")
                button.clicked.connect(lambda _, i=idx: dialog.done(i))

                # Dodaj do układu siatki
                row = idx // 3
                col = idx % 3
                grid_layout.addWidget(label, row * 2, col)
                grid_layout.addWidget(button, row * 2 + 1, col)

            scroll_area.setWidget(scroll_widget)
            layout.addWidget(scroll_area)

            cancel_button = QPushButton("Anuluj")
            cancel_button.clicked.connect(lambda: dialog.done(-1))
            layout.addWidget(cancel_button)

            dialog.setLayout(layout)
            result = dialog.exec_()

            if result >= 0:  # Jeśli wybrano składową
                # Pokaż wybraną składową w nowym oknie
                component_view_dialog = QDialog(self)
                component_view_dialog.setWindowTitle(f"Podgląd składowej {result + 1}")
                component_view_dialog.resize(800, 600)

                component_layout = QVBoxLayout()
                component_label = QLabel()
                component_pixmap = QPixmap.fromImage(QImage(
                    components[result].data, components[result].shape[1], components[result].shape[0],
                    components[result].strides[0], QImage.Format_Grayscale8
                ))
                component_label.setPixmap(component_pixmap.scaled(600, 600, Qt.KeepAspectRatio))
                component_layout.addWidget(component_label)

                # Flaga dla zatwierdzenia wyboru
                confirmed = [False]  # Używamy listy, aby mutowalność pozwoliła na zmianę w funkcji lambda

                # Przycisk "Wróć do listy"
                back_button = QPushButton("Wróć do listy")
                back_button.clicked.connect(lambda: component_view_dialog.reject())  # Zamknięcie okna podglądu bez zatwierdzania

                # Przycisk "Zatwierdź wybór"
                confirm_button = QPushButton("Zatwierdź wybór")
                confirm_button.clicked.connect(lambda: [component_view_dialog.accept(), confirmed.__setitem__(0, True)])

                component_layout.addWidget(back_button)
                component_layout.addWidget(confirm_button)

                component_view_dialog.setLayout(component_layout)
                choice = component_view_dialog.exec_()

                if choice == QDialog.Accepted and confirmed[0]:  # Jeśli wybór został zatwierdzony
                    selected_result = components[result]

                    if selected_result is not None:
                        try:

                            # Tworzenie folderu na wyniki
                            result_folder = f"results_{date.today()}"
                            os.makedirs(result_folder, exist_ok=True)

                            # Generowanie nazwy pliku z timestampem
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            result_path = os.path.join(result_folder, f"result_PCA_{timestamp}.png")
                            temp_result_path = os.path.join(self.temporary_results_folder, f"result_PCA_{timestamp}.png")

                            self.image_categories["result"].append(temp_result_path)

                            # Zapisywanie wyników
                            if not cv2.imwrite(result_path, selected_result):
                                raise IOError(f"Nie udało się zapisać wyniku w {result_path}.")
                            if not cv2.imwrite(temp_result_path, selected_result):
                                raise IOError(f"Nie udało się zapisać wyniku w {temp_result_path}.")

                            # Wyświetlanie wyniku
                            self.show_result_in_new_window(selected_result)

                        except Exception as e:
                            print(f"Błąd podczas zapisu wyniku PCA: {e}")
                    return True # Wyjdź z pętli i zakończ wybór
                elif choice == QDialog.Rejected:  # Jeśli użytkownik wrócił do listy
                    continue  # Powrót do listy
            else:
                # Użytkownik anulował
                selected_result = None
                return None

    @show_loading
    def run_predefined_operation(self, *args):
        # Tworzenie okna dialogowego
        dialog = QDialog()
        dialog.setWindowTitle("Wybierz Predefiniowaną Operację")

        # Layout okna dialogowego
        layout = QVBoxLayout()

        # Nagłówek
        label = QLabel("Wybierz jedną z predefiniowanych operacji:")
        layout.addWidget(label)

        # Przycisk dla Operacji 1
        btn_operation_1 = QPushButton("PCA (white, IR), Addition + CLAHE")
        btn_operation_1.clicked.connect(lambda: self.execute_operation(dialog, 1))
        layout.addWidget(btn_operation_1)

        # Przycisk dla Operacji 2
        btn_operation_2 = QPushButton("NE-SW, NW-SE, Addition + CLAHE")
        btn_operation_2.clicked.connect(lambda: self.execute_operation(dialog, 2))
        layout.addWidget(btn_operation_2)

        # Przycisk dla Operacji 3
        btn_operation_3 = QPushButton("PCA (white) +  CLAHE")
        btn_operation_3.clicked.connect(lambda: self.execute_operation(dialog, 3))
        layout.addWidget(btn_operation_3)

         # Przycisk dla Operacji 4
        btn_operation_4 = QPushButton("NE-SW, NW-SE (white, IR, UV) +  CLAHE")
        btn_operation_4.clicked.connect(lambda: self.execute_operation(dialog, 4))
        layout.addWidget(btn_operation_4)

        # Ustawienie layoutu i wyświetlenie okna dialogowego
        dialog.setLayout(layout)
        dialog.exec_()

    def execute_operation(self, dialog, number):
        """
        Wykonuje predefiniowaną operację na podstawie wybranego numeru.

        Args:
            dialog (QDialog): Dialog, który wywołał operację (zamyka go po wyborze).
            number (int): Numer predefiniowanej operacji do wykonania.
        """
        dialog.close()  # Zamknięcie dialogu po wybraniu operacji
        
        try:
            if number == 1:
                # PCA (white, UV), Addition + CLAHE
                print("Running pre-defined operation 1: PCA (white, IR), Addition + CLAHE")

                # Pobranie zdjęć white i UV
                white_images = []
                white_images.extend(self.get_image_by_category("WHITE", "4 Directions") or [])
                white_images.extend(self.get_image_by_category("WHITE", "all_no_backlight") or [])
                uv_images = []
                uv_images.extend(self.get_image_by_category("IR", "4 Directions") or [])
                uv_images.extend(self.get_image_by_category("IR", "all_no_backlight") or [])

                if not white_images or not uv_images:
                    QMessageBox.warning(self, "Błąd", "Brak wymaganych zdjęć (WHITE, UV).")
                    return

                # Wykonanie PCA
                pca_result_w = self.operation_functions.get("PCA")(white_images, 4)
                if pca_result_w and isinstance(pca_result_w, list):
                    selected_component = self.show_pca_components(pca_result_w)
                    if selected_component is not None:
                        selected_pca_w = pca_result_w[selected_component]
                    else:
                        print("PCA operation canceled.")
                        return
                else:
                    raise ValueError("PCA operation failed or returned invalid results.")
                
                pca_result_u = self.operation_functions.get("PCA")(uv_images, 4)
                if pca_result_u and isinstance(pca_result_u, list):
                    selected_component = self.show_pca_components(pca_result_u)
                    if selected_component is not None:
                        selected_pca_u = pca_result_u[selected_component]
                    else:
                        print("PCA operation canceled.")
                        return
                else:
                    raise ValueError("PCA operation failed or returned invalid results.")

                cv2.imwrite(f"{self.temporary_results_folder}/temporary_pca_w.png", selected_pca_w)
                cv2.imwrite(f"{self.temporary_results_folder}/temporary_pca_u.png", selected_pca_u)
                # Addition
                to_add=[]
                to_add.append(f"{self.temporary_results_folder}/temporary_pca_w.png")
                to_add.append(f"{self.temporary_results_folder}/temporary_pca_u.png")
                addition_result  = cv2.addWeighted(selected_pca_u, 0.5, selected_pca_w, 0.5, 0)


                # Wyświetlenie i zapis wyniku
                self.show_result_in_new_window(addition_result)
                self.save_result(addition_result, "Predefined_Operation_1")
                self.refresh_results()

            elif number == 2:
                # NE-SW, NW-SE, Addition, Equalization
                print("Running pre-defined operation 2: NE-SW, NW-SE, Addition + CLAHE")

                # Pobranie zdjęć NE-SW i NW-SE
                ne_images = []
                nw_images=[]
                se_images=[]
                sw_images=[]
                
                ne_images.extend(self.get_image_by_category("WHITE", "NE") or [])
                nw_images.extend(self.get_image_by_category("WHITE", "NW") or [])
                se_images.extend(self.get_image_by_category("WHITE", "SE") or [])
                sw_images.extend(self.get_image_by_category("WHITE", "SW") or [])
                if not ne_images or not nw_images or not se_images or not sw_images:
                    QMessageBox.warning(self, "Błąd", "Brak wystarczających obrazów dla kategorii (NE, NW, SE, SW).")
                    return
                dif1=[ne_images[0], sw_images[0]]
                dif2=[nw_images[0], se_images[0]]
                dif1_res= self.operation_functions.get("Subtraction")(dif1)
                dif2_res= self.operation_functions.get("Subtraction")(dif2)

                cv2.imwrite(f"{self.temporary_results_folder}/temporary_dif1.png", dif1_res)
                cv2.imwrite(f"{self.temporary_results_folder}/temporary_dif2.png", dif2_res)
                add_final=[f"{self.temporary_results_folder}/temporary_dif2.png", f"{self.temporary_results_folder}/temporary_dif1.png" ]
                #addition_result = self.operation_functions.get("Addition")(add_final)
                addition_result  = cv2.addWeighted(dif1_res, 0.5, dif2_res, 0.5, 0)
                cv2.imwrite(f"{self.temporary_results_folder}/temporary_add_final.png", addition_result)
                eq=[f"{self.temporary_results_folder}/temporary_add_final.png"]
                equalization_final = self.operation_functions.get("CLAHE")(eq)
                # Wyświetlenie i zapis wyniku
                self.show_result_in_new_window(equalization_final)
                self.save_result(equalization_final, "Predefined_Operation_2")
                self.refresh_results()


            elif number == 3:
                # PCA (white, UV), Addition + CLAHE
                print("Running pre-defined operation 1: PCA (white) + CLAHE")

                # Pobranie zdjęć white i UV
                white_images = []
                white_images.extend(self.get_image_by_category("WHITE", "4 Directions") or [])
                white_images.extend(self.get_image_by_category("WHITE", "all_no_backlight") or [])
                white_images.extend(self.get_image_by_category("WHITE", "all") or [])

                #uv_images = self.get_image_by_category("IR")
                if not white_images:
                    QMessageBox.warning(self, "Błąd", "Brak wymaganych zdjęć (WHITE, UV).")
                    return

                # Wykonanie PCA
                pca_result_w = self.operation_functions.get("PCA")(white_images, 4)
                if pca_result_w and isinstance(pca_result_w, list):
                    selected_component = self.show_pca_components(pca_result_w)
                    if selected_component is not None:
                        selected_pca_w = pca_result_w[selected_component]
                    else:
                        print("PCA operation canceled.")
                        return
                else:
                    raise ValueError("PCA operation failed or returned invalid results.")
                
                # pca_result_u = self.operation_functions.get("PCA")(uv_images, 4)
                # if pca_result_u and isinstance(pca_result_u, list):
                #     selected_component = self.show_pca_components(pca_result_u)
                #     if selected_component is not None:
                #         selected_pca_u = pca_result_u[selected_component]
                #     else:
                #         print("PCA operation canceled.")
                #         return
                # else:
                #     raise ValueError("PCA operation failed or returned invalid results.")

                cv2.imwrite(f"{self.temporary_results_folder}/temporary_pca_w.png", selected_pca_w)
                to_clahe=[]
                to_clahe.append(f"{self.temporary_results_folder}/temporary_pca_w.png")
                clahe_result = self.operation_functions.get("CLAHE")(to_clahe)


                # Wyświetlenie i zapis wyniku
                self.show_result_in_new_window(clahe_result)
                self.save_result(clahe_result, "Predefined_Operation_3")
                self.refresh_results()
            elif number == 4:
                # NE-SW, NW-SE, Addition, Equalization
                print("Running pre-defined operation 4: NE-SW, NW-SE, Addition, Equalization")

                # Pobranie zdjęć NE-SW i NW-SE
                ne_images = []
                nw_images=[]
                se_images=[]
                sw_images=[]
                ne_images.extend(self.get_image_by_category("ALL", "NE") or [])
                nw_images.extend(self.get_image_by_category("ALL", "NW") or [])
                se_images.extend(self.get_image_by_category("ALL", "SE") or [])
                sw_images.extend(self.get_image_by_category("ALL", "SW") or [])

                if not ne_images or not nw_images or not se_images or not sw_images:
                    QMessageBox.warning(self, "Błąd", "Brak wystarczających obrazów dla kategorii (NE, NW, SE, SW).")
                    return


                addition_result = self.operation_functions.get("Addition")(ne_images)
                cv2.imwrite(f"{self.temporary_results_folder}/temporary_added_ne.png", addition_result)
                addition_result = self.operation_functions.get("Addition")(nw_images)
                cv2.imwrite(f"{self.temporary_results_folder}/temporary_added_nw.png", addition_result)
                addition_result = self.operation_functions.get("Addition")(se_images)
                cv2.imwrite(f"{self.temporary_results_folder}/temporary_added_se.png", addition_result)
                addition_result = self.operation_functions.get("Addition")(sw_images)
                cv2.imwrite(f"{self.temporary_results_folder}/temporary_added_sw.png", addition_result)


                
                # Histogram Equalization
                dif1=[f"{self.temporary_results_folder}/temporary_added_ne.png", f"{self.temporary_results_folder}/temporary_added_sw.png" ]
                dif2=[f"{self.temporary_results_folder}/temporary_added_nw.png", f"{self.temporary_results_folder}/temporary_added_se.png" ]
                
                dif1_res= self.operation_functions.get("Subtraction")(dif1)
                dif2_res= self.operation_functions.get("Subtraction")(dif2)

                cv2.imwrite(f"{self.temporary_results_folder}/temporary_dif1.png", dif1_res)
                cv2.imwrite(f"{self.temporary_results_folder}/temporary_dif2.png", dif2_res)
                add_final=[f"{self.temporary_results_folder}/temporary_dif2.png", f"{self.temporary_results_folder}/temporary_dif1.png" ]
                addition_result = self.operation_functions.get("Addition")(add_final)
                cv2.imwrite(f"{self.temporary_results_folder}/temporary_add_final.png", addition_result)
                eq=[f"{self.temporary_results_folder}/temporary_add_final.png"]
                equalization_final = self.operation_functions.get("CLAHE")(eq)
                # Wyświetlenie i zapis wyniku
                self.show_result_in_new_window(equalization_final)
                self.save_result(equalization_final, "Predefined_Operation_4")
                self.refresh_results()
            else:
                print(f"Unknown predefined operation number: {number}")

        except Exception as e:
            print(f"Error during predefined operation {number}: {e}")
            traceback.print_exc()

    def save_result(self, result, operation_name):
        """
        Zapisuje wynik operacji w folderze wynikowym.

        Args:
            result (np.ndarray): Obraz wynikowy do zapisania.
            operation_name (str): Nazwa operacji do identyfikacji wyniku.
        """
        try:
            os.makedirs(f"results_{date.today()}", exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_path = f"results_{date.today()}/result_{operation_name}_{timestamp}.png"
            cv2.imwrite(result_path, result)
            print(f"Result saved to: {result_path}")
        except Exception as e:
            print(f"Error saving result for operation '{operation_name}': {e}")

    def get_image_by_category(self, category, directions):
        
        selected_category = category
        selected_configuration= directions
        # Sprawdzamy, czy wybrana kategoria i konfiguracja istnieją w danych
        if selected_category in self.image_categories and selected_configuration in self.image_directions:
            # Lista obrazów dla wybranej kategorii
            category_images = set(self.image_categories[selected_category])
            
            # Lista obrazów dla wybranego kierunku
            direction_images = set(self.image_directions[selected_configuration])
            
            # Znajdowanie wspólnych obrazów dla kategorii i kierunku
            filtered_images = list(category_images & direction_images)
            
            if filtered_images:
                return filtered_images
            else:
                print("Brak obrazów spełniających wybrane kryteria.")

                

    def show_result_in_new_window(self, image, *args):
        """
        Wyświetlenie wyniku operacji w nowym oknie dialogowym.
        """
        self.result_window = ImageWindow(image, self)  # Tworzenie nowego okna dla wyniku
        self.result_window.show()  # Wyświetlenie okna

        
    def closeEvent(self, event):
        """Metoda wywoływana przy zamknięciu okna."""
        reply = QMessageBox.question(
            self,
            "Zamknij aplikację",
            "Czy na pewno chcesz zamknąć aplikację?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Spróbuj usunąć wszystkie foldery i pliki zawierające "temporary" w nazwie
            try:
                base_directory = os.getcwd()  # Katalog roboczy
                for item in os.listdir(base_directory):
                    item_path = os.path.join(base_directory, item)
                    if "temporary" in item.lower():
                        # Sprawdź, czy to folder
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            print(f"Folder {item_path} został usunięty.")
                        # Jeśli to plik
                        elif os.path.isfile(item_path):
                            os.remove(item_path)
                            print(f"Plik {item_path} został usunięty.")
                    else:
                        print(f"Pominięto: {item_path} (nie pasuje do kryteriów).")
            except Exception as e:
                print(f"Nie udało się usunąć niektórych elementów: {e}")

            event.accept()  # Zamknij aplikację
        else:
            event.ignore()  # Anuluj zamknięcie




if __name__ == "__main__":
    # Główna funkcja uruchamiająca aplikację
    app = QApplication(sys.argv)  # Tworzenie instancji aplikacji PyQt
    viewer = MainWindow()  # Tworzenie głównego okna aplikacji
    viewer.show()  # Wyświetlenie głównego okna
    sys.exit(app.exec_())  # Uruchomienie pętli zdarzeń aplikacji
    

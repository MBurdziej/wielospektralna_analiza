import cv2
import numpy as np
import os
import easyocr
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib.widgets import RectangleSelector, Button
from skimage.color import rgb2gray
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QApplication
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QColor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import traceback
from functools import wraps
import shutil

def normalize_image(image, *args):
        return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def add_operation(images, *args):
    """
    Function to perform addition of images for a selected light configuration.
    Accepts a list of image paths.

    Args:
        images (list): List of image paths.
        light (str): Type of light ('WHITE', 'IR', 'UV', or None for all).

    Returns:
        np.ndarray: Resulting sum of images.
    """
    # Initialize matrix to store the sum of images
    result_image = None

    # Read and normalize images, then add them together
    for image_path in images:
        if image_path:
            # Read the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")

            # Normalize image
            normalized_image = normalize_image(image)

            # Add the image to the result
            if result_image is None:
                result_image = normalized_image
            else:
                result_image += normalized_image

    if result_image is None:
        raise ValueError("No available images to add.")
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_add = clahe.apply((result_image * 255).astype(np.uint8))

    # Scale the result to the range 0-255 and return as a NumPy array
    return enhanced_add


def subtract_operation(images, *args):
    """
    Function to calculate the difference between images for a selected light configuration.
    Accepts a list of image paths.

    Args:
        images (list): List of image paths.

    Returns:
        np.ndarray: Resulting difference between images.
    """
    image1 = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(images[1], cv2.IMREAD_GRAYSCALE)
    # Calculate differences if images are available
    diff = cv2.absdiff(image1, image2)


    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_diff = clahe.apply((diff * 255).astype(np.uint8))

    # Return the result as a NumPy array
    return enhanced_diff


def equalize_histogram(image_path, *args):
    """
    Function to perform histogram equalization on grayscale or color images.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Image with equalized histogram.
    """
    # Wczytanie obrazu
    image = cv2.imread(image_path[0])
    
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")
    
    # Sprawdzamy, czy obraz jest w skali szarości
    if len(image.shape) == 2:  # Grayscale
        equalized_image = cv2.equalizeHist(image)
    else:  # RGB
        # Konwersja obrazu na przestrzeń kolorów HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Rozdzielanie kanałów HSV
        h, s, v = cv2.split(hsv_image)
        
        # Równoważenie histogramu tylko na kanale 'V' (jasność)
        v_eq = cv2.equalizeHist(v)
        
        # Łączenie z powrotem z pozostałymi kanałami (H i S)
        equalized_hsv = cv2.merge((h, s, v_eq))
        
        # Konwersja z powrotem do przestrzeni RGB
        equalized_image = cv2.cvtColor(equalized_hsv, cv2.COLOR_HSV2BGR)
    
    return equalized_image



def detect_edges_sobel(images, *args):
    """
    Function to perform edge detection using the Sobel operator with contrast enhancement.

    Args:
        images (list): List of image paths.

    Returns:
        np.ndarray: Image with detected edges.
    """
    # Read the first image in grayscale
    image = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    
    # Normalize the image for consistent intensity
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(normalized_image)
    
    # Compute the Sobel gradients in x and y directions
    sobel_x = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the gradient magnitude
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # Normalize the result to 8-bit range for visualization
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return sobel_magnitude


def detect_edges_canny(images, *args):
    """
    Function to perform edge detection with contrast enhancement.

    Args:
        images (list): List of image paths.

    Returns:
        np.ndarray: Image with detected edges.
    """


    image = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(normalized_image)
    edges = cv2.Canny(enhanced_image, threshold1=50, threshold2=150)
    return edges



def perform_pca_analysis(images, num_components=5, *args):
    """
    Przeprowadza analizę PCA na podanych obrazach i zwraca wszystkie składowe jako obrazy.

    Args:
        images (list): Lista ścieżek do obrazów.
        num_components (int): Liczba głównych składowych do analizy (domyślnie 5).

    Returns:
        list: Lista obrazów reprezentujących główne składowe.
    """
    if len(images) == 0:
        raise ValueError("Brak obrazów do analizy PCA.")

    # Wczytaj obrazy w skali szarości
    loaded_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in images]
    if any(img is None for img in loaded_images):
        raise ValueError("Nie udało się wczytać jednego lub więcej obrazów.")

    # Sprawdź czy wszystkie obrazy mają takie same wymiary
    h, w = loaded_images[0].shape
    if not all(img.shape == (h, w) for img in loaded_images):
        raise ValueError("Wszystkie obrazy muszą mieć takie same wymiary.")

    # Przygotowanie danych do PCA
    data = np.array([img.flatten() for img in loaded_images])

    # Wykonanie PCA
    pca = PCA(n_components=num_components)
    pca.fit(data)

    # Generowanie obrazów składowych
    components = []
    for component in pca.components_:
        reshaped = component.reshape(h, w)
        normalized = cv2.normalize(reshaped, None, 0, 255, cv2.NORM_MINMAX)
        components.append(normalized.astype(np.uint8))

    return components


def enhance_text_with_morphology(image_path, iterations_d=2, iterations_e=1):
    """
    Enhances text on an image using erosion and dilation.
    
    Args:
        image_path (str): Path to the input image.
        
    Returns:
        np.ndarray: Image after morphological processing.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path[0], cv2.IMREAD_GRAYSCALE)
    
    # Normalize the image for consistent intensity
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(normalized_image)
    
    # Thresholding to create a binary image
    _, binary_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Define a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)
    
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)

    dilated_image = cv2.dilate(eroded_image, kernel, iterations=iterations_d)
    # Apply erosion to remove noise
    eroded_image = cv2.erode(dilated_image, kernel, iterations=iterations_e)
    
    # Apply dilation to enhance text
    #dilated_image = cv2.dilate(eroded_image, kernel, iterations=iterations_d)
    # Normalize the result image to ensure consistent intensity
    result_image = cv2.normalize(eroded_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Return the processed image
    return result_image


def clahe(images, *args):
    image = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(normalized_image)
    return enhanced_image


def predefined(images):
    # Normalize image

    # Initialize dictionary for directions
    direction_images = {"NE": None, "NW": None, "SE": None, "SW": None}

    # Assign images to directions based on their names
    for image_path in images:
        if "NE" in image_path.upper():
            direction_images["NE"] = image_path
        elif "NW" in image_path.upper():
            direction_images["NW"] = image_path
        elif "SE" in image_path.upper():
            direction_images["SE"] = image_path
        elif "SW" in image_path.upper():
            direction_images["SW"] = image_path

    # Read available images
    img_NE = cv2.imread(direction_images["NE"], cv2.IMREAD_GRAYSCALE) if direction_images["NE"] else None
    img_NW = cv2.imread(direction_images["NW"], cv2.IMREAD_GRAYSCALE) if direction_images["NW"] else None
    img_SE = cv2.imread(direction_images["SE"], cv2.IMREAD_GRAYSCALE) if direction_images["SE"] else None
    img_SW = cv2.imread(direction_images["SW"], cv2.IMREAD_GRAYSCALE) if direction_images["SW"] else None

    # Normalize available images
    img_NE = normalize_image(img_NE) if img_NE is not None else None
    img_NW = normalize_image(img_NW) if img_NW is not None else None
    img_SE = normalize_image(img_SE) if img_SE is not None else None
    img_SW = normalize_image(img_SW) if img_SW is not None else None

    # Calculate differences if images are available
    diff1 = 100 + img_NE - img_SW if img_NE is not None and img_SW is not None else None
    diff2 = 100 + img_NW - img_SE if img_NW is not None and img_SE is not None else None

    # Sum of differences (if both are available)
    combined_diff = None
    if diff1 is not None and diff2 is not None:
        combined_diff = cv2.addWeighted(diff1, 0.5, diff2, 0.5, 0)
    elif diff1 is not None:  # Only diff1
        combined_diff = diff1
    elif diff2 is not None:  # Only diff2
        combined_diff = diff2

    # Check if result is available
    if combined_diff is None:
        raise ValueError("No sufficient images to process.")

    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_diff = clahe.apply((combined_diff * 255).astype(np.uint8))
    enhanced_hist = cv2.equalizeHist(enhanced_diff)

    gaussian_blur = cv2.GaussianBlur(enhanced_hist, (5, 5), 1)
    unsharp_mask = cv2.addWeighted(enhanced_hist, 1.5, gaussian_blur, -0.5, 0)

    return unsharp_mask


def calculate_shift(img1, img2):
    """
    Oblicz macierz homografii.

    Parametry:
        img1 (numpy.ndarray): Obraz odniesienia.
        img2 (numpy.ndarray): Obraz do wyrównania.

    Zwraca:
        numpy.ndarray: Macierz homografii lub None, jeśli nie udało się obliczyć.
    """
    # Inicjalizacja algorytmu SIFT
    sift = cv2.SIFT_create()

    try:
        # Detekcja punktów charakterystycznych i deskryptorów
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Sprawdzenie, czy zostały wykryte punkty charakterystyczne
        if len(kp1) < 4 or len(kp2) < 4:
            return None

        # Dopasowanie deskryptorów za pomocą BFMatcher z miarą L2
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Sprawdzenie, czy znaleziono wystarczającą liczbę dopasowań
        if len(matches) < 4:  # Muszą być co najmniej 4 dopasowane punkty
            return None

        # Wyodrębnienie współrzędnych dopasowanych punktów
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Obliczenie macierzy homografii
        matrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        return matrix

    except cv2.error as e:
        # Obsługuje błąd w przypadku nieoczekiwanych problemów z OpenCV (np. błędne dane wejściowe)
        print(f"OpenCV Error: {e}")
        return None
    except Exception as e:
        # Obsługuje inne ogólne błędy
        print(f"An error occurred: {e}")
        return None

def apply_perspective_transform(img, matrix, shape):
    """
    Przekształć obraz za pomocą macierzy homografii.

    Parametry:
        img (numpy.ndarray): Obraz do przekształcenia.
        matrix (numpy.ndarray): Macierz homografii.
        shape (tuple): Kształt obrazu odniesienia (wysokość, szerokość).

    Zwraca:
        numpy.ndarray: Przekształcony obraz.
    """
    return cv2.warpPerspective(img, matrix, (shape[1], shape[0]))

def prepare_data(folder_path, output_folder, matrix_file = "temporary_matrix.npy"):
    """
    Przetwórz folder: przekształć obrazy IR, przytnij UV/WHITE i zapisz nałożenie.

    Parametry:
        folder_path (str): Ścieżka do folderu z obrazami.
        output_folder (str): Ścieżka do folderu wynikowego.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Usuwa całą zawartość folderu
    os.makedirs(output_folder, exist_ok=True)  # Tworzy folder na nowo


    # Ścieżki do obrazów odniesienia
    img1_path = os.path.join(folder_path, "all_no_backlight_WHITE.png")
    if not os.path.exists(img1_path):
        # Jeśli plik "all_WHITE.png" nie istnieje, znajdź pierwszy plik zawierający "WHITE" w nazwie
        white_files = [f for f in os.listdir(folder_path) if "WHITE" in f and os.path.isfile(os.path.join(folder_path, f))]
        if white_files:
            img1_path = os.path.join(folder_path, white_files[0])
        else:
            raise FileNotFoundError("Brak pliku z 'WHITE' w nazwie w podanym folderze.")

    # Poszukiwanie pliku IR odpowiadającego plikowi WHITE
    img2_filename = os.path.basename(img1_path).replace("WHITE", "IR")
    img2_path = os.path.join(folder_path, img2_filename)

    try:
        if not os.path.exists(img2_path):
            raise FileNotFoundError(f"Nie znaleziono pliku IR odpowiadającego: {img1_path}")

        # Wczytaj obrazy odniesienia w formacie RGB
        img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    except FileNotFoundError as e:
        # Obsługuje błąd braku pliku IR
        print(str(e))

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Błąd pliku IR")
        msg.setText("Nie znaleziono odpowiedniego pliku IR. Kopiowanie oryginalnych obrazów do folderu wynikowego.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

        # Jeśli plik IR nie został znaleziony, kopiuj pozostałe obrazy bez przekształceń
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            if filename.lower().endswith(".png"):
                cv2.imwrite(output_path, cv2.imread(file_path))
        return  # Zakończ dalsze przetwarzanie, gdy brak pliku IR

    # Wczytaj wersje w skali szarości do obliczeń homografii
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Oblicz macierz homografii
    matrix = calculate_shift(img1_gray, img2_gray)

    if matrix is None:
        print("Nie udało się obliczyć macierzy homografii. Kopiowanie oryginalnych obrazów do folderu wynikowego.")

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)  # Ustawienie ikony ostrzeżenia
        msg.setWindowTitle("Błąd obliczenia homografii")  # Tytuł okna dialogowego
        msg.setText("Nie udało się obliczyć macierzy homografii. Kopiowanie oryginalnych obrazów do folderu wynikowego.")  # Treść komunikatu
        msg.setStandardButtons(QMessageBox.Ok)  # Przyciski do wybrania (w tym przypadku tylko "OK")
        msg.exec_()  # Wywołanie wyświetlania okna dialogowego
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            if filename.lower().endswith(".png"):
                cv2.imwrite(output_path, cv2.imread(file_path))
    else:

        # Zapisz macierz homografii
        np.save(matrix_file, matrix)
        print(f"Macierz homografii zapisana w: {matrix_file}")
        # Przetwarzanie obrazów w folderze
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Pomijaj, jeśli to nie plik graficzny
            if not filename.lower().endswith(".png"):
                continue

            img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            if "IR" in filename:
                # Przekształć obrazy IR za pomocą macierzy homografii
                transformed_img = apply_perspective_transform(img, matrix, img1.shape)
                cv2.imwrite(os.path.join(output_folder, filename), transformed_img)
            elif "UV" in filename or "WHITE" in filename:
                # Skopiuj obraz bez zmian
                cv2.imwrite(os.path.join(output_folder, filename), img)
            else:
                # Kopiuj inne obrazy bez zmian
                cv2.imwrite(os.path.join(output_folder, filename), img)

def copy_data(src_folder, dest_folder):
    """
    Kopiuje wszystkie pliki i foldery z jednego folderu do drugiego.

    Parametry:
        src_folder (str): Ścieżka do folderu źródłowego.
        dest_folder (str): Ścieżka do folderu docelowego.
    """
    if not os.path.exists(src_folder):
        raise FileNotFoundError(f"Folder źródłowy nie istnieje: {src_folder}")

    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)  # Usuwa całą zawartość folderu
    os.makedirs(dest_folder, exist_ok=True)  # Tworzy folder na nowo

    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dest_path = os.path.join(dest_folder, item)

        if os.path.isdir(src_path):
            # Tworzymy folder docelowy
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
                print(f"Utworzono folder: {dest_path}")
            # Rekurencyjnie kopiujemy zawartość podfolderu
            copy_data(src_path, dest_path)
        elif os.path.isfile(src_path):
            # Kopiujemy plik
            with open(src_path, 'rb') as src_file:
                with open(dest_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())
            print(f"Skopiowano plik: {src_path} -> {dest_path}")


# Ścieżki wejściowe i wyjściowe
# input_folder = r"C:\Users\maks\photo_edition\dokument"  # Zmień na ścieżkę do folderu wejściowego
# output_folder = r"C:\Users\maks\photo_edition\dokument2"  # Zmień na ścieżkę do folderu wyjściowego

# # Przetwórz folder
# prepare_data(input_folder, output_folder)


def neutralize_images(white_folder, calibration_folder, output_folder, new_output_folder):
    """
    Funkcja wykonująca neutralizację obrazów.
    """
    # Utwórz folder na macierze kalibracyjne, jeśli nie istnieje
    if not os.path.exists(calibration_folder):
        os.makedirs(calibration_folder)


    # Wykonaj neutralizację obrazów
    process_white_sheets(white_folder, calibration_folder)
    apply_calibration(output_folder, calibration_folder, new_output_folder)

    return new_output_folder

def process_white_sheets(input_folder, output_folder, matrix_file = "temporary_matrix.npy"):
    """
    Tworzy macierze kalibracyjne dla kanału jasności (V) w przestrzeni HSV
    dla wszystkich zdjęć białych kartek w folderze.
    """

    # Wczytaj macierz homografii
    if not os.path.exists(matrix_file):
        print(f"Brak pliku macierzy homografii: {matrix_file}")
        matrix = None
    else:
        matrix = np.load(matrix_file)
        print(f"Wczytano macierz homografii z: {matrix_file}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        
        # Sprawdzenie czy to obraz
        if not (file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg')):
            continue
        
        # Wczytanie obrazu białej kartki
        image = cv2.imread(input_path)
        if image is None:
            print(f"Nie udało się wczytać obrazu: {file_name}")
            continue
        
        if matrix is not None and ("IR" in file_name): # obrazy IR były przekształcane
            transformed_image = apply_perspective_transform(image, matrix, image.shape)
        else:
            transformed_image = image
        # Konwersja na HSV
        hsv_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        v_channel = hsv_image[:, :, 2]  # Kanał jasności (V)
        
        # Obliczenie macierzy kalibracyjnej
        brightness = 200
        calibration_matrix = brightness / (v_channel + 1e-5)  # Unikamy dzielenia przez zero

        calibration_matrix = cv2.GaussianBlur(calibration_matrix, (5, 5), 1)
        
        # Zapis macierzy kalibracyjnej
        output_file = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_V.npy")
        np.save(output_file, calibration_matrix)
        print(f"Zapisano macierz kalibracyjną dla kanału jasności V: {output_file}")


def apply_calibration(input_folder, calibration_folder, output_folder):
    """
    Kalibruje wszystkie zdjęcia w katalogu `input_folder` w przestrzeni HSV
    na podstawie macierzy kalibracyjnych w katalogu `calibration_folder`.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Lista plików bez macierzy kalibracyjnej
    missing_calibration_files = []

    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        
        # Sprawdzenie, czy to obraz
        if not (file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg')):
            continue
        
        # Wczytanie obrazu do poprawy
        image = cv2.imread(input_path)
        if image is None:
            print(f"Nie udało się wczytać obrazu: {file_name}")
            continue
        
        # Konwersja na HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        v_channel = hsv_image[:, :, 2]  # Kanał jasności (V)
        
        # Wczytanie macierzy kalibracyjnej
        calibration_file = os.path.join(calibration_folder, f"{os.path.splitext(file_name)[0]}_V.npy")
        if not os.path.exists(calibration_file):
            print(f"Brak pliku kalibracyjnego dla: {file_name}")
            missing_calibration_files.append(file_name)  # Dodajemy brakujący plik do listy
            continue
        
        calibration_matrix = np.load(calibration_file)
        
        # Kalibracja jasności
        calibrated_v_channel = np.clip(v_channel * calibration_matrix, 0, 255)
        hsv_image[:, :, 2] = calibrated_v_channel
        
        # Konwersja z powrotem do BGR
        calibrated_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Zapisanie skalibrowanego obrazu
        output_file = os.path.join(output_folder, file_name)
        cv2.imwrite(output_file, calibrated_image)
        print(f"Zapisano skalibrowane zdjęcie: {output_file}")
    
    # Jeśli istnieją brakujące pliki kalibracyjne, wyświetl dialog
    if missing_calibration_files:
        missing_files_str = ""
        for missing in missing_calibration_files:
            missing_files_str += f"\n- {missing}"  # Łączenie nazw plików w jeden ciąg
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)  # Ustawienie ikony ostrzeżenia
        msg.setWindowTitle("Brakujące pliki kalibracyjne")  # Tytuł okna dialogowego
        msg.setText(f"Nie znaleziono pliku kalibracyjnego dla:\n{missing_files_str} \nPliki te zostaną niezmienione")  # Treść komunikatu
        msg.setStandardButtons(QMessageBox.Ok)  # Przyciski do wybrania (w tym przypadku tylko "OK")
        msg.exec_()  # Wywołanie wyświetlania okna dialogowego


def bayes_classification(image_paths, *args):
    """
    Funkcja wykonuje klasyfikację Bayesa z interaktywnym zaznaczaniem obszarów w GUI Matplotlib.
    """
    global classes, current_class, finished_selecting, image_display
    classes = {}
    current_class = 1
    finished_selecting = False

    def load_images_to_vector(image_paths):
        img_white, img_other1, img_other2 = None, None, None
        for path in image_paths:
            if "WHITE" in path.upper():
                img_white = plt.imread(path)
            elif img_other1 is None:
                img_other1 = rgb2gray(plt.imread(path))
            else:
                img_other2 = rgb2gray(plt.imread(path))

        if img_white is None or img_other1 is None or img_other2 is None:
            raise ValueError("Niewystarczające obrazy do klasyfikacji.")


        h, w, _ = img_white.shape
        combined_features = np.zeros((h, w, 5))
        combined_features[:, :, :3] = img_white
        combined_features[:, :, 3] = img_other1
        combined_features[:, :, 4] = img_other2

        return combined_features.reshape(-1, 5), h, w

    def onselect(eclick, erelease):
        global classes, current_class
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        mask = np.zeros(image_display.shape[:2], dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        classes[current_class] = classes.get(current_class, mask) | mask
        plt.title(f"Zaznaczono klasę {current_class}")
        fig.canvas.draw_idle()

    def stop_selecting(event):
        global finished_selecting
        finished_selecting = True
        plt.close(fig)

    def next_class(event):
        global current_class
        current_class += 1
        plt.title(f"Zaznaczanie klasy {current_class}")
        fig.canvas.draw_idle()

    def run_classification(combined_features, h, w):
        global classes
        labels = np.zeros(combined_features.shape[0], dtype=int)
        for class_id, mask in classes.items():
            labels[mask.flatten()] = class_id

        X_train = combined_features[labels > 0]
        y_train = labels[labels > 0]

        unique_classes = np.unique(y_train)
        means, covariances, priors = {}, {}, {}

        for cls in unique_classes:
            X_cls = X_train[y_train == cls]
            means[cls] = np.mean(X_cls, axis=0)
            covariances[cls] = np.cov(X_cls, rowvar=False)
            priors[cls] = X_cls.shape[0] / X_train.shape[0]

        def gaussian_density(x, mean, covariance):
            dim = x.shape[-1]
            covariance_inv = np.linalg.inv(covariance)
            det_covariance = np.linalg.det(covariance)
            factor = 1.0 / np.sqrt((2 * np.pi) ** dim * det_covariance)
            diff = x - mean
            exponent = -0.5 * np.einsum('ij,jk,ik->i', diff, covariance_inv, diff)
            return factor * np.exp(exponent)

        likelihoods = np.zeros((combined_features.shape[0], len(unique_classes)))
        for i, cls in enumerate(unique_classes):
            likelihoods[:, i] = gaussian_density(combined_features, means[cls], covariances[cls]) * priors[cls]

        y_pred = np.argmax(likelihoods, axis=1) + 1
        classified_image = y_pred.reshape(h, w)
        plt.figure()
        plt.imshow(classified_image, cmap='tab10')
        plt.title("Wynik klasyfikacji")
        plt.colorbar(label="Klasy")
        plt.show()

    combined_features, h, w = load_images_to_vector(image_paths)
    fig, ax = plt.subplots()
    image_display = plt.imread([p for p in image_paths if "WHITE" in p.upper()][0])
    ax.imshow(image_display)
    plt.title(f"Zaznacz prostokąty dla klasy {current_class}")

    selector = RectangleSelector(ax, onselect, useblit=False, button=[1])
    done_ax = plt.axes([0.8, 0.01, 0.15, 0.05])
    Button(done_ax, "Zakończ zaznaczanie").on_clicked(stop_selecting)

    next_ax = plt.axes([0.6, 0.01, 0.15, 0.05])
    Button(next_ax, "Następna klasa").on_clicked(next_class)

    plt.show(block=False)

    if finished_selecting:
        run_classification(combined_features, h, w)


def text_detection(image_paths, *args):
    reader = easyocr.Reader(['en', 'pl'], gpu=True)  

    # Wczytujemy obraz
    sciezka_do_obrazu = image_paths[0]
    wyniki = reader.readtext(sciezka_do_obrazu, contrast_ths=0.2, adjust_contrast=True, text_threshold=0.5)

    # Wizualizacja wyników na obrazie
    #póki co skala szarości
    obraz = cv2.imread(sciezka_do_obrazu, cv2.IMREAD_GRAYSCALE)
    for (bbox, tekst, prawdopodobienstwo) in wyniki:
        (p1, p2), (p3, p4) = bbox[0], bbox[2]
        cv2.rectangle(obraz, (int(p1), int(p2)), (int(p3), int(p4)), (0, 255, 0), 2)
        cv2.putText(obraz, tekst, (int(p1), int(p2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Wyświetlamy tekst
    print("Rozpoznany tekst:")
    for bbox, tekst, prawdopodobienstwo in wyniki:
        print(f"Tekst: {tekst}, Prawdopodobieństwo: {prawdopodobienstwo}")

    # Normalizacja obrazu
    result_image = cv2.normalize(obraz, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return result_image, wyniki





class BayesClassificationDialog(QDialog):
    def __init__(self, images, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bayes Classification")
        self.resize(1000, 1300)  # Ustawienie rozmiaru okna na 1000x1300 pikseli
        self.images = [cv2.imread(path) for path in images]

        # Znajdź obraz z "WHITE" w nazwie (nie zakładaj, że będzie pierwszy)
        white_image_path = None
        for path in images:
            if "WHITE" in path.upper():  # Szukamy "WHITE" w nazwie
                white_image_path = path
                break

        if white_image_path is None:
            raise ValueError("Nie znaleziono obrazu z 'WHITE' w nazwie!")

        # Załaduj obraz RGB (zakładamy, że ma 3 kanały)
        self.white_image = cv2.imread(white_image_path)

        # Określ pozostałe obrazy (np. inne obrazy pomocnicze)
        self.other_images = [cv2.imread(path) for path in images if path != white_image_path]

        self.current_class = 1
        self.classes = {}
        self.selection_start = QPoint()
        self.selection_end = QPoint()
        self.drawing = False
        self.result_image = None

        # Wyświetl obraz RGB z "WHITE" (zakładamy, że ma 3 kanały)
        self.image_display = self.white_image.copy()
        self.image_pixmap, self.scale_factor = self.convert_to_pixmap(self.image_display, max_width=900, max_height=1200)
        self.resize(int(self.image_display.shape[1] * self.scale_factor),
                    int(self.image_display.shape[0] * self.scale_factor))

        # Layout
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel(self)
        self.image_label.setPixmap(self.image_pixmap)
        self.layout.addWidget(self.image_label)

        # Control buttons
        button_layout = QHBoxLayout()
        self.add_class_button = QPushButton("Next Class")
        self.add_class_button.clicked.connect(self.next_class)
        button_layout.addWidget(self.add_class_button)

        self.finish_button = QPushButton("Finish")
        self.finish_button.clicked.connect(self.finish_selection)
        button_layout.addWidget(self.finish_button)

        self.layout.addLayout(button_layout)

        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release

    def convert_to_pixmap(self, image, max_width=900, max_height=1200):
        """Convert a NumPy image to a QPixmap with scaling. Returns pixmap and scaling factor."""
        height, width, channels = image.shape

        # Skalowanie obrazu, jeśli przekracza maksymalne wymiary
        scaling_factor = min(max_width / width, max_height / height, 1.0)  # Nie powiększaj, jeśli mniejszy
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Konwersja na QPixmap
        bytes_per_line = channels * resized_image.shape[1]
        qimage = QImage(resized_image.data, resized_image.shape[1], resized_image.shape[0], bytes_per_line, QImage.Format_BGR888)
        return QPixmap.fromImage(qimage), scaling_factor

    def mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.selection_start = event.pos()
            self.drawing = True

    def mouse_move(self, event):
        if self.drawing:
            self.selection_end = event.pos()
            self.update()

    def mouse_release(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.selection_end = event.pos()
            self.add_selection()
            self.drawing = False

    def add_selection(self):
        # Przeskalowanie współrzędnych zaznaczenia do oryginalnych wymiarów obrazu
        x1, y1 = int(self.selection_start.x() / self.scale_factor), int(self.selection_start.y() / self.scale_factor)
        x2, y2 = int(self.selection_end.x() / self.scale_factor), int(self.selection_end.y() / self.scale_factor)
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        class_colors = {
            1: QColor(0, 0, 255),       # Czerwony
            2: QColor(0, 255, 0),       # Zielony
            3: QColor(255, 0, 0),       # Niebieski
            4: QColor(255, 255, 0),     # Żółty
            5: QColor(0, 255, 255),     # Cyjan
            6: QColor(255, 0, 255),     # Magenta
            7: QColor(192, 192, 192),   # Szary
            8: QColor(128, 128, 128),   # Ciemny szary
            9: QColor(255, 165, 0),     # Pomarańczowy
            10: QColor(255, 105, 180),  # Różowy
        }

        def set_pen_color(current_class):
            color = class_colors.get(current_class, QColor(0, 0, 0))  # Domyślnie czarny, jeśli klasa nie istnieje
            return QPen(color)

        # Tworzenie maski
        rect = QRect(self.selection_start, self.selection_end)
        painter = QPainter(self.image_pixmap)
        pen = set_pen_color(self.current_class)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(rect)
        painter.end()

        mask = np.zeros(self.image_display.shape[:2], dtype=bool)
        mask[y_min:y_max, x_min:x_max] = True
        self.classes[self.current_class] = self.classes.get(self.current_class, mask) | mask
        self.image_label.setPixmap(self.image_pixmap)

    def next_class(self):
        self.current_class += 1

    def finish_selection(self):
        if len(self.classes) > 1:
            self.accept()  # Zamknij okno z wynikiem "zaakceptowanym"
            self.run_classification()

    def run_classification(self):
        """Run Bayes classification using QDA and return the result image with masks applied."""
        # Prepare features and masks
        combined_features, h, w = self.prepare_data()
        labels = np.zeros(combined_features.shape[0], dtype=int)
        for class_id, mask in self.classes.items():
            labels[mask.flatten()] = class_id

        # Train Bayes classifier using QDA
        X_train = combined_features[labels > 0]
        y_train = labels[labels > 0]
        classifier = QuadraticDiscriminantAnalysis()
        classifier.fit(X_train, y_train)

        # Predict labels for all pixels
        predictions = classifier.predict(combined_features)
        result_labels = predictions.reshape(h, w)

        # Apply masks to the original image
        self.result_image = self.apply_masks(self.white_image, result_labels)
        return self.result_image

    def prepare_data(self):
        """Prepare image features for classification."""
        # 'self.white_image' to obraz RGB, więc można użyć jego kanałów bez zmiany
        img_white = self.white_image

        # Używamy innych obrazów (np. szaro-skalowych) jako cech pomocniczych
        img_other1, img_other2 = self.other_images[0], self.other_images[1]
        img_other1 = cv2.cvtColor(img_other1, cv2.COLOR_BGR2GRAY)
        img_other2 = cv2.cvtColor(img_other2, cv2.COLOR_BGR2GRAY)

        h, w, _ = img_white.shape
        combined_features = np.zeros((h, w, 5))
        combined_features[:, :, :3] = img_white / 255.0  # Kanały RGB
        combined_features[:, :, 3] = img_other1 / 255.0
        combined_features[:, :, 4] = img_other2 / 255.0

        return combined_features.reshape(-1, 5), h, w

    def apply_masks(self, image, labels):
        """Apply classification masks to the image."""
        colors = {
            1: (0, 0, 255),  # Klasa 1: Czerwony
            2: (0, 255, 0),  # Klasa 2: Zielony
            3: (255, 0, 0),  # Klasa 3: Niebieski
            4: (255, 255, 0),  # Klasa 4: Żółty
            5: (0, 255, 255),  # Klasa 5: Cyjan
            6: (255, 0, 255),  # Klasa 6: Magenta
            7: (192, 192, 192),  # Klasa 7: Szary
            8: (128, 128, 128),  # Klasa 8: Ciemny szary
            9: (255, 165, 0),  # Klasa 9: Pomarańczowy
            10: (255, 105, 180),  # Klasa 10: Różowy
        }
        output_image = image.copy()
        for class_id, color in colors.items():
            mask = (labels == class_id)
            output_image[mask] = color
        return output_image
    
    def get_result_image(self):
        """Zwraca wynikowy obraz."""
        if self.result_image is not None:
            return self.result_image
        else:
            raise ValueError("Klasyfikacja nie została przeprowadzona.")

    

def run_bayes(image_paths):
    try:
        dialog = BayesClassificationDialog(image_paths)
        result = dialog.exec_()  # Uruchom dialog jako modalne okno
        if result == QDialog.Accepted:
            print("Bayes zakończony poprawnie.")
            result_image = dialog.get_result_image()  # Pobierz wynikowy obraz
            print(result_image)
            return result_image
        else:
            print("Bayes anulowany.")
            return None
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error during Bayes: {e}")
        print("Traceback:")
        print(tb)
        return None

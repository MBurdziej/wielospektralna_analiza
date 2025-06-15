import RPi.GPIO as GPIO
import time
import os
from datetime import datetime
from picamera2 import Picamera2
from PIL import Image
import numpy as np

# Ustawienia pinów GPIO
DATA_PIN = 22   # Pin 2 - Data In rejestru 4094
CLOCK_PIN = 27  # Pin 3 - Clock rejestru
STROBE_PIN = 17 # Pin 1 - Strobe/Latch Enable rejestru

# Liczba diod
NUM_LEDS = 24  # 3 rejestry * 8 diod na rejestr

# Diody do pominięcia
SKIP_LEDS = {1, 2, 9, 10, 17, 18, 16, 24, 5}

# Diody do zapalania jednocześnie
GROUPS = [
    {12, 16},   # Diody 12 + 16
    {21, 24},   # Diody 21 + 24
    {3, 5}      # Diody 3 + 5
]

# Typy diod
IR_LEDS = range(1, 9)
UV_LEDS = range(9, 17)
WHITE_LEDS = range(17, 25)

# Mapowanie diod na kierunki
DIRECTION_MAP = {
    "NE": {11, 19, 7},
    "NW": {13, 20, 8},
    "SE": {15, 22, 4},
    "SW": {14, 23, 6},
    "backlight": {12, 16, 21, 24, 3, 5}
}

# Konfiguracja GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(DATA_PIN, GPIO.OUT)
GPIO.setup(CLOCK_PIN, GPIO.OUT)
GPIO.setup(STROBE_PIN, GPIO.OUT)

# Konfiguracja kamery
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (4056, 3040), "format": "RGB888"})  # Maksymalna rozdzielczość
picam2.configure(camera_config)

# Definicja ustawień kamery dla każdej sytuacji
camera_settings = {
    "IR":           {"ExposureTime": 200000, "AnalogueGain": 1.0, "ColourGains": (1.0, 1.0)},
    "UV":           {"ExposureTime": 25000, "AnalogueGain": 1.0, "ColourGains": (1.0, 1.0)},
    "WHITE":        {"ExposureTime": 50000, "AnalogueGain": 1.0, "ColourGains": (1.5, 1.8)}, #50000
    "IR_BL":        {"ExposureTime": 150000, "AnalogueGain": 1.0, "ColourGains": (1.0, 1.0)},
    "UV_BL":        {"ExposureTime": 25000, "AnalogueGain": 1.0, "ColourGains": (1.0, 1.0)},
    "WHITE_BL":     {"ExposureTime": 30000, "AnalogueGain": 1.0, "ColourGains": (1.5, 1.8)}, #150000
    "ALL_IR":       {"ExposureTime": 80000, "AnalogueGain": 1.0, "ColourGains": (1.0, 1.0)},
    "ALL_UV":       {"ExposureTime": 10000, "AnalogueGain": 1.0, "ColourGains": (1.0, 1.0)},
    "ALL_WHITE":    {"ExposureTime": 10000, "AnalogueGain": 1.0, "ColourGains": (1.5, 1.8)}, #20000
    "ALL_NB_IR":    {"ExposureTime": 100000, "AnalogueGain": 1.0, "ColourGains": (1.0, 1.0)},
    "ALL_NB_UV":    {"ExposureTime": 10000, "AnalogueGain": 1.0, "ColourGains": (1.0, 1.0)},
    "ALL_NB_WHITE": {"ExposureTime": 15000, "AnalogueGain": 1.0, "ColourGains": (1.5, 1.8)} #22000
}

# Funkcja stosująca ustawienia kamery
def apply_camera_settings(mode, direction):
    if direction == "backlight":
        mode = mode + "_BL"
    elif direction == "all":
        mode = "ALL_" + mode
    elif direction == "all_no_backlight":
        mode = "ALL_NB_" + mode
    settings = camera_settings[mode]

    picam2.set_controls({
        "AeEnable": False,
        "AwbEnable": False,
        "ExposureTime": settings["ExposureTime"],
        "AnalogueGain": settings["AnalogueGain"],
        "ColourGains": settings["ColourGains"]
    })

    while(not check_camera_settings(settings)):

        picam2.set_controls({
            "ExposureTime": 100000,  # Neutralny czas ekspozycji
            "ColourGains": (1.0, 1.0),
            "AnalogueGain": 1.0
        })

        picam2.stop()
        time.sleep(0.2)  # Krótkie opóźnienie
        camera_config = picam2.create_still_configuration(main={"size": (4056, 3040), "format": "RGB888"})
        picam2.configure(camera_config)
        picam2.start()

        # picam2.set_controls({
        #     "ExposureTime": 100000,  # Neutralny czas ekspozycji
        #     "ColourGains": (1.0, 1.0),
        #     "AnalogueGain": 1.0
        # })

        picam2.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": settings["ExposureTime"],
            "AnalogueGain": settings["AnalogueGain"],
            "ColourGains": settings["ColourGains"],
            "NoiseReductionMode": 2,
            "Sharpness": 1.5
        })
            
    print(f"Zastosowano ustawienia kamery dla trybu {mode}: {settings}")

picam2.start()

def check_camera_settings(expected_settings, timeout=5):
    def is_close(expected, actual, tolerance=0.01):
        """
        Porównuje dwie wartości liczbowe z tolerancją.
        """
        return abs(expected - actual) <= tolerance * expected

    start_time = time.time()
    while time.time() - start_time < timeout:
        current_settings = picam2.capture_metadata()
        
        # print("Oczekiwane:", expected_settings, "\nBieżące:", current_settings, "\n")

        # Sprawdź każde ustawienie z tolerancją 1%
        if (
            is_close(expected_settings["ExposureTime"], current_settings.get("ExposureTime", 0)) and
            is_close(expected_settings["AnalogueGain"], current_settings.get("AnalogueGain", 0)) and
            all(
                is_close(expected, actual)
                for expected, actual in zip(expected_settings["ColourGains"], current_settings.get("ColourGains", (0, 0)))
            )
        ):
            print("Ustawienia kamery zostały poprawnie zaaplikowane.")
            return True
        time.sleep(0.1)  # Poczekaj 100 ms przed ponownym sprawdzeniem
    
    print("Nie udało się ustawić wszystkich parametrów kamery w czasie.")
    return False



# Utwórz folder na zdjęcia
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = f"./photos_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

camera_data_file = os.path.join(output_folder, "camera_data.txt")

def log_camera_settings(filename, photo_name):
    settings = picam2.capture_metadata()
    with open(filename, "a") as f:
        f.write(f"Photo: {photo_name}\n")
        f.write(f"  ExposureTime: {settings.get('ExposureTime', 'N/A')} µs\n")
        f.write(f"  AnalogueGain: {settings.get('AnalogueGain', 'N/A')}\n")
        f.write(f"  ColourGains: {settings.get('ColourGains', 'N/A')}\n")
        f.write(f"  NoiseReductionMode: {settings.get('NoiseReductionMode', 'N/A')}\n")
        f.write(f"  Sharpness: {settings.get('Sharpness', 'N/A')}\n")
        f.write("\n")
    print(f"Zapisano ustawienia kamery dla zdjęcia: {photo_name}")

# Funkcja przesuwania danych do rejestru 4094
def shift_out(data):
    for i in range(24):
        GPIO.output(DATA_PIN, (data >> (23 - i)) & 1)
        GPIO.output(CLOCK_PIN, GPIO.HIGH)
        time.sleep(0.001)
        GPIO.output(CLOCK_PIN, GPIO.LOW)
    GPIO.output(STROBE_PIN, GPIO.HIGH)
    time.sleep(0.001)
    GPIO.output(STROBE_PIN, GPIO.LOW)

# Funkcja ustawiająca wybrane diody (jako zestaw bitów)
def set_leds(led_numbers):
    global leds_state
    leds_state = 0  # Resetujemy stan wszystkich diod
    
    for led_number in led_numbers:
        if 1 <= led_number <= NUM_LEDS:
            leds_state |= (1 << (NUM_LEDS - led_number))  # Ustaw odpowiedni bit (MSB first)
    shift_out(leds_state)

# Funkcja przetwarzania obrazu
def process_image(filename, mode):
    image = Image.open(filename)
    data = np.array(image)

    if mode == "IR":
        # Zostawiamy tylko składową R
        data[:, :, 1] = 0  # G = 0
        data[:, :, 2] = 0  # B = 0
    elif mode == "UV":
        # Zostawiamy tylko składową B
        data[:, :, 0] = 0  # R = 0
        data[:, :, 1] = 0  # G = 0

    processed_image = Image.fromarray(data)
    processed_image.save(filename)
    print(f"Przetworzono obraz: {filename}")

# Funkcja robienia zdjęcia z różnymi ustawieniami
def take_photo(led_numbers):
    # Wybór trybu na podstawie typu diod
    mode = None
    if any(led in IR_LEDS for led in led_numbers):
        mode = "IR"
    elif any(led in UV_LEDS for led in led_numbers):
        mode = "UV"
    elif any(led in WHITE_LEDS for led in led_numbers):
        mode = "WHITE"

    

    # Określenie kierunku
    direction = None
    for key, leds in DIRECTION_MAP.items():
        if leds.intersection(led_numbers):
            direction = key
            break
    if len(led_numbers) < 3:
        direction = direction or "unknown"
    elif {12, 16, 21, 24, 3, 5} & led_numbers:
        direction = "all"
    else:
        direction = "all_no_backlight"

    apply_camera_settings(mode, direction)

    time.sleep(0.1)  # Stabilizacja przed zdjęciem
    filename = f"{output_folder}/{direction}_{mode}.png"
    picam2.capture_file(filename)
    log_camera_settings(camera_data_file, filename)

    # # Przetwarzanie obrazu dla IR i UV
    # if mode in ["IR", "UV"]:
    #     process_image(filename, mode)

    print(f"Zapisano zdjęcie: {filename}")

# Funkcja obsługująca wszystkie diody danego koloru
def take_led_photos(led_type_label):
    # Mapa typów LED do ich odpowiednich zmiennych
    led_map = {
        "IR": IR_LEDS,
        "UV": UV_LEDS,
        "WHITE": WHITE_LEDS
    }

    if led_type_label not in led_map:
        raise ValueError("Invalid LED type. Please use 'UV', 'IR', or 'WHITE'.")

    # Pobranie zestawu LED na podstawie etykiety
    led_type = led_map[led_type_label]

    # Wszystkie diody danego koloru
    all_leds = set(led_type)
    set_leds(all_leds)
    take_photo(all_leds)
    time.sleep(0.5)

    # Wszystkie diody oprócz backlight
    non_backlight_leds = all_leds - DIRECTION_MAP["backlight"]
    set_leds(non_backlight_leds)
    take_photo(non_backlight_leds)
    time.sleep(0.5)


# Główna pętla
try:
    for led in range(1, 9):
        if led in SKIP_LEDS:
            continue
        
        group_to_light = None
        for group in GROUPS:
            if led in group:
                group_to_light = group
                break

        if group_to_light:
            set_leds(group_to_light)
            take_photo(group_to_light)
            time.sleep(0.5)
        else:
            set_leds({led})
            take_photo({led})
            time.sleep(0.5)

    take_led_photos("IR")

    set_leds(WHITE_LEDS)

    print("Zdjęcia dla światła IR zostały wykonane.")
    input("Wciśnij Enter, aby kontynuować i wykonać zdjęcia dla pozostałych trybów...")

    for led in range(9, 16):
        if led in SKIP_LEDS:
            continue
        
        group_to_light = None
        for group in GROUPS:
            if led in group:
                group_to_light = group
                break

        if group_to_light:
            set_leds(group_to_light)
            take_photo(group_to_light)
            time.sleep(0.5)
        else:
            set_leds({led})
            take_photo({led})
            time.sleep(0.5)

    take_led_photos("UV")

    # print("Zdjęcia dla światła UV zostały wykonane.")
    # input("Wciśnij Enter, aby kontynuować i wykonać zdjęcia dla pozostałych trybów...")

    for led in range(16, NUM_LEDS + 1):
        if led in SKIP_LEDS:
            continue
        
        group_to_light = None
        for group in GROUPS:
            if led in group:
                group_to_light = group
                break

        if group_to_light:
            set_leds(group_to_light)
            take_photo(group_to_light)
            time.sleep(0.5)
        else:
            set_leds({led})
            take_photo({led})
            time.sleep(0.5)

    take_led_photos("WHITE")


except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
    picam2.stop()

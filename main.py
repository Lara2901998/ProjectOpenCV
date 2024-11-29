import cv2
import numpy as np
import glob


# 1. KALIBRACIJA KAMERE

chessboard_size = (9, 6)

# Kreiranje 3D tačaka u stvarnom svetu (šahovska tabla)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Liste za 3D tačke u stvarnom svetu i 2D tačke sa slika
objpoints = []  
imgpoints = []

# Učitavanje slika za kalibraciju sa šahovskom tablom
images = glob.glob('camera_cal/calibration*.jpg')
if not images:
    print("Nema slika u direktorijumu 'camera_cal'. Proveri putanju!")
    exit()

# Prolaz kroz sve slike sa šahovskom tablom
for fname in images:
    img = cv2.imread(fname)
    
    # Konvertovanje slike u sivu boju (potrebno za detekciju uglova)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pronalaženje uglova šahovske table na slici
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        # Ako su uglovi pronađeni, dodajemo ih u liste
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Iscrtavanje uglova na slici
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        
        # Prikazivanje slike sa označenim uglovima
        cv2.imshow('Chessboard corners', img)
        cv2.waitKey(500)

# Zatvaranje svih prikazanih prozora
cv2.destroyAllWindows()

# Ako nisu pronađene tačke, izlazi se iz programa
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("Detekcija šahovske table nije uspela.")
    exit()

# Kalibracija kamere: izračunavanje matrice kamere i koeficijenata distorzije
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
if not ret:
    print("Kalibracija nije uspela.")
    exit()

# 2. KOREKCIJA DISTORZIJE
# Učitavanje slike koju želimo da ispravimo
test_img = cv2.imread('test1.jpg')
# Ispravka distorzije slike korišćenjem matrice kamere i koeficijenata distorzije
undistorted = cv2.undistort(test_img, mtx, dist, None, mtx)


# Prikaz originalne i korigovane slike
cv2.imshow('Original Image', test_img) 
cv2.imshow('Undistorted Image', undistorted) 


cv2.waitKey(0)
cv2.destroyAllWindows()
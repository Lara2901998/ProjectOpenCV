
import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
from sklearn import calibration

# Učitavanje slika
chessboard_image = cv2.imread('camera_cal/calibration1.jpg')
#test_img = cv2.imread('test_images/test1.jpg')
#test_img = cv2.imread('test_images/test2.jpg')
#test_img = cv2.imread('test_images/test3.jpg')
#test_img = cv2.imread('test_images/test4.jpg')
#test_img = cv2.imread('test_images/test5.jpg')
#test_img = cv2.imread('test_images/test6.jpg')
#test_img = cv2.imread('test_images/challange00101.jpg')
#test_img = cv2.imread('test_images/challange00111.jpg')
#test_img = cv2.imread('test_images/challange00136.jpg')
#test_img = cv2.imread('test_images/solidWhiteCurve.jpg')
#test_img = cv2.imread('test_images/solidWhiteRight.jpg')
#test_img = cv2.imread('test_images/solidYellowCurve.jpg')
#test_img = cv2.imread('test_images/solidYellowCurve2.jpg')
#test_img = cv2.imread('test_images/solidYellowLeft.jpg')
#test_img = cv2.imread('test_images/straight_lines1.jpg')
test_img = cv2.imread('test_images/straight_lines2.jpg')


# 1. ********************KALIBRACIJA KAMERE*******************

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
        #cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        
        # Prikazivanje slike sa označenim uglovima
        #cv2.imshow('Chessboard corners', img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()

# Ako nisu pronađene tačke, izlazi se iz programa
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("Detekcija šahovske table nije uspela.")
    exit()

# Kalibracija kamere: izračunavanje matrice kamere i koeficijenata distorzije
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
if not ret:
    print("Kalibracija nije uspela.")
    exit()


# 2. *******************KOREKCIJA DISTORZIJE*******************

# Ispravka distorzije slike korišćenjem matrice kamere i koeficijenata distorzije
undistorted_chessboard = cv2.undistort(chessboard_image, mtx, dist, None, mtx)
undistorted = cv2.undistort(test_img, mtx, dist, None, mtx)


# 3. *******************KREIRANJE BINARNE SLIKE*******************

# Konvertovanje slike u HSV boje, što omogućava lakšu selekciju boja na slici
hsv = cv2.cvtColor(undistorted, cv2.COLOR_BGR2HSV)

# Kreiranje maske za žute i bele trake na osnovu opsega boja u HSV prostoru
yellow_mask = cv2.inRange(hsv, (15, 120, 200), (50, 255, 255))
white_mask = cv2.inRange(hsv, (0, 0, 200), (255, 25, 255))

# Kombinovanje žute i bele maske u jednu masku koja obuhvata obe trake
mask = cv2.bitwise_or(yellow_mask, white_mask)

# Primena Canny edge detekcije na maskiranoj slici kako bi se dobile ivice
binary_img = cv2.Canny(cv2.GaussianBlur(mask, (5, 5), 0), 50, 150)


# 4. *******************PRIMENA PERSPEKTIVNE TRANSFORMACIJE*******************

# Dobijamo visinu i širinu slike bez distorzije
height, width = undistorted.shape[:2]

# Kreiramo praznu masku iste veličine kao originalna maska (sa vrednostima 0)
roi_mask = np.zeros_like(mask)

# Postavljamo masku u donju polovinu slike (gore je 0, a dole je 255)
roi_mask[int(height / 2):, :] = 255

# Koristimo masku da izdvojimo samo ivice koje pripadaju donjoj polovini slike
roi_edges = cv2.bitwise_and(binary_img, roi_mask)

# Definišemo četiri tačke u originalnoj slici koje će se koristiti za perspektivnu transformaciju
pts1 = np.float32([  
    [100, height],  # Donji levi ugao
    [width - 100, height],  # Donji desni ugao
    [width - 200, height - 200],  # Gornji desni ugao
    [250, height - 200]  # Gornji levi ugao
])

# Definišemo četiri tačke u ciljanom prostoru koje će predstavljati ispravljenu sliku
pts2 = np.float32([
    [100, height],  # Donji levi ugao
    [width - 100, height],  # Donji desni ugao
    [width - 100, height - 650],  # Gornji desni ugao
    [100, height - 650]  # Gornji levi ugao
])

# Kreiramo matricu za perspektivnu transformaciju između definisanih tačaka
matrix = cv2.getPerspectiveTransform(pts1, pts2)

# Primena perspektivne transformacije na sliku ivica (edges) koristeći definisanu matricu
warped_image = cv2.warpPerspective(roi_edges, matrix, (width, height))



# 5. *******************DETEKCIJA PIKSELA ZA TRAKE*******************

# Pronalazimo sve ne-nulte piksele na binarnoj slici (warped_image)
nonzero = warped_image.nonzero()
y_coords = np.array(nonzero[0])
x_coords = np.array(nonzero[1])

# Uslovi za razdvajanje piksela na levu i desnu traku
left_condition = (x_coords > width // 4) & (x_coords < width // 2)
right_condition = (x_coords > width // 2) & (x_coords < 3 * width // 4)

# Linearno fitovanje piksela za levu i desnu traku
fit_left = np.polyfit(y_coords[left_condition], x_coords[left_condition], 1)
fit_right = np.polyfit(y_coords[right_condition], x_coords[right_condition], 1)

# Generišemo tačke za linije na osnovu fitovanja
plot_y = np.linspace(0, height - 1, height)
left_line = fit_left[0] * plot_y + fit_left[1]
right_line = fit_right[0] * plot_y + fit_right[1]

# Računanje histograma za celu warped sliku
histogram = np.sum(warped_image, axis=0)  # Sumira sve piksela po kolonama u celoj slici


# 6. *******************ODREĐIVANJE ZAKRIVLJENOSTI I POZICIJE VOZILA*******************

# Izračunavanje krivine za levu i desnu traku
left_curvature = ((1 + (2 * fit_left[0] * height + fit_left[1])**2)**(3/2)) / np.abs(2 * fit_left[0])
right_curvature = ((1 + (2 * fit_right[0] * height + fit_right[1])**2)**(3/2)) / np.abs(2 * fit_right[0])

# Izračunavanje pozicije vozila u odnosu na centar
lane_center = (left_line[height - 1] + right_line[height - 1]) / 2  # Centar trake
vehicle_offset = (lane_center - width / 2) * 3.7 / width  # Pretvaranje u metre (širina trake = 3.7 m)


# 7. *******************PROJEKCIJA DETEKTOVANIH GRANICA TRAKE*******************

# Kreiramo poligon između linija
left_points = np.array([np.transpose(np.vstack([left_line, plot_y]))])
right_points = np.array([np.flipud(np.transpose(np.vstack([right_line, plot_y])))] )
points = np.hstack((left_points, right_points))

# Transparentni sloj sa zelenim poligonom
lane_overlay = np.zeros_like(undistorted, dtype=np.uint8)

# Sada punimo poligon sa zelenom bojom
cv2.fillPoly(lane_overlay, np.int_([points]), (0, 255, 0))

# Crtamo linije na sloju (crvene i plave linije) nakon popunjavanja poligona
for y in range(height):
    if 0 <= left_line[y] < width:
        cv2.circle(lane_overlay, (int(left_line[y]), int(plot_y[y])), 5, (255, 0, 0), -1)  # Plava
    if 0 <= right_line[y] < width:
        cv2.circle(lane_overlay, (int(right_line[y]), int(plot_y[y])), 5, (0, 0, 255), -1)  # Crvena

# Vraćamo sloj u originalnu perspektivu
inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
projected_overlay = cv2.warpPerspective(lane_overlay, inv_matrix, (width, height))

# Kombinujemo sa originalnom slikom
final_result = cv2.addWeighted(undistorted, 1, projected_overlay, 0.5, 0)


# 8. *******************PRIKAZIVANJE REZULTATA*******************

# Dodajemo tekstualne informacije na sliku
cv2.putText(final_result, f'Left Curvature: {left_curvature:.2f} m', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(final_result, f'Right Curvature: {right_curvature:.2f} m', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(final_result, f'Vehicle Position: {vehicle_offset:.2f} m', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


# *******************PRIKAZ SLIKA*******************
cv2.imshow('Original Chessboard Image', chessboard_image)
cv2.waitKey(0)
cv2.imshow('Undistorted Chessboard Image', undistorted_chessboard)
cv2.waitKey(0)
cv2.imshow('Original Image', test_img)
cv2.waitKey(0)
cv2.imshow('Undistorted Image', undistorted)
cv2.waitKey(0)
cv2.imshow('Binary Image', binary_img)
cv2.waitKey(0)
cv2.imshow('Warped Image (Perspective Change)', warped_image)
cv2.waitKey(0)
cv2.imshow('Lane Detection on Road', final_result)


# Prikaz histograma
# plt.figure(figsize=(10, 5))
# plt.title("Histogram detektovanih traka na 'Warped Image'")
# plt.xlabel("Širina slike (pikseli)")
# plt.ylabel("Intenzitet")
# plt.plot(histogram, color='blue')
# plt.grid()
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

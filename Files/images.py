import cv2
image = cv2.imread("../Images/baboon.jpg")
cv2.imshow("My Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2                                                              

image = cv2.imread('aerial4.png')                                       
image_new = cv2.resize(image, (750,750), interpolation=cv2.INTER_AREA)  
cv2.imwrite('aerial4_2.jpg', image_new)                                 

image = cv2.imread('ground4.png')                                       
image_new = cv2.resize(image, (1232,224), interpolation=cv2.INTER_AREA) 
cv2.imwrite('ground4_2.jpg', image_new)                                 

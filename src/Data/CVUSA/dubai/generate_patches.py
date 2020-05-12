import cv2                                                              

PATCHES_PER_DIM = 2
PATCH_DIM = 600
STRIDE = 100

map_h = 900*PATCHES_PER_DIM
map_w = 900*PATCHES_PER_DIM
path_prefix = 'sat/'

map_sat = cv2.imread('big_map3.png') 
h , w , _ = map_sat.shape                                      
map_resized = cv2.resize(map_sat, (map_w,map_h), interpolation=cv2.INTER_AREA) 
k = 0
j = 0
while(j*STRIDE + PATCH_DIM < map_h):
    i = 0
    while(i*STRIDE + PATCH_DIM < map_w):
        im = map_resized[j*STRIDE:j*STRIDE + PATCH_DIM -1, i*STRIDE:i*STRIDE + PATCH_DIM -1]
        cv2.imwrite(path_prefix + str(k) + '.jpg', im)      
        k += 1
        i +=1 
    j += 1                      
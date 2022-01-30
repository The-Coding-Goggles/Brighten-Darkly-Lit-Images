import cv2
import numpy as np

def process(img):
    alpha_beta = 3.5

    added_image = cv2.addWeighted(img,alpha_beta,img,alpha_beta,0)
    #added_image = cv2.GaussianBlur(added_image,(5,5),cv2.BORDER_DEFAULT)
    #added_image = cv2.detailEnhance(added_image, sigma_s=1, sigma_r=10000)
    added_image = cv2.fastNlMeansDenoisingColored(added_image,None,10,10,5,100)

    return added_image

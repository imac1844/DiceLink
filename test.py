import cv2
from PIL import Image
import numpy as np
import os

class FM:
    def feature_match():
        targetdir = 'C:/Users/imac1/OneDrive/Documents/Programs/dice_connection/Material-Dice/'

        for i in range(1, 4):
            matchlist = []
            image = Image.open('{}test{}.jpg'.format(targetdir, i))
            die = np.asarray(image)
            cv2.imshow("test{}".format(i), die)
            GrayImg = cv2.cvtColor(die, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create()
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            kp_target, des_target = orb.detectAndCompute(GrayImg,None)


            for root, dirs, files in os.walk(".\\DiceSets\\", topdown=True):
                for name in files:
                    pathname = os.path.join(root, name)
                    ref_img = cv2.imread(pathname, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow()
                    try:
                        kp_ref, des_ref = orb.detectAndCompute(ref_img,None)
                    except cv2.error:
                        continue

                    # print(type(des_target), "\n", type(des_ref))
                    matches = bf.match(des_target, des_ref)
                    matches = sorted(matches, key = lambda x:x.distance)

                    top5 = 0
                    for i in range (0,4):
                        top5 += matches[i].distance
                    matchlist.append((pathname, top5))

                    # img3 = cv2.drawMatches(die,kp_target,ref_img,kp_ref,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    # plt.imshow(img3),plt.show()


            matchlist = sorted(matchlist, key = lambda x:x[1])
            
            raw_result = matchlist[0][0].split('\\')
            value = raw_result[-1][:-4]
            dietype = raw_result[-2]

            print(dietype, "rolled", value )

            # print(matches[0].distance)


if __name__ == '__main__':
    FM.feature_match()
    cv2.waitKey()
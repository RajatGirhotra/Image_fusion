import cv2
import numpy as np
from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 10

img1 = cv2.imread('/Users/rajatgirhotra/Desktop/marker2.jpg',0)  #queryimage # left image
img2 = cv2.imread('/Users/rajatgirhotra/Desktop/3.jpg',0) #trainimage # right image


sift = cv2.xfeatures2d.SIFT_create()

#freakExtractor = cv2.xfeatures2d.FREAK_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)



# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []

# ratio test as per Lowe's paper
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
def warp(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result

if len(good)>MIN_MATCH_COUNT:
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

out = warp(img1, img2, M)
cv2.imwrite('out5.png',out)




#def drawlines(img1,img2,lines,pts1,pts2):
#    ''' img1 - image on which we draw the epilines for the points in img2
#        lines - corresponding epilines '''
#    r,c = img1.shape
#    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
#    for r,pt1,pt2 in zip(lines,pts1,pts2):
#        color = tuple(np.random.randint(0,255,3).tolist())
#        x0,y0 = map(int, [0, -r[2]/r[1] ])
#        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
#        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
#        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
#    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image


#lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,-1,2),2,F)
#lines1 = lines1.reshape(-1,3)
#img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image

#lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,-1,2), 1,F)
#lines2 = lines2.reshape(-1,3)
#img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

#plt.subplot(121),plt.imshow(img5)
#plt.subplot(122),plt.imshow(img3)
#plt.show()

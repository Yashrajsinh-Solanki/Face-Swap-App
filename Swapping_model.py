import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import dlib
# from matplotlib import pyplot as plt

def face_swap(image_1,image_2):
    
    if(image_1.shape[2] != 3 or image_2.shape[2] != 3):
        print("The input images do not have 3 channels (RGB)");
        return;
    
    #Converting the images to graysclae
    img1_gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    
    #Detectors and predictor 
    detect_face = dlib.get_frontal_face_detector()
    predict_out = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



    #Function to get the landmark points given a grayscale image as input
    def get_landmarks(img_gray):
    
        bound_rect = detect_face(img_gray)[0]
    
        #The values that act as the x and y coordinates of the outline on the face
        landmarks = predict_out(img_gray, bound_rect)
        landmarks_points = [] 

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

        points = np.array(landmarks_points, np.int32)

        return points,landmarks_points



    #Getting the landmark points and convexhull for image 1
    img1_points, img1_landmarks = get_landmarks(img1_gray)
    img1_convex = cv2.convexHull(img1_points)
    
    #Getting the landmark points and convexhull for image 2
    img2_points, img2_landmarks = get_landmarks(img2_gray)
    img2_convex = cv2.convexHull(img2_points)
    
    bound_rect = cv2.boundingRect(img1_convex)

    points_subdiv = cv2.Subdiv2D(bound_rect)
    points_subdiv.insert(img1_landmarks)
    
    triangles = points_subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)



    #Creating the delaunay triangles for image 1
    triangle_coords = []
    img1_cp = image_1.copy()

    def get_index(arr):
        index = 0
        if arr[0]:
            index = arr[0][0]
        return index

    for triangle in triangles :

        #Obtain the vertices of the triangle
        pt1 = (triangle[0], triangle[1])
        pt2 = (triangle[2], triangle[3])
        pt3 = (triangle[4], triangle[5])

        cv2.line(img1_cp, pt1, pt2, (255, 255, 255), 3,  0)
        cv2.line(img1_cp, pt2, pt3, (255, 255, 255), 3,  0)
        cv2.line(img1_cp, pt3, pt1, (255, 255, 255), 3,  0)

        index_pt1 = np.where((img1_points == pt1).all(axis=1))
        index_pt1 = get_index(index_pt1)
        index_pt2 = np.where((img1_points == pt2).all(axis=1))
        index_pt2 = get_index(index_pt2)
        index_pt3 = np.where((img1_points == pt3).all(axis=1))
        index_pt3 = get_index(index_pt3)

        #Append coordinates only if triangle exists
        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            vertices = [index_pt1, index_pt2, index_pt3]
            triangle_coords.append(vertices)

    height, width, channels = image_2.shape
    new_img1 = np.zeros((height, width, channels), np.uint8)
    img2_new_img1 = np.zeros((height, width, channels), np.uint8)

    height, width = img1_gray.shape
    new_img_mask = np.zeros((height, width), np.uint8)



    for triangle in triangle_coords:

        #Coordinates of the triangle for the first person
        pt1 = img1_landmarks[triangle[0]]
        pt2 = img1_landmarks[triangle[1]]
        pt3 = img1_landmarks[triangle[2]]

        #Obtain the triangle coordinates
        (x, y, w, h) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
        cropped_triangle = image_1[y: y+h, x: x+w]
        
        cropped_mask = np.zeros((h, w), np.uint8)

        points = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32)
        cv2.fillConvexPoly(cropped_mask, points, 255)

        cv2.line(new_img_mask, pt1, pt2, 255)
        cv2.line(new_img_mask, pt2, pt3, 255)
        cv2.line(new_img_mask, pt1, pt3, 255)

        lines_space = cv2.bitwise_and(image_1, image_1, mask=new_img_mask)

        #Obtaining the delaunay triangles for the person 2

        #Coordinates of the triangle for image 2
        pt1 = img2_landmarks[triangle[0]]
        pt2 = img2_landmarks[triangle[1]]
        pt3 = img2_landmarks[triangle[2]]

        #Obtain the triangle coordinates
        (x, y, w, h) = cv2.boundingRect(np.array([pt1, pt2, pt3], np.int32))
        cropped_mask2 = np.zeros((h,w), np.uint8)

        points2 = np.array([[pt1[0]-x, pt1[1]-y], [pt2[0]-x, pt2[1]-y], [pt3[0]-x, pt3[1]-y]], np.int32)
        cv2.fillConvexPoly(cropped_mask2, points2, 255)


        #Warping the triangle to match the dimensions
        points =  np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)  
        dist_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=cropped_mask2)

        img2_new_img1_bound_rect_area = img2_new_img1[y: y+h, x: x+w]
        img2_new_img1_bound_rect_area_gray = cv2.cvtColor(img2_new_img1_bound_rect_area, cv2.COLOR_BGR2GRAY)

        masked_triangle = cv2.threshold(img2_new_img1_bound_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        dist_triangle = cv2.bitwise_and(dist_triangle, dist_triangle, mask=masked_triangle[1])

        img2_new_img1_bound_rect_area = cv2.add(img2_new_img1_bound_rect_area, dist_triangle)
        img2_new_img1[y: y+h, x: x+w] = img2_new_img1_bound_rect_area



        #Combining the masks
    img2_img1_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_img1_mask, img2_convex, 255)
    img2_img1_mask = cv2.bitwise_not(img2_head_mask)

    img2_maskless = cv2.bitwise_and(image_2, image_2, mask=img2_img1_mask)
    result = cv2.add(img2_maskless, img2_new_img1)
    
    #Making the seamlessClone for better output
    (x, y, w, h) = cv2.boundingRect(img2_convex)
    face_center = (int((x+x+w)/2), int((y+y+h)/2))

    final_swapped_image = cv2.seamlessClone(result, image_2, img2_head_mask, face_center, cv2.NORMAL_CLONE)
    
    return final_swapped_image

# #Testing the function
# imag1 = cv2.imread('girl face kyleroxas.jpg')
# imag2 = cv2.imread('boy body harsh.jpg')

# #Swap faces
# final_image_1 = face_swap(imag1,imag2)
# final_image_2 = face_swap(imag2,imag1)

# fig, axs = plt.subplots(2,3,figsize=(15,15));

# axs[0,0].imshow(cv2.cvtColor(imag1,cv2.COLOR_BGR2RGB));
# axs[0,0].set_title("Original image 1")
# axs[0,1].imshow(cv2.cvtColor(imag2,cv2.COLOR_BGR2RGB));
# axs[0,1].set_title("Original image 2")
# axs[0,2].imshow(cv2.cvtColor(final_image_1,cv2.COLOR_BGR2RGB));
# axs[0,2].set_title("Swapping image 1 contents to image 2")

# axs[1,0].imshow(cv2.cvtColor(imag2,cv2.COLOR_BGR2RGB));
# axs[1,0].set_title("Original image 2")
# axs[1,1].imshow(cv2.cvtColor(imag1,cv2.COLOR_BGR2RGB));
# axs[1,1].set_title("Original image 1")
# axs[1,2].imshow(cv2.cvtColor(final_image_2,cv2.COLOR_BGR2RGB));
# axs[1,2].set_title("Swapping image 2 contents to image 1")

# [axi.axis('off') for axi in axs.ravel()]
# fig.show()
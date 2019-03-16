import numpy as np
import scipy.spatial as spatial
import scipy.cluster as clstr
from time import time
from collections import defaultdict
from functools import partial
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import os, glob, skimage, cv2, shutil
from scipy.spatial.distance import cdist
import os
from mongodbprovider import MongoDBProvider

import configurations

SQUARE_SIDE_LENGTH = 227
categories = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']

class ChessBoardSegmentation:

    def __init__(self, mongo_db_provider = None):
        self._mongo_db_provider = mongo_db_provider
        split_images_location = os.path.join(configurations.IMAGES_LOCATION, "splitimages")
        if not os.path.exists(split_images_location):
            print("Created split images at location provided.")
            os.makedirs(split_images_location)
        

    def auto_canny(self,image, sigma=0.33):
        """
        Canny edge detection with automatic thresholds.
        """
        # compute the median of the single channel pixel intensities
        v = np.median(image)
    
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
    
        # return the edged image
        return edged

    def hor_vert_lines(self,lines):
        """
        A line is given by rho and theta. Given a list of lines, returns a list of
        horizontal lines (theta=90 deg) and a list of vertical lines (theta=0 deg).
        """
        h = []
        v = []
        for distance, angle in lines:
            if angle < np.pi / 4 or (angle > (np.pi - (np.pi / 4))):
                v.append([distance, angle])
            else:
                h.append([distance, angle])
        return h, v

    def intersections(self, h, v):
        """
        Given lists of horizontal and vertical lines in (rho, theta) form, returns list
        of (x, y) intersection points.
        """
        points = []
        for d1, a1 in h:
            for d2, a2 in v:
                A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
                b = np.array([d1, d2])
                point = np.linalg.solve(A, b)
                points.append(point)
                
        # print(points)
        print("Number of points: " + str(len(points)))
        return np.array(points)

    def cluster(self, points, max_dist=50):
        """
        Given a list of points, returns a list of cluster centers.
        """
        Y = spatial.distance.pdist(points)
        Z = clstr.hierarchy.single(Y)
        T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
        clusters = defaultdict(list)
        for i in range(len(T)):
            clusters[T[i]].append(points[i])
        clusters = clusters.values()
        clusters = map(lambda arr: (np.mean(np.array(arr)[:,0]), np.mean(np.array(arr)[:,1])), clusters)
        clusters_list = list(clusters)
        print(clusters_list)
        print("Number of clusters: " + str(len(clusters_list)))
        return clusters_list

    def closest_point(self, points, loc):
        """
        Returns the list of points, sorted by distance from loc.
        """
        dists = np.array(map(partial(spatial.distance.euclidean, loc), points))
        return points[dists.argmin()]

    def find_corners(self, points, img_dim):
        """
        Given a list of points, returns a list containing the four corner points.
        """
        print("Image dimensions: " + str(img_dim))
        center_point = self.closest_point(points, (img_dim[0] / 2, img_dim[1] / 2))
        print("Closest point to {0}, {1} is {2}, {3}".format(img_dim[0] / 2, img_dim[1] / 2, center_point[0], center_point[1]))
        points.remove(center_point)
        center_adjacent_point = self.closest_point(points, center_point)
        points.append(center_point)
        grid_dist = spatial.distance.euclidean(np.array(center_point), np.array(center_adjacent_point))
        print("Grid distance: " + str(grid_dist))
        
        img_corners = [(0, 0), (0, img_dim[1]), img_dim, (img_dim[0], 0)]
        board_corners = []
        tolerance = 0.25 # bigger = more tolerance
        for img_corner in img_corners:
            while True:
                cand_board_corner = self.closest_point(points, img_corner)
                points.remove(cand_board_corner)
                cand_board_corner_adjacent = self.closest_point(points, cand_board_corner)
                corner_grid_dist = spatial.distance.euclidean(np.array(cand_board_corner), np.array(cand_board_corner_adjacent))
                print("Corner grid distance: " + str(corner_grid_dist))
                if corner_grid_dist > (1 - tolerance) * grid_dist and corner_grid_dist < (1 + tolerance) * grid_dist:
                    print("Added board corner: " + str(cand_board_corner))
                    points.append(cand_board_corner)
                    board_corners.append(cand_board_corner)
                    break
        return board_corners

    def four_point_transform(self,img, points, square_length=SQUARE_SIDE_LENGTH):
        board_length = square_length * 8
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [0, board_length], [board_length, board_length], [board_length, 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, (board_length, board_length))

    def find_board(self, fname, outputs_folder_name = None, is_file=True):
        """
        Given a filename or the image, returns the board image.
        """
        start = time()

        if is_file:
            img = cv2.imread(fname)
        else:
            img = fname

        assert img is not None
        cv2.imwrite(os.path.join(configurations.IMAGES_LOCATION,  'img.jpg'), img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3, 3)) # TODO auto adjust the size of the blur
        
        # Canny edge detection
        edges = self.auto_canny(gray)
        cv2.imwrite(os.path.join(configurations.IMAGES_LOCATION,  'edges.jpg'), edges)
        assert np.count_nonzero(edges) / float(gray.shape[0] * gray.shape[1]) < 0.015
        print("Number of edges: "+str(edges))

        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        print(lines.shape)
        
        lines = np.reshape(lines, (-1, 2))
        print(lines.shape)
        
        # Compute intersection points
        h, v = self.hor_vert_lines(lines)
        print(h)
        print(v)
        assert len(h) >= 9
        assert len(v) >= 9
        
        print("Number of horizontal lines: " + str(len(h)))
        print("Number of vertical lines: " + str(len(v)))
        points = self.intersections(h, v)
            
        if True:
            for rho, theta in lines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 4000*(-b))
                y1 = int(y0 + 4000*(a))
                x2 = int(x0 - 4000*(-b))
                y2 = int(y0 - 4000*(a))
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

            if outputs_folder_name:
                cv2.imwrite(os.path.join(configurations.IMAGES_LOCATION, outputs_folder_name +  '_lines.jpg'), img)
        
        # Cluster intersection points
        points = self.cluster(points)
        
        if True:
            for point in points:
                cv2.circle(img, tuple(point), 5, (0,0,255), -1)
            
            if outputs_folder_name:
                cv2.imwrite(os.path.join(configurations.IMAGES_LOCATION, outputs_folder_name +  '_all_points.jpg'), img)
        
        # Find corners
        img_shape = np.shape(img)
        corner_points = self.find_corners(points, (img_shape[1], img_shape[0]))
        
        if True:
            for point in corner_points:
                cv2.circle(img, tuple(point), 5, (0,255,0), -1)
            
            if outputs_folder_name:
                cv2.imwrite(os.path.join(configurations.IMAGES_LOCATION, outputs_folder_name +  '_corner_points.jpg'), img)
        
        # Perspective transform
        new_img = self.four_point_transform(img, corner_points)

        return new_img

    def split_board(self, img, create_files=True):
        """
        Given a board image, returns an array of 64 smaller images.
        """
        row = "abcdefgh"
        arr = []
        sq_len = img.shape[0] // 8
        for i in range(8):
            for j in range(8):
                image = img[i * sq_len : (i + 1) * sq_len, j * sq_len : (j + 1) * sq_len]
                position = str(row[j]) + str(8 - i)
                arr.append({"image": image, "position": position})

                if create_files:
                    cv2.imwrite(os.path.join(configurations.IMAGES_LOCATION, "splitimages", position+".jpg"), image)
        return arr

    def segment_board_corners_provided(self, fname, is_file=True):
        """
        Given a filename or the image, segments the board.
        """
        start = time()

        if is_file:
            img = cv2.imread(fname)
        else:
            img = fname  

        assert img is not None 

        cv2.imwrite(os.path.join(configurations.IMAGES_LOCATION,  'input_img.jpg'), img)
        return self.split_board(img)     

if __name__ == '__main__':
    cbs = ChessBoardSegmentation()
    #board = cbs.find_board('C:\\Users\\Sriraj\\Documents\\Boardgames\\API\\outputs\\IMG_7318.jpg', "IMG_7318")
    #cv2.imwrite('C:\\Users\\Sriraj\\Documents\\Boardgames\\API\\outputs\\IMG_7318.jpg', board)
    #cbs.split_board(board)

    cbs.segment_board_corners_provided("H:\\AR-ExtendingOnlineGames\\my_board\\chess_board.jpg")
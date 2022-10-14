from typing import List

import cv2
import numpy as np
from scipy.spatial import distance


def get_2d_gaussian(size: List[int], fwhm: float = 3, center: List[float] = None):
    x = np.expand_dims(np.arange(0, size[0], 1), axis=1)
    y = np.expand_dims(np.arange(0, size[1], 1), axis=0)

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[1]
        y0 = center[0]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def get_2d_mix_gaussian(size: List[int], fwhm_list: List[float], center_list = List[List[float]]):
    mix_gaussian = 0
    for fwhm, center in zip(fwhm_list, center_list):
        mix_gaussian += get_2d_gaussian(size, fwhm, center)
    return mix_gaussian

def l2_distance(n: List[float], m: List[float]) -> float:
    if len(n) != len(m):
        raise "Inputs must have equal length"
    return sum((y-x)**2 for x, y in zip(n, m)) ** 0.5

def f_dst_weights(frame, x, y, w, h):
    X, Y, _ = frame.shape
    weights = np.zeros((X, Y)) + 0.15

    # defining a zone of curiosity
    ww = min(x+int(1.5*w), X) - max(x-int(w/2), 0)
    hh = min(y+int(1.5*h), Y) - max(y-int(h/2), 0)
    template = np.indices((ww, hh))
    template[0] += max(x-int(w/2), 0)
    template[1] += max(y-int(h/2), 0)

    # the center from which the distance will be counted
    target = np.array([[x+int(w/2),y+int(h/2)]])

    # Calculate the distance from the center to all points of interest.
    d = distance.cdist(template.reshape(2, ww*hh).T, target, 'euclidean').reshape(ww,hh)
    # we use the Gaussian distribution to transform the distance
    std = 25
    gaussian = (1/(std*((2*np.pi)**0.5)))*np.exp( -((d)**2)/(2*std*std) )
    # normalization
    cv2.normalize(gaussian, gaussian, 0.15, 1, cv2.NORM_MINMAX)
    # Creating weights for the density.
    weights[max(x-int(w/2), 0):min(x+int(1.5*w), X), max(y-int(h/2), 0): min(y+int(1.5*h), Y)] = gaussian
    # cv2.imshow('Weights', weights)
    return weights

def get_gradient_magnitude(frame_g):
    dx = cv2.Sobel(frame_g,cv2.CV_64F,1,0,ksize=3)
    dy = cv2.Sobel(frame_g,cv2.CV_64F,0,1,ksize=3)
    return  np.hypot(dx,dy).astype('uint8')

def get_gradient_orientation(frame_g):
    dx = cv2.Sobel(frame_g,cv2.CV_64F,1,0,ksize=3)
    dy = cv2.Sobel(frame_g,cv2.CV_64F,0,1,ksize=3)
    # Compute the orientation
    return  (np.arctan2(dy,dx) * 180 / np.pi)

def build_r_table(obj):
    X,Y =  obj.shape
    gradient_magnitude = get_gradient_magnitude(obj)
    _ , filtered = cv2.threshold(gradient_magnitude, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow('r_table', filtered)
    orientation = get_gradient_orientation(filtered)
    orientation[filtered == 0] = -255
    unique_orientation = np.unique(orientation)

    r_table = dict()
    center = np.array([[int(X/2) ,int(Y/2)]])

    for teta in unique_orientation:
        if teta == -255:
            continue
        r_table[teta] = center - np.argwhere(orientation == teta)

    return r_table

def transform_hough(image, r_table):
    X, Y = image.shape
    gradient_magnitude = get_gradient_magnitude(image)
    _ , filtered = cv2.threshold(gradient_magnitude, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow('filtered_gradient_magnitude', filtered)
    orientation = get_gradient_orientation(filtered)
    orientation[filtered == 0] = -255

    vote = np.zeros(image.shape)

    for teta in r_table:
        tmp = np.argwhere(orientation == teta)
        if tmp.shape[0] == 0 :
            continue
        for r in r_table[teta]:
            ind_for_vote = tmp + r
            ind_for_vote = ind_for_vote[ (ind_for_vote[:,0] < X) & (ind_for_vote[:,0] > 0) &(ind_for_vote[:,1] < Y) & (ind_for_vote[:,1] > 0)  ]
            vote[ind_for_vote[:,0], ind_for_vote[:,1]] += 1
    return vote

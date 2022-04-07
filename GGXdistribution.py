#AllanTian
#Assignment 2: Multispectral Sampling, BRDF

import math
import numpy as np 
from PIL import Image

#final colors are computed starting at line 233
#using lightsamps[5] uses LED light, lightsamps[5] uses sunlight 

#assuming air as incidence
#magnetic surface
def nonmagR(n, thetaI):
    thetaT = math.asin(math.sin(thetaI) / n)
    RP = ((n*math.cos(thetaI) - math.cos(thetaT)) / (n*math.cos(thetaI) + math.cos(thetaT)))**2
    RS = ((math.cos(thetaI) - n*math.cos(thetaT)) / (math.cos(thetaI) + n*math.cos(thetaT)))**2
    R = (RP + RS) / 2
    return R
#magnetic surface
def magR(n, k, thetaI):
    RP = (n**2 + k**2 - 2*n*math.cos(thetaI) + math.cos(thetaI)**2) / (n**2 + k**2 + 2*n*math.cos(thetaI) + math.cos(thetaI)**2)
    RS = RP*((n**2 + k**2)*math.cos(thetaI)**2 - 2*n*math.cos(thetaI)*math.sin(thetaI)**2 +math.sin(thetaI)**4) / ((n**2 + k**2)*math.cos(thetaI)**2 + 2*n*math.cos(thetaI)*math.sin(thetaI)**2 +math.sin(thetaI)**4)
    R = (RP + RS) / 2
    return R
# (F, lray, -vray, norm, roughness)
def cooktorr(F, L, V, N, m):
    H = -(L+V)/2
    cosalpha = np.dot(N, H)
    D = m**2 / (math.pi * ((cosalpha**2) * (m**2 - 1) + 1)**2) #GGX distribution
    masking = (2*np.dot(N, H)*np.dot(N, V)) / np.dot(V, H)
    shadowing = (2*np.dot(N, H)*np.dot(N, L)) / np.dot(V, H)
    G = min(1, masking, shadowing)
    cooktorr = (F * D * G) / (4*math.pi*np.dot(N, L)*np.dot(N, V))
    print(F, D, G)
    print(cooktorr)
    return cooktorr

class obj:
    pass

sphere0 = obj()
sphere0.cent = np.array([-15, 8, 24])
sphere0.rad = 6
sphere0.diff = np.array([185, 148, 69])
sphere0.mat = np.array([ #Gold (Au)
    [1.4684, 1.3831, 0.97112, 0.42415, 0.24873, 0.15557, 0.13100], #n
    [1.9530, 1.9155, 1.8737, 2.4721, 3.0740, 3.6024, 4.0624], #k
])
sphere0.rough = 0

sphere1 = obj()
sphere1.cent = np.array([0, 15, 24])
sphere1.rad = 6
sphere1.diff = np.array([185, 148, 69])
sphere1.mat = np.array([ #Gold (Au)
    [1.4684, 1.3831, 0.97112, 0.42415, 0.24873, 0.15557, 0.13100], #n
    [1.9530, 1.9155, 1.8737, 2.4721, 3.0740, 3.6024, 4.0624], #k
])
sphere1.rough = 0.45

sphere2 = obj()
sphere2.cent = np.array([15, 8, 24])
sphere2.rad = 6
sphere2.diff = np.array([185, 148, 69])
sphere2.mat = np.array([ #Gold (Au)
    [1.4684, 1.3831, 0.97112, 0.42415, 0.24873, 0.15557, 0.13100], #n
    [1.9530, 1.9155, 1.8737, 2.4721, 3.0740, 3.6024, 4.0624], #k
])
sphere2.rough = 0.6

sphere3 = obj()
sphere3.cent = np.array([-15, -8, 24])
sphere3.rad = 6
sphere3.diff = np.array([180, 192, 180])
sphere3.mat = np.array([ #Silicon (Si)
    [5.5674, 4.6784, 4.2992, 4.0870, 3.9485, 3.8515, 3.7838], #n
    [0.38612, 0.14851, 0.070425, 0.040882, 0.027397, 0.016460, 0.012170], #k
])
sphere3.rough = 0.45

sphere4 = obj()
sphere4.cent = np.array([0, -15, 24])
sphere4.rad = 6
sphere4.diff = np.array([192, 180, 180])
sphere4.mat = np.array([ #Chromium (Cr)
    [2.0150, 2.3230, 2.7804, 3.1812, 3.1943, 3.1071, 3.0536], #n
    [2.8488, 3.1350, 3.3048, 3.3291, 3.3000, 3.3314, 3.3856], #k
])
sphere4.rough = 0.45

sphere5 = obj()
sphere5.cent = np.array([15, -8, 24])
sphere5.rad = 6
sphere5.diff = np.array([42, 42, 54])
sphere5.mat = np.array([ #Carbon (C)
    [2.4602, 2.4392, 2.4289, 2.4227, 2.4148, 2.4093, 2.4065], #n
    [0, 0, 0, 0, 0, 0, 0], #k
])
sphere5.rough = 0.45

lightsamps = np.array([
    [400, 450, 500, 550, 600, 650, 700], #wavelength
    [0.0143, 0.3362, 0.0049, 0.4334, 1.0622, 0.2835, 0.0114], #x-values
    [0.0004, 0.0380, 0.3230, 0.9950, 0.6310, 0.1070, 0.0041], #y-values
    [0.0679, 1.7721, 0.2720, 0.0087, 0.0008, 0.0000, 0.0000], #z-values
    [0.51126, 0.96813, 1.1336, 1.4638, 1.4398, 1.3018, 1.1184], #sunlight intensity
    [0.0018743, 0.15759, 0.036911, 0.11352, 0.089698, 0.035154, 0.0097982] #ledtech (par20) bulb intensity
])

#XYZ to RGB
convert2RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314], 
    [-0.9692660, 1.8760108, 0.0415560], 
    [0.0556434, -0.2040259, 1.0572252]
])

#RGB to XYZ (just in case)
convert2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

sunx = np.dot(lightsamps[1],lightsamps[5]/9)
suny = np.dot(lightsamps[2],lightsamps[5]/9)
sunz = np.dot(lightsamps[3],lightsamps[5]/9)

finalxyz = np.array([sunx, suny, sunz])
lightcolor = np.matmul(convert2RGB, finalxyz)
brightness = max(lightcolor)
brightness = 255/brightness

#unit vector conversion
def unit(x):
    a = np.array(x)/np.linalg.norm(np.array(x))
    return a

#setting up eye, light, viewport, etc...
up = unit([0, 1, 0]) #local up direction, change to rotate view 
#light information
light = np.array([0,0,-3]) #light location 
lview = unit([2, -1, 1]) #light direction for spotlight/directional/etc later \\ default:[0.5, -0.2, 2]
l0 = unit(np.cross(up, lview)) #light right/left
l3 = unit(np.cross(lview, l0)) #light up/down
eye = np.array([0, 0, -24]) #eye center
view = unit([0, 0, 1]) #eye direction
dist = 24
xscale = 54 #x-fov 
yscale = 54 #y-fov  
n2 = view #view direction
n0 = unit(np.cross(up, view)) #right/left
n3 = unit(np.cross(n2, n0)) #up/down
vcenter = eye + dist*n2 #viewport center
vcorner = vcenter - (xscale/2)*n0 - (yscale/2)*n3
#output size
xmax = int(400*1) 
ymax = int(400*1)

class obj:
    pass

#shoots to all objects, stores the closest intersection
def t(x, y): 
    tlist = None #stores t values for nearest interesection
    count = 0
    for i in objects:
        sphrdist = x - i.cent
        b = np.dot(sphrdist, y)
        c = np.dot(sphrdist, sphrdist) - i.rad**2
        #arrays that will be used to store information
        sphere = np.zeros((6,3))
        if b**2 - c >= 0 and b <= 0:
            sphere[0] = min([-b - (b**2 - c)**0.5, -b + (b**2 - c)**0.5]) #t-value
            if sphere[0][0] > 0: #not zero to account for rounding
                sphere[1] = x + sphere[0]*y #intersection
                sphere[2] = unit(sphere[1] - i.cent) #normal
                sphere[4] = max([-b - (b**2 - c)**0.5, -b + (b**2 - c)**0.5]) #further t-value
                sphere[3] = i.diff
                sphere[5] = count
            if tlist is None:
                tlist = sphere
            elif tlist[0][0] > sphere[0][0]:
                tlist = sphere    
        else:
            pass
        count += 1
    if tlist is None:
        return False
    else:
        return tlist #[t-value, intersection, normal, further t-value, diffuse, object number]

#antialias jittering
def antialias(x, y):
    w = 1
    h = 1
    location = np.zeros((h, w, 3))
    for j in range(h):
        for i in range(w):
            subx = np.around(x + (i + np.random.random_sample())/w, 2)
            suby = np.around(y + (j + np.random.random_sample())/h, 2)
            location[j][i] = vcorner + xscale*n0*(subx/xmax) + yscale*n3*(suby/ymax)
    return(location)

#objects to be rendered
objects = [sphere0, sphere1, sphere2, sphere3, sphere4, sphere5]

out = np.zeros((ymax, xmax, 3), dtype=np.uint8)
print("\nInitializing")
var_out = input("Output name: ")
print("\nProgress: 0%")
#begins rendering for each pixel
for j in range(ymax):
    for i in range(xmax):
        subpixel = antialias(i, j)
        #antialias jitter
        for ll in range(np.shape(subpixel)[0]):
            for kk in range(np.shape(subpixel)[1]):
                vloc = subpixel[ll][kk]
                vray = np.array([0,0,1]) #unit(vloc - eye)
                info = t(vloc, vray) #t(eye,vray) #finds intersections and other information, if no intersect, returns false
                if info is not False: #if there are intersections... uses info about the intersection (see above)
                    diffuse = info[3] #base color
                    p = info[1]
                    norm = unit(info[2])            
                    lray = unit(light-p) #pointlight
                    # refl = unit(-lray + 2*np.dot(lray, norm)*norm) #reflected light ray
                    ndotl = max(np.dot(lray, norm), 0) #diffuse based off light and surface norm
                    speccolor = np.zeros((3), dtype=np.uint8)
                    XYZ = np.zeros((7,3))
                    RGB = np.zeros((3), dtype=np.uint8)
                    s = 0.9 #specularity interpolation             
                    intersectedobj = objects[int(info[5][0])] #intersected objected information
                    
                    #this is the get color function                    
                    if len(intersectedobj.mat) == 2: #checks if there's a k value; ie if it's a magnetic surface
                        roughness = intersectedobj.rough
                        count = 0
                        for n in intersectedobj.mat[0]: #n for each wavelength sample for the intersected object
                            k = intersectedobj.mat[1][count] #k
                            thetaI = math.acos(np.dot(lray,norm))
                            F = magR(n, k, thetaI) #Fresnel calculation
                            spec = cooktorr(F, lray, -vray, norm, roughness)
                            lightXYZ = np.array([ #XYZ value of light at respective wavelength
                                lightsamps[1][count],
                                lightsamps[2][count],
                                lightsamps[3][count]
                            ])
                            lightintensity = lightsamps[5][count]*brightness
                            avgintensity = np.mean(lightsamps[5])
                            XYZ[count] = lightintensity * lightXYZ * spec #specular XYZ for each wavelength
                            count += 1
                        RGB = np.sum(XYZ, axis=0)
                        RGB = np.matmul(convert2RGB, RGB) #summed XYZ of all wavelengths and converted to RGB
                        overall = (avgintensity*diffuse*(1-s) + RGB*s)*ndotl

                    else: #same thing, but runs with the corresponding fresnel equation
                        roughness = intersectedobj.rough
                        count = 0
                        for n in intersectedobj.mat[0]: #n for each wavelength sample for the intersected object
                            thetaI = math.acos(np.dot(lray,norm))
                            F = nonmagR(n, thetaI) #Fresnel calculation
                            spec = cooktorr(F, lray, -vray, norm, roughness)
                            lightXYZ = np.array([ #XYZ value of light at respective wavelength
                                lightsamps[1][count],
                                lightsamps[2][count],
                                lightsamps[3][count]
                            ])
                            lightintensity = lightsamps[5][count]*brightness
                            avgintensity = np.mean(lightsamps[5])
                            XYZ[count] = lightintensity * lightXYZ * spec #specular XYZ for each wavelength
                            count += 1
                        RGB = np.sum(XYZ, axis=0) 
                        RGB = np.matmul(convert2RGB, RGB) #summed XYZ of all wavelengths and converted to RGB
                        overall = (avgintensity*diffuse*(1-s) + RGB*s)*ndotl

                    subpixel[ll][kk] = overall
                else: #if no intersection, display black
                    subpixel[ll][kk] = np.array([0, 0, 0])
        color = np.array(np.mean(np.mean(subpixel, axis=0), axis=0), dtype=np.uint8) #averages subpixel samples
        j = ymax - (j + 1) #flips to corresponding pixel
        #creates square of light color
        if 7*ymax/15 < j and j < 8*ymax/15 and 7*ymax/15 < i and i < 8*ymax/15:
            out[j][i] = np.matmul(convert2RGB, finalxyz*brightness)
        else:
            out[j][i] = color
    progress = int(ymax/5)
    if j != 0 and j%progress == 0:
        print(int(j/progress)*20, "%")


#saves image
fin = Image.fromarray(out, 'RGB')
fin.save('{}.png'.format(var_out))
print('100%\nDone')

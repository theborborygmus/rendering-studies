import sys
import math
import numpy as np 
from PIL import Image

#defining colors
print("\nInitializing")
texture = Image.open('new0.png')
texture = np.array(texture)
var_out = input("Output name: ")
framestart = int(input("Frame start: "))
frameend = int(input("Frame end: "))

#specular interpolation
def s0(x, y, m):
    maxi = 1 + m**2 #more roughness => higher max
    mini = 0.6 - (min(m**3, 0.6)) #more roughness => higher min
    s = max(np.dot(x,y), 0)
    s = (s-mini)/(maxi-mini)
    if s <= 0:
        s = 0    
    elif s >= 1:
        s = 1
    else:
        s = s
    return s

#resolves rounding issue
def issue(y):
    x = y 
    if abs(x) > 1: 
        if x > 1:
            x = 1
        else:
            x = -1
    return x

def rotx(x, y):
    rotx = np.array([[1, 0, 0], 
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]])
    return np.matmul(rotx, y)
def roty(x, y):
    roty = np.array([
        [np.cos(x), 0, np.sin(x)], 
        [0, 1, 0],
        [-np.sin(x), 0, np.cos(x)]])
    return np.matmul(roty, y)
def rotz(x, y):
    rotz = np.array([
        [np.cos(x), -np.sin(x), 0], 
        [np.sin(x), np.cos(x), 0],
        [0, 0, 1]])
    return np.matmul(rotz, y)

#unit vector conversion
def unit(x):
    a = np.array(x)/np.linalg.norm(np.array(x))
    return a

#setting up eye, light, viewport, etc...
up = unit([0, 1, 0]) #local up direction, change to rotate view 
#light information
light = 1.5*np.array([0, 6, -9]) #light location \\ default: [-9, 3, -6]
lview = unit([1, -1, 1]) #light direction for spotlight/directional/etc 
l0 = unit(np.cross(up, lview)) #light right/left
l3 = unit(np.cross(lview, l0)) #light up/down
lcolor = np.array([255, 255, 255]) #use for specular color

#eye information
eye = np.array([-12,7.5,-10]) #eye location default: [-17, 3, -17]
view = unit([1.2, -1, 1]) #eye direction default: [1, -0.1, 1]
dist = 30
xscale = 36 #x-fov 
yscale = 27 #y-fov  
n2 = view #view direction
n0 = unit(np.cross(up, view)) #right/left
n3 = unit(np.cross(n2, n0)) #up/down
vcenter = eye + dist*n2 #viewport center
vcorner = vcenter - (xscale/2)*n0 - (yscale/2)*n3
#output size
xmax = int(400*1.5) 
ymax = int(300*1.5)

#projection point
projectpoint = np.array([0,6,0])

#area of face
def A(x):
    x.A = []
    for i in range(len(x.f)):
        p2 = x.v[x.f[i][2][0]-1]
        p1 = x.v[x.f[i][1][0]-1]
        p0 = x.v[x.f[i][0][0]-1]
        A = np.cross(p1 - p0, p2 - p0)
        x.A.append(A)
    return x.A #returns area of face

class obj:
    pass

#sphere parameters
sphere0 = obj()
sphere0.type = "sphere"
sphere0.center = np.array([0, 9, 0])
sphere0.radius = 12
sphere0.diffuse = np.array([255,100,100])
sphere0.mat = "diffuse" #False, "diffuse", "glass", "mirror"
sphere0.n = 1.5
sphere0.rough = 0.9
sphere0.id = "sphere0"

#sphere parameters
sphere1 = obj()
sphere1.type = "sphere"
sphere1.center = np.array([0, 9, 0])
sphere1.radius = 11
sphere1.diffuse = np.array([255, 255, 100])
sphere1.mat = "diffuse" #False, "diffuse", "glass", "mirror"
sphere1.n = 1.5
sphere1.rough = 0.3
sphere1.id = "sphere1"

#sphere parameters
sphere2 = obj()
sphere2.type = "sphere"
sphere2.center = np.array([0, 9, 0])
sphere2.radius = 13
sphere2.diffuse = np.array([100, 100, 255])
sphere2.mat = "diffuse" #False, "diffuse", "glass", "mirror"
sphere2.n = 1.5
sphere2.rough = 0.6
sphere2.id = "sphere1"



#returns information about a ray & plane intersection
def planeintersect(i, x, y):
    #arrays that will be used to store information
    plane = obj()
    if np.dot(i.nrm, y) != 0:
        plane.tvalue = -(np.dot(i.nrm, (x - i.center)))/np.dot(i.nrm, y) #t-value
        if plane.tvalue > 0: #not zero to account for rounding
            intersection = x + plane.tvalue*y #intersection
            plane.normal = i.nrm #normal
            plane.color = i.diffuse
            plane.thickness = [0.03, i.id, 1]
            plane.id = i.id
            plane.type = i.type
            plane.nratio = i.n
            plane.rough = i.rough
            plane.tvalue2 = None
            if i.texture is not "None":
                scale = 36
                xshift = i.center - (i.x*scale)/2
                yshift = i.center - (i.y*scale)/2
                u = (np.dot(i.x, (intersection-xshift)))/scale
                v = (np.dot(i.y, (intersection-yshift)))/scale
                u = int((u)*np.shape(texture)[1])
                v = int((v)*np.shape(texture)[0])
                if 0 <= u < np.shape(texture)[1] and 0 <= v < np.shape(texture)[0]:
                    plane.color = texture[v][u][:3]
                    return plane
                else:
                    return False
            else:
                return plane
        else:
            return False
    else:
        return False 
#returns information about a ray & sphere intersection
def sphereintersect(i, x, y):
    sphrdist = x - i.center
    b = np.dot(sphrdist, y)
    c = np.dot(sphrdist, sphrdist) - i.radius**2
    sphere = obj()
    if b**2 - c >= 0 and b <= 0:
        sphere.tvalue = min([-b - (b**2 - c)**0.5, -b + (b**2 - c)**0.5]) #t-value
        if sphere.tvalue > -0.001: #not zero to account for rounding
            sphere.tvalue2 = max([-b - (b**2 - c)**0.5, -b + (b**2 - c)**0.5]) #further t-value
            sphere.center = i.center #normal
            sphere.color = i.diffuse
            sphere.thickness = sphere.tvalue2 - sphere.tvalue
            sphere.id = i.id
            sphere.type = i.type
            sphere.nratio = i.n
            sphere.rough = i.rough
            return sphere
        else:
            return False
    else:
        return False 

#finding intersections based off x, the location, and y, the ray from x
def t(x, y, z, objectlist): 
    tlist = {'placehold': 'placehold'} #stores info
    for i in objectlist:
        if i.id == z: 
            pass
        else:
            i.intersect = None
            if i.type is "plane":
                i.intersect = planeintersect(i, x, y)
            if i.type is "sphere":
                i.intersect = sphereintersect(i, x, y)

            if i.intersect is False:
                pass
            else:
                tlist[i.intersect.tvalue] = i.intersect
                if i.type is "sphere":
                    tlist[i.intersect.tvalue2] = i.intersect
    del tlist['placehold']
    if len(tlist) == 0:
        return False 
    else:                 
        return tlist 
      
#aliasing
def antialias(x, y):
    w = 2
    h = 2
    location = np.zeros((h, w, 3))
    for j in range(h):
        for i in range(w):
            subx = np.around(x + (i + np.random.random_sample())/w, 2)
            suby = np.around(y + (j + np.random.random_sample())/h, 2)
            location[j][i] = vcorner + xscale*n0*(subx/xmax) + yscale*n3*(suby/ymax)
    return(location)

out = np.zeros((ymax, xmax, 3), dtype=np.uint8)
# caustics = np.zeros((ymax, xmax, 3))

#plane parameters
plane0 = obj()
plane0.type = "plane"
plane0.center = np.array([0, -9, 0])
plane0.nrm = unit([0, 1, 0])
plane0.x = unit([1,0,0])
plane0.y = unit([0,0,1])
plane0.diffuse = np.array([255,255,255])
plane0.mat = "diffuse" #False, "diffuse", "glass", "mirror"
plane0.n = 1.5
plane0.rough = 0.9
plane0.id = "plane0"
plane0.texture = texture


for frame in range(framestart, frameend):
    plane0.center = np.array([0, -9, 0]) #add a frame dependent function to animate
    plane0.x = unit([1,0,0]) #add a frame dependent function to animate
    plane0.y = unit([0,0,1]) #add a frame dependent function to animate
    #objects to be rendered
    objects = [sphere0, sphere1, sphere2]
    print("\nProgress: 0%")
    #begins rendering for each pixel
    for j in range(ymax):
        for i in range(xmax):
            pixel = antialias(i, j) 
            #antialiasing
            for l in range(np.shape(pixel)[0]):
                for k in range(np.shape(pixel)[1]):
                    vloc = pixel[l][k] #subpixel location
                    vray = unit(vloc - eye)
                    intersections = t(eye,vray, None, objects) #finds intersections and other information, if no intersect, returns false
                    # print(info)
                    
                    if intersections is not False: #if there are intersections... uses info about the intersection (see above)
                        while len(intersections.keys()) > 0:
                            tvalue = min(intersections.keys())
                            info = intersections.pop(tvalue)
                            p = eye + tvalue*vray #surface location
                            if info.type is 'plane':
                                norm = info.normal #surface norm
                            if info.type is 'sphere':
                                norm = unit(p - info.center)
                            if np.dot(-vray,norm) < 0:
                                norm = -norm
                            projectray = unit(p - projectpoint) 
                            projectint = t(projectpoint, projectray, None, [plane0])
                            if projectint is not False:
                                projectint = projectint[min(projectint.keys())]
                                if np.mean(projectint.color) < 100:
                                    intersections.clear()
                                    matid = info.id
                                    
                                # reflect = unit(y + 2*d*tlist[2])
                                    lray = unit(light - p) #light ray, change this for directional light
                                    refl = unit(-lray + 2*np.dot(lray, norm)*norm) #reflected light ray
                                    ndotl = max(np.dot(lray, norm), 0) #dark and light interpolation
                                    roughness = info.rough
                                    thetaI = math.acos(np.dot(lray, norm))
                                    diffuse = info.color
                                    

                                    spec = s0(-vray, refl, roughness)
                                    
                                    pixel[l][k] = diffuse*(1-spec) + lcolor*spec 
                                    pixel[l][k] = ndotl*pixel[l][k] + (1-ndotl)*(pixel[l][k]/3)
                                    
                                else:   
                                    if len(list(intersections.keys())) == 0:
                                        pixel[l][k] = np.array([255, 255, 255])
                            else:
                                if len(list(intersections.keys())) == 0:
                                    pixel[l][k] = np.array([255, 255, 255])

                    else: #if no intersect, display white
                        pixel[l][k] = np.array([255, 255, 255])
            color = pixel[l][k] #np.array(np.mean(np.mean(pixel, axis=0), axis=0), dtype=np.uint8) #averages subpixel samples
            j = ymax - (j + 1) #flips to corresponding pixel
            out[j][i] = color
        #progress checker
        progress = int(ymax/5)
        if j != 0 and j%progress == 0:
            print(int(j/progress)*20, "%")

    #saves image
    fin = Image.fromarray(out, 'RGB')
    if frame < 10:
        fin.save('{}_00{}.png'.format(var_out, frame))
    elif frame < 100:
        fin.save('{}_0{}.png'.format(var_out, frame))
    else: 
        fin.save('{}_{}.png'.format(var_out, frame))
    print(frame, 'of {}: Done\n'.format(frameend))
    
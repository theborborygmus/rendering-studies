import sys
import math
import numpy as np 
from PIL import Image

#defining colors
print("\nInitializing")

def fresnel(n, thetaI):
    thetaT = math.asin(math.sin(thetaI) / n)
    RP = ((n*math.cos(thetaI) - math.cos(thetaT)) / (n*math.cos(thetaI) + math.cos(thetaT)))**2
    RS = ((math.cos(thetaI) - n*math.cos(thetaT)) / (math.cos(thetaI) + n*math.cos(thetaT)))**2
    R = (RP + RS) / 2
    return R

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

#unit vector conversion
def unit(x):
    a = np.array(x)/np.linalg.norm(np.array(x))
    return a

#setting up eye, light, viewport, etc...
up = unit([0, 1, 0]) #local up direction, change to rotate view 
#light information
light = 1.5*np.array([-9, 9, -9]) #light location \\ default: [-9, 3, -6]
lview = unit([1, -1, 1]) #light direction for spotlight/directional/etc 
l0 = unit(np.cross(up, lview)) #light right/left
l3 = unit(np.cross(lview, l0)) #light up/down
lcolor = np.array([255, 255, 255]) #use for specular color

#eye information
eye = np.array([0, 3, -21]) #eye location default: [-17, 3, -17]
view = unit([0, 0, 1.2]) #eye direction default: [1, -0.1, 1]
dist = 30
xscale = 36 #x-fov 
yscale = 27 #y-fov  
n2 = view #view direction
n0 = unit(np.cross(up, view)) #right/left
n3 = unit(np.cross(n2, n0)) #up/down
vcenter = eye + dist*n2 #viewport center
vcorner = vcenter - (xscale/2)*n0 - (yscale/2)*n3
#output size
xmax = int(400*1.2) 
ymax = int(300*1.2)

class obj:
    pass

#plane parameters
plane0 = obj()
plane0.type = "plane"
plane0.center = np.array([0, -4.8, 0])
plane0.nrm = unit([0, 1, 0])
plane0.x = unit([1,0,0])
plane0.y = unit([0,0,1])
plane0.diffuse = np.array([255,255,255])
plane0.mat = "diffuse" #False, "diffuse", "glass", "mirror"
plane0.n = 1.5
plane0.rough = 0.9
plane0.id = 0

#plane parameters
plane1 = obj()
plane1.type = "plane"
plane1.center = np.array([0, 0, 9])
plane1.nrm = unit([0, 0, -1])
plane1.x = unit([1,0,0])
plane1.y = unit([0,0,1])
plane1.diffuse = np.array([255,255,100])
plane1.mat = "diffuse" #False, "diffuse", "glass", "mirror"
plane1.n = 1.5
plane1.rough = 0.9
plane1.id = 1

#sphere parameters
sphere0 = obj()
sphere0.type = "sphere"
sphere0.center = np.array([-5.4, 0.3, -0.9])
sphere0.radius = 4.8
sphere0.diffuse = np.array([255,100,100])
sphere0.mat = "diffuse" #False, "diffuse", "glass", "mirror"
sphere0.n = 1.5
sphere0.rough = 0.9
sphere0.id = 2

#sphere parameters
sphere1 = obj()
sphere1.type = "sphere"
sphere1.center = np.array([-1.2, 6.6, 3])
sphere1.radius = 4.2
sphere1.diffuse = np.array([100,255,100])
sphere1.mat = "diffuse" #False, "diffuse", "glass", "mirror"
sphere1.n = 1.5
sphere1.rough = 0.6
sphere1.id = 3

#sphere parameters
sphere2 = obj()
sphere2.type = "sphere"
sphere2.center = np.array([3.6, -0.6, -0.3])
sphere2.radius = 3.6
sphere2.diffuse = np.array([100,100,255])
sphere2.mat = "diffuse" #False, "diffuse", "glass", "mirror"
sphere2.n = 1.5
sphere2.rough = 0.3
sphere2.id = 4

#sphere parameters
sphere3 = obj()
sphere3.type = "sphere"
sphere3.center = np.array([7.2, 4.2, 3.6])
sphere3.radius = 3
sphere3.diffuse = np.array([100,100,100])
sphere3.mat = "mirror" #False, "diffuse", "glass", "mirror"
sphere3.n = 1.5
sphere3.rough = 0
sphere3.id = 5

#objects to be rendered
objects = [plane0, plane1, sphere0, sphere1, sphere2, sphere3]

#returns information about a ray & plane intersection
def planeintersect(i, x, y):
    #arrays that will be used to store information
    plane = np.zeros((7,3))
    if np.dot(i.nrm, y) != 0:
        plane[0] = -(np.dot(i.nrm, (x - i.center)))/np.dot(i.nrm, y) #t-value
        if plane[0][0] > 0: #not zero to account for rounding
            plane[1] = x + plane[0]*y #intersection
            plane[2] = i.nrm #normal
            plane[3] = i.diffuse
            plane[4] = [0.03, i.id, 1]
            plane[5] = [0, i.n, i.rough]
            if i.mat is "diffuse":
                plane[5][0] = 0
            if i.mat is "glass":
                plane[5][0] = 1
            if i.mat is "mirror":
                plane[5][0] = 2
            plane[6] = None
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
    #arrays that will be used to store information
    sphere = np.zeros((7,3))
    if b**2 - c >= 0 and b <= 0:
        sphere[0] = min([-b - (b**2 - c)**0.5, -b + (b**2 - c)**0.5]) #t-value
        if sphere[0][0] > 0: #not zero to account for rounding
            sphere[1] = x + sphere[0]*y #intersection
            sphere[2] = unit(sphere[1] - i.center) #normal
            sphere[3] = i.diffuse
            sphere[6] = max([-b - (b**2 - c)**0.5, -b + (b**2 - c)**0.5]) #further t-value
            sphere[4] = [sphere[6][0] - sphere[0][0], i.id, 2]
            sphere[5] = [0, i.n, i.rough]
            if i.mat is "diffuse":
                sphere[5][0] = 0
            if i.mat is "glass":
                sphere[5][0] = 1
            if i.mat is "mirror":
                sphere[5][0] = 2
            sphere[6] = x + sphere[6]*y
            return sphere
        else:
            return False
    else:
        return False

#finding intersections based off x, the location, and y, the ray from x
def t(x, y, z, objectlist): 
    tlist = None #stores t values for nearest interesection
    tlistmax = None #stores t values for furthest intersection    
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
                if tlist is None:
                    tlist = i.intersect
                    tlistmax = i.intersect
                else:
                    if i.intersect[0][0] < tlist[0][0]:
                        tlist = i.intersect
                    if i.intersect[0][0] > tlistmax[0][0]:
                        tlistmax = i.intersect

    if tlist is None:
        return False 
    else:                 
        tlist[4][0] = tlistmax[4][0] + tlist[4][0]
        return tlist #[closest tvalue, point of intersection, normal, color, thickness, and material information]

def secondarypath(material, norm, view):
    #probability distribution functions:
    if material[0] == 0: #diffuse
        probDiff = 0.81 + material[2]*(1-0.81)
        probTrans = 0
        probRef = 1-probDiff
    if material[0] == 1: #glass
        probDiff = 0.01
        probTrans = 0.99 - F
        probRef = F
    if material[0] == 2: #mirror
        probDiff = 0.01
        probTrans = 0
        probRef = 0.99
    roll = np.random.random_sample()
    if roll < probTrans:
        # print("transmit")
        ptype = 1
        rayimp = probTrans
        d = np.dot(-view, norm)
        term = (d**2 - 1)/(material[1]**2) + 1
        if term > 0:
            outray = unit((view + norm*d)/material[1] - norm*abs(term)**0.5)  
        else:
            outray = view    
    elif roll < probTrans + probRef:
        # print("mirror")
        ptype = 2
        rayimp = probRef
        outray = unit(2*np.dot(-view, norm)*norm+view)
    else:
        # print("diffuse")
        ptype = 0
        rayimp = probDiff
        outray = unit(np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)]))
        outray = unit(outray + norm*1.2) 
    # print(probDiff, probTrans, probRef, ptype)    
    return (rayimp, outray, ptype)
        
#aliasing
def antialias(x, y):
    w = 3
    h = 3
    location = np.zeros((h, w, 3))
    for j in range(h):
        for i in range(w):
            subx = np.around(x + (i + np.random.random_sample())/w, 2)
            suby = np.around(y + (j + np.random.random_sample())/h, 2)
            location[j][i] = vcorner + xscale*n0*(subx/xmax) + yscale*n3*(suby/ymax)
    return(location)

out = np.zeros((ymax, xmax, 3), dtype=np.uint8)
# caustics = np.zeros((ymax, xmax, 3))

var_out = input("Output name: ")
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
                info = t(eye,vray, None, objects) #finds intersections and other information, if no intersect, returns false
                # print(info)
                if info is not False: #if there are intersections... uses info about the intersection (see above)
                    p = info[1] #surface location
                    matid = info[4][1]
                    norm = info[2] #surface norm                  
                    # reflect = unit(y + 2*d*tlist[2])
                    lray = unit(light - p) #light ray, change this for directional light
                    refl = unit(-lray + 2*np.dot(lray, norm)*norm) #reflected light ray
                    ndotl = max(np.dot(lray, norm), 0) #dark and light interpolation
                    n = info[5][1]
                    roughness = info[5][2]
                    thetaI = math.acos(np.dot(lray,norm))
                    F = fresnel(n, thetaI)
                    spec = s0(-vray, refl, info[5][2])
                    diffuse = info[3]
                    #path tracing
                    denom = 6
                    threshold = 1/denom
                    paths = np.zeros((30,3))
                    caustic = 0
                    for path in range(len(paths)):
                        paths[path] = diffuse
                        newpoint = info[1]
                        newnorm = info[2]
                        inray = vray
                        newmat = info[5]
                        importance = 1
                        paths[path] = diffuse
                        bounce = 0
                        secondpath = secondarypath(newmat, newnorm, inray)
                        caustic = 0
                        while importance > threshold:
                            
                            outray = secondpath[1]
                            newinfo = t(newpoint, outray, None, objects)
                            if newinfo is not False:
                                bounce += 1
                                if newmat[0] == 0:
                                    paths[path] = paths[path]*(1-(1/(1+newmat[2]+bounce*0.5))) + newinfo[3]*(1/(1+newmat[2]+bounce*0.5)) #for diffuse surfaces, reduce exact reflections
                                else:
                                    paths[path] = paths[path]*(1-(1/(1+newmat[2]*0.5+bounce*0.5))) + newinfo[3]*(1/(1+newmat[2]*0.5+bounce*0.5))
                                newpoint = newinfo[1]
                                newnorm = newinfo[2]
                                inray = outray
                                newmat = newinfo[5]
                                importance = importance*secondpath[0]
                                if importance <= threshold:
                                    if np.random.random_sample() > threshold:
                                        importance = 0
                                    else:
                                        importance = importance * denom
                                secondpath = secondarypath(newmat, newnorm, inray)
                                #bidirectional ray tracing
                                # if bounce == 1:
                                #     tolight = unit(light - newpoint)
                                #     test = t(newpoint, tolight, None, objects)
                                #     if test is False:
                                #         lightrefl = unit(-tolight + 2*np.dot(tolight, newnorm)*newnorm)
                                #         caustic = (np.dot(lightrefl, -inray)-0.1)*(1-newmat[2])
                                #         if caustic < 0:
                                #             caustic = 0
                                #         if caustic > 1:
                                #             causticc = 1       
                            else:
                                importance = 0 
                                #if first bounce off mirror does not intersect: color black
                                if newmat[0] == 2 and bounce == 0: 
                                    paths[path] = [0,0,0]
                        
                        paths[path] = paths[path]*(1-caustic) + lcolor*caustic

                    pixel[l][k] = np.array(np.mean(paths, axis=0))
                    pixel[l][k] = pixel[l][k]*(1-spec) + lcolor*spec  
                    # pixel[l][k] = diffuse*(1-spec) + lcolor*spec 
                    pixel[l][k] = ndotl*pixel[l][k] + (1-ndotl)*(pixel[l][k]/3)
                    
                    #shadowing
                    lighttransport = t(p, lray, matid, objects) 
                    if lighttransport is not False:
                        if np.linalg.norm(light - p) > lighttransport[0][0]:# and lighttransport[4][1] != info[4][1]:
                            shadow = lighttransport[4][0] / 4.5
                            if shadow > 1:
                                shadow = 1
                            pixel[l][k] = pixel[l][k]*(1-shadow) + (pixel[l][k]/3)*shadow

                    pixel[l][k] = pixel[l][k]*(1-caustic) + lcolor*caustic

                else: #if no intersect, display black
                    pixel[l][k] = np.array([0, 0, 0])
        color = np.array(np.mean(np.mean(pixel, axis=0), axis=0), dtype=np.uint8) #averages subpixel samples
        j = ymax - (j + 1) #flips to corresponding pixel
        out[j][i] = color
    #progress checker
    progress = int(ymax/5)
    if j != 0 and j%progress == 0:
        print(int(j/progress)*20, "%")

#saves image
fin = Image.fromarray(out, 'RGB')
fin.save('{}.png'.format(var_out))
print('100%\nDone')

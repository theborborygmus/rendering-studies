import sys
import math
import numpy as np 
from PIL import Image

#defining colors
print("\nInitializing")
shading = input("Toon Shading? [y/n]: ")

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
#toonshade specular spot
def s0toon(x, y, m):
    maxi = 1 + m**2 #more roughness => higher max
    mini = 0.6 - (min(m**3, 0.6)) #more roughness => higher min
    s = max(np.dot(x,y), 0)
    s = (s-mini)/(maxi-mini)
    if s > 0.9:
        s = 1
    elif s > 0.6:
        s = 0.9
    elif s > 0.3:
        s = 0.3
    else:
        s = 0 
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
xmax = int(400*1.8) 
ymax = int(300*1.8)

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

#tetrahedron data from .obj


#sphere parameters
sphere0 = obj()
sphere0.type = "sphere"
sphere0.center = np.array([-5.1, 2.4, -0.9])
sphere0.radius = 4.8
sphere0.diffuse = np.array([255,100,100])
sphere0.mat = "diffuse" #False, "diffuse", "glass", "mirror"
sphere0.n = 1.5
sphere0.rough = 0.9
sphere0.id = 2

#sphere parameters
sphere1 = obj()
sphere1.type = "sphere"
sphere1.center = np.array([0.6, 6.3, -1.2])
sphere1.radius = 3.6
sphere1.diffuse = np.array([100,100,255])
sphere1.mat = "diffuse" #False, "diffuse", "glass", "mirror"
sphere1.n = 1.5
sphere1.rough = 0.3
sphere1.id = 0

#tetrahedron
tetra = obj()
tetra.type = "polygon"
tetra.v = 9*np.array([
    [-0.500000, -0.750000, -0.866026],
    [-0.500000, -0.750000, 0.866025],
    [1.000000, -0.750000, 0.000000],
    [0.000000, 0.750000, 0.000000]
])
tetra.vn = np.array([
    [0.000000, -1.000000, 0.000000],
    [0.000000, -1.000000, 0.000000],
    [0.000000, -1.000000, 0.000000],
    [-0.948683, 0.316228, -0.000000],
    [-0.948683, 0.316228, -0.000000],
    [-0.948683, 0.316228, -0.000000],
    [0.474342, 0.316228, 0.821584],
    [0.474342, 0.316228, 0.821584],
    [0.474342, 0.316228, 0.821584],
    [0.474342, 0.316228, -0.821584],
    [0.474342, 0.316228, -0.821584],
    [0.474342, 0.316228, -0.821584]
])
tetra.f = np.array([
    [[1,1], [3,2], [2,3]],
    [[1,4], [2,5], [4,6]],
    [[2,7], [3,8], [4,9]],
    [[3,10], [1,11], [4,12]]
])
tetra.diffuse = np.array([100,255,100])
tetra.n = 1.5
tetra.rough = 0.6
tetra.id = 3
#tramsformations
for vertex in range(len(tetra.v)):
    tetra.v[vertex] = roty(np.pi/2, tetra.v[vertex])
for vertnorm in range(len(tetra.vn)):
    tetra.vn[vertnorm] = roty(np.pi/2, tetra.vn[vertnorm])
tetra.v = tetra.v + [6.3,3.3,3.6]
tetra.A = A(tetra)

#objects to be rendered
objects = [tetra, sphere0,sphere1]

#determine intersection information for each face of each polygon
def triintersect(i, x, y):
    tplane = [None, None] 
    tplanemax = [None, None] 
    h = len(i.f)
    plane = np.zeros((h,7,3)) #to store values: [t-value, intersection, normal, color, furthest intersection, thickness, uv]
    for j in range(len(i.f)):
        plnnrm = unit(i.A[j])
        
        if np.dot(i.A[j], y) != 0:
            plncntr = i.v[i.f[j][0][0]-1]
            plane[j][0] += -(np.dot(plnnrm, (x - plncntr)))/np.dot(plnnrm, y) #t-value
            plane[j][1] = x + plane[j][0]*y #intersection
            
            #calculations to check if intersection is on face
            p0 = i.v[i.f[j][0][0]-1]
            p1 = i.v[i.f[j][1][0]-1]
            p2 = i.v[i.f[j][2][0]-1]
            ph = plane[j][1]
            A0 = np.cross(ph - p1, ph - p2)
            A1 = np.cross(ph - p2, ph - p0)
            A2 = np.cross(ph - p0, ph - p1)
            #uv coordinates for face
            u = sum(A1)/sum(i.A[j])
            v = sum(A2)/sum(i.A[j])
            uv = sum(A0)/sum(i.A[j])
            #checking if intersection is on face of polygon
            if 0 <= uv <= 1  and 0 <= u <= 1 and 0 <= v <= 1:
                n0 = i.f[j][0][1]-1
                n1 = i.f[j][1][1]-1
                n2 = i.f[j][2][1]-1
                plane[j][2] = unit(np.mean([i.vn[n0], i.vn[n1], i.vn[n2]], axis = 0))#normal
                if tplanemax[0] is None:
                    if plane[j][0][0] > 0:
                        tplanemax = [plane[j][0][0], j]
                else:
                    if plane[j][0][0] > tplanemax[0]: 
                        tplanemax = [plane[j][0][0], j] 
                
                if tplane[0] is None:
                    if plane[j][0][0] > 0:
                        tplane = [plane[j][0][0], j] 
                else:
                    if plane[j][0][0] < tplane[0]:
                        tplane = [plane[j][0][0], j] 
                #texturing
                plane[j][3] = i.diffuse
            else:
                pass
    if tplane[0] is None:
        return False
    else:
        #returns the intersection information of the closest face
        plane[tplane[1]][6] = tplanemax[0] #furthest intersection
        plane[tplane[1]][4] = [tplanemax[0] - tplane[0], i.id, 3] #thickness
        plane[tplane[1]][5] = [0, i.n, i.rough]
        return plane[tplane[1]] #[t value, intersection, normal, diffuse,[thickness, id, object type], [0, i.n, i.rough], furthest intersection ]

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
            if i.type is "polygon":
                i.intersect = triintersect(i, x, y)

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
normout = np.zeros((ymax, xmax, 3))
# caustics = np.zeros((ymax, xmax, 3))

var_out = input("Output name: ")
print("\nProgress: 0%")
#begins rendering for each pixel
for j in range(ymax):
    for i in range(xmax):
        pixel = antialias(i, j) 
        normsums = np.array([0,0,0])
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
                    normsums = normsums + norm                 
                    # reflect = unit(y + 2*d*tlist[2])
                    lray = unit(light - p) #light ray, change this for directional light
                    refl = unit(-lray + 2*np.dot(lray, norm)*norm) #reflected light ray
                    ndotl = max(np.dot(lray, norm), 0) #dark and light interpolation
                    n = info[5][1]
                    roughness = info[5][2]
                    thetaI = math.acos(np.dot(lray,norm))
                    diffuse = info[3]
                    
                    #toonshading n dot l
                    if shading == 'y':
                        if ndotl < 0.0006:
                            ndotl = 0
                        elif ndotl < 0.09:
                            ndotl = 0.5
                        else:
                            ndotl = 1
                        spec = s0toon(-vray, refl, info[5][2])
                    #notoonshading
                    else:
                        spec = s0(-vray, refl, info[5][2])
                    
                    pixel[l][k] = diffuse*(1-spec) + lcolor*spec 
                    pixel[l][k] = ndotl*pixel[l][k] + (1-ndotl)*(pixel[l][k]/3)
                    
                    #shadowing
                    lighttransport = t(p, lray, matid, objects) 
                    if lighttransport is not False:
                        if np.linalg.norm(light - p) > lighttransport[0][0]:
                            pixel[l][k] = pixel[l][k]/3
                else: #if no intersect, display black
                    pixel[l][k] = np.array([255, 255, 255])
                    normsums = normsums + np.array([0,0,-2])
        color = np.array(np.mean(np.mean(pixel, axis=0), axis=0), dtype=np.uint8) #averages subpixel samples
               
        j = ymax - (j + 1) #flips to corresponding pixel
        normout[j][i] = unit(normsums) 
        out[j][i] = color
    #progress checker
    progress = int(ymax/5)
    if j != 0 and j%progress == 0:
        print(int(j/progress)*20, "%")


#outlining edges
if shading == 'y':
    print('Finalizing')
    for j in range(ymax):
        for i in range(xmax):
            grid = []
            for x in [-1,0,1]:
                for y in [-1,1]:
                    if i+x in range(xmax) and j+y in range(ymax):
                        grid.append(normout[j+y][i+x])
            for x in [-1,1]:
                for y in [0]:
                    if i+x in range(xmax) and j+y in range(ymax):
                        grid.append(normout[j+y][i+x])
            grid = np.array(grid)
            original = out[j][i]
            dotlist = []   
            for pixels in grid:
                dot = np.dot(normout[j][i], pixels)
                dotlist.append(dot)
            
            dotlist = min(dotlist)
            border = (0.9 - dotlist) / (0.9 - 0.6)
            if border > 1:
                border = 1
            elif border > 0.6:
                border = 0.6
            elif border > 0.3:
                border = 0.3
            else:
                border = 0
            out[j][i] = original*(1-border) + np.array([0,0,0])*border
        #progress checker
        progress = int(ymax/5)
        if j != 0 and j%progress == 0:
            print(int(j/progress)*20, "%")



#saves image
fin = Image.fromarray(out, 'RGB')
fin.save('{}.png'.format(var_out))
print('100%\nDone')

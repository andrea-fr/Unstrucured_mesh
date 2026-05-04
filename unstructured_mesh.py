import numpy as np 
from scipy.spatial import Delaunay
#from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import random
import math
import os
import pandas as pd

#Rectangular geometry with the bottom-left corner centered in a cartesian coordinate system. Creation of nodes with Poisson disk sampling. 
#Generation of prism layers on the lower side and generation of the core mesh using Delaunay triangulation.
def get_cell_coords(point):
    #return index of a point
    return int(point[0] / size_check_points), int((point[1]-current_height) / size_check_points)

def is_valid(point, points):
    x, y = point
    #inside domain?
    if x < 0 or x > lx or y < current_height or y > ly:
        return False
    gx, gy = get_cell_coords(point)
    # check points on the dummy grid    
    for i in range(max(0, gy-2), min(ny_grid, gy+3)):
        for j in range(max(0, gx-2), min(nx_grid, gx+3)):
            idx = grid[i][j]
            if idx is not None:
                px,py=points[idx-1]
                dx = px - x
                dy = py - y
                if math.sqrt(dx*dx + dy*dy) < r:
                    return False
    return True

def triangle_area(p0, p1, p2):
    return 0.5 * (
        p0[0]*(p1[1] - p2[1]) +
        p1[0]*(p2[1] - p0[1]) +
        p2[0]*(p0[1] - p1[1])
    )

def ensure_ccw_tria(nodes):
    p0, p1, p2 = global_points[nodes]
    A = triangle_area(p0, p1, p2)
    if A < 0:
        return [nodes[0], nodes[2], nodes[1]]
    return nodes

def edge_normal_outward(p1, p2, centroid):
    edge_vec = p2 - p1 #edge vector
    normal = np.array([edge_vec[1], -edge_vec[0]])
    length = np.linalg.norm(edge_vec)
    normal = normal / np.linalg.norm(normal)
    midpoint = 0.5 * (p1 + p2)
    if np.dot(midpoint - centroid, normal) < 0:
        normal = -normal
    return normal, length, midpoint

def laplacian_smoothing(points, n_boundary, alpha=0.4, n_iter=5):
    neighbors = {i: set() for i in range(len(points))}
    for simplex in simplices:
        for k in range(3):
            a = simplex[k]
            b = simplex[(k+1) % 3]
            neighbors[a].add(b)
            neighbors[b].add(a)

    for _ in range(n_iter):
        new_points = points.copy()
        for i in range(n_boundary, len(points)):
            neigh = list(neighbors[i])
            if len(neigh) == 0:
                continue
            bary = np.mean(points[neigh], axis=0)
            new_points[i] = (1 - alpha) * points[i] + alpha * bary
        points = new_points
    return points

lx=2
ly=1
r=0.05 #radius for poisson disk sampling
nx=int(lx/r)+1
ny=int(ly/r)+1
x_bottom=np.linspace(0,lx,nx)
y_bottom=np.zeros(nx)

##############################
# PRISM LAYERS
##############################
n_layers = 22
h0 = 0.001
growth_rate_PL = 1.2
ny=int((ly-(h0*(1-growth_rate_PL**n_layers)/(1-growth_rate_PL)))/r)+1

layer_lines = [np.vstack((x_bottom, y_bottom)).T] 

current_height = 0.0
dh = h0
for i in range(n_layers):
    current_height += dh
    x_layer = x_bottom.copy()
    y_layer = np.ones_like(x_layer) * current_height
    layer_lines.append(np.vstack((x_layer, y_layer)).T)
    dh *= growth_rate_PL        


PL_points = []
for layer in layer_lines:
    for p in layer:
        PL_points.append(p.tolist())


Nx = len(x_bottom)
quad_cells = []
for k in range(n_layers):          # between layer k and k+1
    for i in range(Nx - 1):
        
        n0 = k * Nx + i
        n1 = k * Nx + i + 1
        n2 = (k + 1) * Nx + i + 1
        n3 = (k + 1) * Nx + i
        
        quad_cells.append([n0, n1, n2, n3])

##############################
# BOUNDARY POINTS
##############################
y_left=np.linspace(ly,current_height,ny)
x_left=np.zeros(len(y_left))
x_up=np.linspace(lx,0,nx) 
y_up=np.ones(nx)*ly
y_right=np.linspace(current_height,ly,ny)
x_right=np.ones(len(y_left))*lx

boundary_points=np.array([
    np.concatenate((layer_lines[-1][:,0],x_right,x_up,x_left)),
    np.concatenate((layer_lines[-1][:,1],y_right,y_up,y_left)),]).T


_,idx_unique=np.unique(boundary_points,axis=0,return_index=True)
boundary_nodes=boundary_points[np.sort(idx_unique)]

##############################
# POISSON DISK
##############################
#To make faster the algorithm, a check has be done on a smaller portion of domain
size_check_points=r/math.sqrt(2)
nx_grid=int(lx/size_check_points)+1
ny_grid=int((ly-current_height)/size_check_points)+1

grid = [[None for _ in range(nx_grid)] for _ in range(ny_grid)] 
points = []
active_list = [] #list of point that are able to generate other points

for p in boundary_points: 
    x,y=p
    gx,gy=get_cell_coords([x,y])    
    grid[gy][gx]=len(points)
    points.append([x,y])     

first_point = (random.uniform(0, lx), random.uniform(0, ly))
while is_valid(first_point,points)==False:
    first_point = (random.uniform(0, lx), random.uniform(0, ly))

points.append(first_point)
active_list.append(first_point)
gx, gy = get_cell_coords(first_point)
grid[gy][gx] = len(points)


k=30 #itearations to generate new points
while active_list:
    # choose a random active point
    point = random.choice(active_list)
    found = False
    for _ in range(k):
        radius = random.uniform(r, 2*r)
        angle = random.uniform(0, 2*math.pi)
        new_x = point[0] + radius * math.cos(angle)
        new_y = point[1] + radius * math.sin(angle)
        new_point = (new_x, new_y)
        if is_valid(new_point, points):
            points.append(new_point)
            active_list.append(new_point)
            gx, gy = get_cell_coords(new_point)
            grid[gy][gx] = len(points)
            found = True
            break
    # if no valid point found, remove active point
    if not found:
        active_list.remove(point)

poisson_points = np.array(points)


##############################
# GLOBAL NODES (tria+quad)
##############################
PL_points = np.array(PL_points)
global_points = np.vstack((PL_points, poisson_points))
n_PL = len(PL_points)


#####################################
# MESH FEATURES 
####################################
cells = []

# 1) Prism layer cells (quad)
for idx,quad in enumerate(quad_cells):
    # from spatial coordinates to global index
    quad_np = global_points[quad] #prendo le coordinate dei 4 vertici
    x, y = quad_np[:,0], quad_np[:,1]
    area = 0.5*np.abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))
    cx = np.sum((x+np.roll(x,-1))*(x*np.roll(y,-1)-np.roll(x,-1)*y))/(6*area)
    cy = np.sum((y+np.roll(y,-1))*(x*np.roll(y,-1)-np.roll(x,-1)*y))/(6*area)
    cells.append({
        "id": len(cells),
        "nodes": quad,
        "centroid": np.array([cx, cy]),
        "area": area,
        "faces": [],
        "type": "prism"
    })

# 2) Tria cells
tri = Delaunay(poisson_points)
simplices = tri.simplices
n_boundary = len(boundary_points)

poisson_points = laplacian_smoothing(
    poisson_points,
    n_boundary,
    alpha=0.2,
    n_iter=30
)

tri = Delaunay(poisson_points)
simplices = tri.simplices
global_points = np.vstack((PL_points, poisson_points))

for simplex in simplices:
    nodes_idx = [n_PL + idx for idx in simplex]
    nodes_idx = ensure_ccw_tria(nodes_idx)
    p0, p1, p2 = global_points[nodes_idx]
    area = abs(triangle_area(p0,p1,p2))
    centroid = (p0 + p1 + p2)/3.0
    cells.append({
        "id": len(cells),
        "nodes": nodes_idx,
        "centroid": centroid,
        "area": area,
        "faces": [],
        "type": "tria"
    })



edges = {}
for cell_id, cell in enumerate(cells):
    nodes = cell["nodes"]
    n_nodes = len(nodes)
    for i in range(n_nodes):
        n1 = nodes[i]
        n2 = nodes[(i+1)%n_nodes]
        edge = tuple(sorted((n1,n2)))
        if edge not in edges:
            edges[edge] = [cell_id]
        else:
            edges[edge].append(cell_id)


# ==========================================================
# FACES + NEIGHBORS
# ==========================================================

for edge_nodes, attached_cells in edges.items():
    n1, n2 = edge_nodes
    p1, p2 = global_points[[n1,n2]]
    if len(attached_cells) == 2:
        c1, c2 = attached_cells
        internal = True
    else:
        c1 = attached_cells[0]
        c2 = None
        internal = False
    centroid1 = cells[c1]["centroid"]
    normal1, length, midpoint = edge_normal_outward(p1,p2,centroid1)
    d = np.linalg.norm(cells[c2]["centroid"] - centroid1) if internal else None
    face1 = {
        "nodes": (n1,n2),
        "neighbor": c2,
        "normal": normal1,
        "length": length,
        "center": midpoint,
        "distance": d
    }
    cells[c1]["faces"].append(face1)
    if internal:
        centroid2 = cells[c2]["centroid"]
        normal2, _, _ = edge_normal_outward(p1,p2,centroid2)
        face2 = {
            "nodes": (n1,n2),
            "neighbor": c1,
            "normal": normal2,
            "length": length,
            "center": midpoint,
            "distance": d
        }
        cells[c2]["faces"].append(face2)

# ==========================================================
# CSV
# ==========================================================

cells_data = []
for cell in cells:
    nodes = cell["nodes"]
    n_nodes = len(nodes)
    cx, cy = cell["centroid"]
    
    # If cell has less than 4 nodes, list the fourth as None
    nodes_padded = list(nodes) + [None]*(4-n_nodes)
    
    cells_data.append([
        cell["id"],
        *nodes_padded,
        cell["area"],
        cx, cy,
        len(cell["faces"]),
        cell["type"]
    ])

df_cells = pd.DataFrame(cells_data, columns=[
    "cell_id",
    "node1", "node2", "node3", "node4",
    "area",
    "centroid_x", "centroid_y",
    "num_faces",
    "type"
])

edges_data = []
for edge_nodes, attached_cells in edges.items():
    n1, n2 = edge_nodes
    if len(attached_cells) == 2:
        c1, c2 = attached_cells
    else:
        c1, c2 = attached_cells[0], None
    edges_data.append([n1, n2, c1, c2])

df_edges = pd.DataFrame(edges_data, columns=[
    "node1", "node2",
    "cell1", "cell2"
])

faces_data = []
for cell in cells:
    for face in cell["faces"]:
        n1, n2 = face["nodes"]
        nx, ny = face["normal"]
        cx, cy = face["center"]
        faces_data.append([
            cell["id"],
            n1, n2,
            face["neighbor"],
            nx, ny,
            face["length"],
            cx, cy,
            face["distance"]
        ])

df_faces = pd.DataFrame(faces_data, columns=[
    "cell_id",
    "node1", "node2",
    "neighbor",
    "normal_x", "normal_y",
    "length",
    "center_x", "center_y",
    "distance"
])


file_path = os.path.dirname(os.path.abspath(__file__))
excel_file = os.path.join(file_path, "mesh_data.xlsx")

with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
    df_cells.to_excel(writer, sheet_name="Cells", index=False)
    df_edges.to_excel(writer, sheet_name="Edges", index=False)
    df_faces.to_excel(writer, sheet_name="Faces", index=False)


# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(8,6))
ax = plt.gca()
ax.set_aspect('equal')

patches_list = []
colors = []

for cell in cells:
    nodes = cell["nodes"]
    coords = np.array([global_points[n] for n in nodes if n is not None])
    polygon = patches.Polygon(coords, closed=True)
    patches_list.append(polygon)
    
    if cell["type"] == "prism":
        colors.append("orange")
    else:
        colors.append("skyblue")

pc = PatchCollection(patches_list, facecolor=colors, edgecolor='k', linewidths=0.5, alpha=0.5)
ax.add_collection(pc)

# print IDs at the centroid of cell
"""for cell in cells:
    cx, cy = cell["centroid"]
    plt.text(cx, cy, str(cell["id"]), fontsize=6, ha='center', va='center', color='red')"""

plt.scatter(global_points[:,0], global_points[:,1], s=2, color='black')  # nodi globali
plt.title("Mesh")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()



def edge_lengths(coords):
    return [np.linalg.norm(coords[i] - coords[(i+1)%len(coords)]) for i in range(len(coords))]

def triangle_angles(coords):
    a = np.linalg.norm(coords[1]-coords[0])
    b = np.linalg.norm(coords[2]-coords[1])
    c = np.linalg.norm(coords[0]-coords[2])
    angles = [
        np.degrees(np.arccos((b**2 + c**2 - a**2)/(2*b*c))),
        np.degrees(np.arccos((a**2 + c**2 - b**2)/(2*a*c))),
        np.degrees(np.arccos((a**2 + b**2 - c**2)/(2*a*b)))
    ]
    return angles

def quadrilateral_angles(coords):
    angles = triangle_angles([coords[0], coords[1], coords[2]]) + triangle_angles([coords[0], coords[2], coords[3]])
    return angles


AR_vals = []
Skew_vals = []
cell_coords = []

for cell in cells:
    nodes = cell["nodes"]
    coords = np.array([global_points[n] for n in nodes if n is not None])
    cell_coords.append(coords)
    # Aspect ratio
    lengths = edge_lengths(coords)
    ar = max(lengths)/min(lengths)
    AR_vals.append(ar)
    # Skewness
    if cell["type"] == "tria":
        angles = triangle_angles(coords)
        skew = max([abs(a-60)/60 for a in angles])
    else:
        angles = quadrilateral_angles(coords)
        skew = max([abs(a-90)/90 for a in angles])
    Skew_vals.append(skew)

# =================================================
# Plot Aspect Ratio
# =================================================
plt.figure(figsize=(8,6))
ax = plt.gca()
ax.set_aspect('equal')
for i, coords in enumerate(cell_coords):
    polygon = patches.Polygon(coords, closed=True, facecolor=plt.cm.viridis(AR_vals[i]/max(AR_vals)), edgecolor='k', linewidth=0.5)
    ax.add_patch(polygon)
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(AR_vals), vmax=max(AR_vals))), ax=ax, label='Aspect Ratio')
plt.scatter(global_points[:,0], global_points[:,1], s=2, color='black')
plt.title("Aspect Ratio with smoothing")
plt.show()

# =================================================
# Plot Skewness
# =================================================
plt.figure(figsize=(8,6))
ax = plt.gca()
ax.set_aspect('equal')
for i, coords in enumerate(cell_coords):
    polygon = patches.Polygon(coords, closed=True, facecolor=plt.cm.plasma(Skew_vals[i]/max(Skew_vals)), edgecolor='k', linewidth=0.5)
    ax.add_patch(polygon)
plt.colorbar(plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=min(Skew_vals), vmax=max(Skew_vals))), ax=ax, label='Skewness')
plt.scatter(global_points[:,0], global_points[:,1], s=2, color='black')
plt.title("Skewness with smoothing")
plt.show()

np.savez("mesh_data.npz",
        points=global_points,
        cells=cells,
        edges=edges)
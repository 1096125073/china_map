import geopandas as gpd
from utils.util import read_pickle
import matplotlib.pyplot as plt
import shapely.geometry

from utils.util import write_pickle
from collections import defaultdict
# shp = gpd.read_file("shp/city/city.shp")
# shp['representative_point'] = shp.representative_point()
# print(shp.columns)
#
# example_data = {}
#
# for row in shp.values:
#     if row[0] and row[2]:
#         if example_data.get(row[0]) is None:
#             example_data[row[0]] = {}
#         if example_data.get(row[0]).get(row[2]) is None:
#             example_data[row[0]][row[2]] = {}
#             example_data[row[0]][row[2]]['geometry'] = []
#             example_data[row[0]][row[2]]['representative_point'] = []
#         example_data[row[0]][row[2]]['geometry'].append(row[-2])
#         example_data[row[0]][row[2]]['representative_point'].append(row[-1])
# print(example_data['四川省'])
# write_pickle(example_data,'shp/city/city.bin')


# shp = gpd.read_file("shp/nine/nine.shp")
# province = read_pickle("shp/province/province.bin")
# geometries = []
# for key,value in province.items():
#     for geo in [value['geometry']]:
#         if isinstance(geo,shapely.geometry.Polygon):
#             geometries.append(geo)
#         elif isinstance(geo,shapely.geometry.MultiPolygon):
#             for g in geo:
#                 geometries.append(g)
#
# example_data = shapely.geometry.MultiPolygon(geometries)
# write_pickle(example_data,"shp/china.bin")
# gpd.GeoSeries([example_data]).plot()
# plt.show()
# shp = gpd.read_file("shp/nine/nine.shp")
# example_data = [row[-1] for row in shp.values]
# from shapely.ops import unary_union
# union = unary_union(example_data)
# write_pickle(union,"shp/nine/nine.bin")

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.patches as mpatches
# import matplotlib.path as mpath
#
# # some arbitrary example_data to plot
# xx, yy = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-10, 10, 20), copy=False)
# zz = np.sqrt(xx ** 2 + yy ** 2)
#
# poly_verts = [
#     (0, 0),
#     (2.5, 0),
#     (2.5, 2.5),
#     (0, 2.5),
#     (0, 0)
# ]
# poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 2) * [mpath.Path.LINETO] + [mpath.Path.CLOSEPOLY]
#
# # create a Path from the polygon vertices
# path = mpath.Path(poly_verts, poly_codes)
#
# # create a Patch from the path
#
# poly_verts2 = [
#     (2.5, 2.5),
#     (5.0, 2.5),
#     (5.0, 5.0),
#     (2.5, 5.0),
#     (2.5, 2.5)
# ]
# poly_codes2 = [mpath.Path.MOVETO] + (len(poly_verts2) - 2) * [mpath.Path.LINETO] + [mpath.Path.CLOSEPOLY]
#
# # create a Path from the polygon vertices
# path2 = mpath.Path(poly_verts2, poly_codes)
#
#
# plt.figure()
# ax = plt.gca()
# patch = mpatches.PathPatch(path, facecolor='none', edgecolor='k',transform= ax.transData)
# cont = plt.imshow(xx)
# patch2 = mpatches.PathPatch(path2, facecolor='none', edgecolor='k',transform= ax.transData)
# # add the patch to the axes
# from matplotlib.collections import PatchCollection
# p = []
# c = []
# for v in path.vertices:
#     p.append(v)
# for v in path2.vertices:
#     p.append(v)
# for v in path.codes:
#     c.append(v)
# for v in path.codes:
#     c.append(v)
#
# totol = mpath.Path(p,c)
# totol = mpatches.PathPatch(totol,facecolor='none',edgecolor='k',transform= ax.transData)
# cont.set_clip_path(totol)  ## TRY COMMENTING THIS OUT
#
# plt.show()


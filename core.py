from typing import List, Dict, Tuple, Optional, Union
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.image
import matplotlib.path as mpath
from matplotlib import rcParams
from matplotlib.patches import PathPatch
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import functools
import shapely.geometry

rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def _geometry2vc(geometries: Union[shapely.geometry.base.BaseGeometry,
                                   List[shapely.geometry.base.BaseGeometry]]
                 ) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    内部使用函数，提取传入的geometry的顶点和code，具体看matplotlib.path,在imshow的时候需要扣取感兴趣的区域
    :param geometries:
    :return:
    """
    v, c = [], []
    if isinstance(geometries, shapely.geometry.Polygon) and not isinstance(geometries, list):
        geometries = [geometries]
    for geometry in geometries:
        if isinstance(geometry, shapely.geometry.Polygon):
            geo_x, geo_y = geometry.exterior.xy
            vertices = np.stack([geo_x, geo_y], axis=-1)
            codes = [mpath.Path.MOVETO] + (len(vertices) - 2) * [mpath.Path.LINETO] + [mpath.Path.CLOSEPOLY]
            v.extend(vertices)
            c.extend(codes)
        elif isinstance(geometry, shapely.geometry.MultiPolygon):
            t_v, t_c = _geometry2vc(geometry)
            v.extend(t_v)
            c.extend(t_c)
    return v, c


def clip_boundary(ax: plt.Axes, im: matplotlib.image.AxesImage,
                  border_geometries: Union[shapely.geometry.base.BaseGeometry,
                                           List[shapely.geometry.base.BaseGeometry]],
                  proj: ccrs.CRS) -> None:
    """
    根据fun:_geometry2vc 计算的顶点和codes，裁减图像
    :param ax:坐标轴
    :param im:ax.imshow函数返回的值
    :param border_geometries:在图像种，想保留的区域的几何轮廓
    :param proj:投影类
    :return:
    """
    v, c = _geometry2vc(border_geometries)
    path = mpath.Path(v, c)
    patch = PathPatch(path, facecolor='none', transform=proj._as_mpl_transform(ax))
    im.set_clip_path(patch)


def add_rectangular_grid_line(ax: plt.Axes,
                              x_min: float, y_min: float, x_max: float, y_max: float,
                              x_n_ticks: int, y_n_ticks: int, proj: ccrs.CRS) -> None:
    """
    为图像添加经纬度坐标，仅支撑投影坐标系为矩形
    :param ax:子图句柄
    :param x_min:最小经度
    :param y_min:最小维度
    :param x_max:最大经度
    :param y_max:最大纬度
    :param x_n_ticks:经度要分割成几份
    :param y_n_ticks:维度要分割成几份
    :param proj:投影类
    :return:None
    """
    assert x_min < x_max, "x_min must lower than x_max"
    x_ticks = np.round(np.linspace(x_min, x_max, x_n_ticks), 2)
    assert y_min < y_max, "y_min must lower than y_max"
    y_ticks = np.round(np.linspace(y_min, y_max, y_n_ticks), 2)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.set_xticks(x_ticks, crs=proj)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.set_yticks(y_ticks, crs=proj)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.gridlines(crs=proj, xlocs=x_ticks, ylocs=y_ticks)


def add_ball_grid_line(ax: plt.Axes,
                       x_min: float, y_min: float, x_max: float, y_max: float,
                       x_n_ticks: int, y_n_ticks: int) -> None:
    """同fun: add_rectangular_grid_line,只不过这个是用于柱形投影坐标系使用，比如兰伯特投影
    :param ax:子图句柄
    :param x_min:最小经度
    :param y_min:最小维度
    :param x_max:最大经度
    :param y_max:最大纬度
    :param x_n_ticks:经度要分割成几份
    :param y_n_ticks:维度要分割成几份
    :return:None
    """
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = None
    assert x_min < x_max, "x_min must lower than x_max"
    x_ticks = np.round(np.linspace(x_min, x_max, x_n_ticks), 2)
    assert y_min < y_max, "y_min must lower than y_max"
    y_ticks = np.round(np.linspace(y_min, y_max, y_n_ticks), 2)
    gl.xlocator = mticker.FixedLocator(x_ticks)
    gl.ylocator = mticker.FixedLocator(y_ticks)
    gl.ylabel_style = {'size': 7, 'color': 'k'}
    gl.xlabel_style = {'size': 7, 'color': 'k'}
    gl.rotate_labels = False


def add_level_text(ax: plt.Axes, level2text: Dict[int, str],
                   level2color: Dict[int, str], legend_loc: str,
                   legend_title: Optional[str] = None) -> None:
    """

    :param ax:图句柄
    :param level2text:等级到文字的映射，比如{1:"一般",2:"严重"}
    :param level2color:等级到颜色的映射，比如{1:"blue",2:"yellow"}
    :param legend_loc:图例的位置，比如"lower left"
    :param legend_title:图例的名字
    :return:None
    """
    legend_artists = [Line2D([0], [0], color='white', linewidth=3)]
    legend_texts = ['未知']
    for level, text in level2text.items():
        legend_artists.append(Line2D([0], [0], color=level2color[level], linewidth=3))
        legend_texts.append(text)
        legend = ax.legend(legend_artists, legend_texts, fancybox=False, loc=legend_loc,
                           framealpha=0.5, title=legend_title)
        legend.legendPatch.set_facecolor("grey")


def add_tag(ax: plt.Axes, texts: List[Tuple[str, shapely.geometry.Point]],
            proj: ccrs.CRS, text_size: int, text_color='m') -> None:
    """
    为每个子geometry添加文字说明，比如省份名称
    :param text_color: 文字颜色
    :param ax: 图句柄
    :param texts: 文字到点的数据对，比如[(Point,'a'),(Point,'b')]
    :param proj: 投影类
    :param text_size: 文字大小
    :return:
    """
    for text, point in texts:
        ax.text(point.x, point.y, text, transform=proj, size=text_size, color=text_color, horizontalalignment='center',
                verticalalignment='center')


def continue_bar_fun_wrapper(fun):
    """
    装饰器函数
    fun: plot_roi_contourf 与 fun: plot_roi_image 的装饰器，因为这两者有很多步骤都是重复的
    :param fun:
    :return:
    """

    @functools.wraps(fun)
    def wrapper(x: List[float], y: List[float], value: np.ndarray, proj: ccrs.CRS = ccrs.PlateCarree(),
                outer_geometries: Union[shapely.geometry.base.BaseGeometry,
                                        List[shapely.geometry.base.BaseGeometry]] = None,
                inner_geometries: Union[shapely.geometry.base.BaseGeometry,
                                        List[shapely.geometry.base.BaseGeometry]] = None,
                cmap: str = "RdBu_r", crop: bool = False, title: Optional[str] = None,
                bar_label: Optional[str] = None, add_grid_line: bool = True,
                save_path: Optional[str] = None,
                texts: Optional[List[Tuple[str, shapely.geometry.Point]]] = None,
                text_size: int = 7, text_color: str = 'm'):
        """
        这些参数才是fun:plot_roi_contourf 和 fun: plot_roi_image 真正的参数列表
        :param x: 经度范围，从小到大排列，最少需要两个经度
        :param y: 维度范围，从小到大排列，最少需要两个维度
        :param value: np.ndarray 一个二维数组，与上述坐标对应，也就是value[0][0]为地图的左下角处的值
        :param proj: 投影类
        :param outer_geometries: 外部轮廓，比如一个省的边界信息
        :param inner_geometries: 内部轮廓，比如一个省里有很多个市
        :param cmap: matplotlib 色彩，具体查看 https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
        :param crop:是否裁减，如果为True，则只会保留外部轮廓里面的数据，比如从全国降水数据中只提取四川的部分
        :param title:子图标题，默认为None
        :param bar_label:图例标题
        :param add_grid_line:是否加经纬度线
        :param save_path:若不为None，则会保存图片到指定路径
        :param texts:要添加的文字，比如[(Point,'a'),(Point,'b')]
        :param text_size:文字大小
        :param text_color:文字颜色
        :return:装饰过的函数
        """
        x_max, x_min = np.max(x), np.min(x)
        y_max, y_min = np.max(y), np.min(y)
        fig = plt.figure()
        ax = plt.axes([0.2, 0.2, 0.6, 0.6], projection=proj)
        if title:
            ax.set_title(title)
        ax.set_extent([x_min, x_max, y_min, y_max], crs=proj)
        ca = fun(ax, x, y, value, proj, cmap)
        position = fig.add_axes([0.2, 0.1, 0.6, 0.02])
        plt.colorbar(mappable=ca, ax=ax, cax=position, orientation='horizontal',
                     shrink=0.2, label=bar_label, pad=0.1)
        if outer_geometries is not None:
            if not isinstance(outer_geometries, list):
                outer_geometries = [outer_geometries]
            ax.add_geometries(outer_geometries, proj, facecolor='none', edgecolor='k', linewidth=0.5)
            if crop:
                clip_boundary(ax, ca, outer_geometries, proj)
        if inner_geometries is not None:
            if not isinstance(inner_geometries, list):
                inner_geometries = [inner_geometries]
            ax.add_geometries(inner_geometries, proj, facecolor='none', edgecolor='k', linewidth=0.5)
        if add_grid_line:
            n_ticks = 6
            add_rectangular_grid_line(ax, x_min, y_min, x_max, y_max, n_ticks, n_ticks, proj)
        if texts is not None:
            add_tag(ax, texts, proj, text_size, text_color)
        if save_path:
            plt.savefig(save_path)
        plt.show()
    return wrapper


@continue_bar_fun_wrapper
def plot_roi_contourf(ax, x, y, value, proj, cmap):
    """
    画等高线图，参数见 fun: continue_bar_fun_wrapper
    :param ax:
    :param x:
    :param y:
    :param value:
    :param proj:
    :param cmap:
    :return:
    """
    ca = ax.contourf(x, y, value, transform=proj, cmap=cmap)
    return ca


@continue_bar_fun_wrapper
def plot_roi_image(ax, x, y, value, proj, cmap):
    """
    画一般图像，参数见 fun: continue_bar_fun_wrapper
    :param ax:
    :param x:
    :param y:
    :param value:
    :param proj:
    :param cmap:
    :return:
    """
    x_max, x_min = np.max(x), np.min(x)
    y_max, y_min = np.max(y), np.min(y)
    ca = ax.imshow(value, extent=[x_min, x_max, y_max, y_min], transform=proj, interpolation='bilinear', cmap=cmap)
    return ca


def plot_roi_level(x: List[float], y: List[float], value: List[Tuple[shapely.geometry.Point, int]],
                   proj: ccrs.CRS, inner_geometries: Union[shapely.geometry.base.BaseGeometry,
                                                           List[shapely.geometry.base.BaseGeometry]],
                   level2color: Dict[int, str], level2text: Optional[Dict[int, str]] = None,
                   outer_geometries: Union[shapely.geometry.base.BaseGeometry,
                                           List[shapely.geometry.base.BaseGeometry]] = None,
                   title: str = None, add_grid_line: bool = True, legend_title: Optional[str] = None,
                   legend_loc: Optional[str] = 'lower right',
                   save_path: Optional[str] = None, texts: Optional[List[Tuple[str, shapely.geometry.Point]]] = None,
                   text_size: int = 7, text_color: str = 'm'):
    """
    :param x: 经度范围，从小到大排列，最少需要两个经度
    :param y: 维度范围，从小到大排列，最少需要两个维度
    :param value: 一个list，包含等级数据，格式为List[Tuple[Point,int]],需要与inter_geometries一起使用，比如[Point(成都),1],那么在
    inter_geometries中，包含成都这个经纬度点的几何区域就为1等级，至于1等级的颜色，由level2color决定
    :param proj: 投影类
    :param inner_geometries: 内部轮廓，一般为List
    :param level2color: 等级到颜色的映射
    :param level2text: 等级到文字的映射，如果不为None，则会添加相应图例
    :param outer_geometries: 外部轮廓
    :param title: 标题
    :param add_grid_line: 是否添加经纬度线
    :param legend_title: 图例标题
    :param legend_loc: 图例位置
    :param save_path: 若不为None，则会保存图片到指定路径
    :param texts: 要添加的文字，比如[(Point,'a'),(Point,'b')]
    :param text_size: 文字大小
    :param text_color: 文字颜色
    :return:
    """
    x_max, x_min = np.max(x), np.min(x)
    y_max, y_min = np.max(y), np.min(y)
    plt.figure()
    ax = plt.axes(projection=proj)
    if title:
        ax.set_title(title)
    ax.set_extent([x_min, x_max, y_min, y_max], crs=proj)

    def colorize_geometry(geometry):
        for point, level in value:
            if geometry.contains(point):
                color = level2color[level]
                return {'facecolor': color, 'edgecolor': 'k', "linewidth": 0.5}
        else:
            return {'facecolor': 'none', 'edgecolor': 'none'}

    if not isinstance(inner_geometries, list):
        inner_geometries = [inner_geometries]
    ax.add_geometries(inner_geometries, crs=proj, styler=colorize_geometry)
    if outer_geometries and not isinstance(outer_geometries, list):
        outer_geometries = [outer_geometries]
        ax.add_geometries(outer_geometries, crs=proj, facecolor='none', edgecolor='k', linewidth=0.5)
    if add_grid_line:
        n_ticks = 6
        add_rectangular_grid_line(ax, x_min, y_min, x_max, y_max, n_ticks, n_ticks, proj)
    if texts is not None:
        add_tag(ax, texts, proj, text_size, text_color)
    if level2text:
        add_level_text(ax, level2text, level2color, legend_loc, legend_title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_china_image(x, y, value, proj, outer_geometries, nine, cmap="Set2", title=None,
                     bar_label=None, add_grid_line=True, save_path=None, texts=None,
                     text_size=7, text_color='m'):
    """
    画全国地图的二维网格图像
    :param x: 经度范围，从小到大排列，最少需要两个经度
    :param y: 维度范围，从小到大排列，最少需要两个维度
    :param value: np.ndarray 一个二维数组，与上述坐标对应，也就是value[0][0]为地图的左下角处的值
    :param proj: 投影类,一般为ccrs.LambertConformal(central_longitude=107.5,
                                                    central_latitude=36.0,standard_parallels=(25, 47)
    :param outer_geometries: 外部轮廓，比如一个省的边界信息
    :param nine 南海九段线
    :param cmap: matplotlib 色彩，具体查看 https://matplotlib.org/2.0.2/examples/color/colormaps_reference.html
    :param title:子图标题，默认为None
    :param bar_label:图例标题
    :param add_grid_line:是否加经纬度线
    :param save_path:若不为None，则会保存图片到指定路径
    :param texts:要添加的文字，比如[(Point,'a'),(Point,'b')]
    :param text_size:文字大小
    :param text_color:文字颜色
    :return:装饰过的函数
    """

    x_max, x_min = np.max(x), np.min(x)
    y_max, y_min = np.max(y), np.min(y)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.8], projection=proj)
    if title:
        ax.set_title(title)
    ax.set_extent([80, 130, 18, 52.5], crs=ccrs.PlateCarree())
    ca = ax.imshow(value, extent=[x_min, x_max, y_max, y_min], transform=ccrs.PlateCarree(),
                   interpolation='bilinear', cmap=cmap)

    if not isinstance(outer_geometries, list):
        outer_geometries = [outer_geometries]
    ax.add_geometries(outer_geometries, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5)
    ax.add_geometries(nine, ccrs.PlateCarree(), facecolor='none', edgecolor='r', linewidth=2)
    clip_boundary(ax, ca, outer_geometries, ccrs.PlateCarree())

    if add_grid_line:
        n_ticks = 6
        add_ball_grid_line(ax, x_min, y_min, x_max, y_max, n_ticks, n_ticks)
    if texts is not None:
        add_tag(ax, texts, ccrs.PlateCarree(), text_size, text_color)

    ax2 = fig.add_axes([0.76, 0.18, 0.13, 0.18], projection=proj)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_extent([105, 120, 3, 26], crs=ccrs.PlateCarree())
    if add_grid_line:
        ax2.gridlines(draw_labels=None, x_inline=False, y_inline=False,
                      linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    ax2.add_geometries(outer_geometries, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5)
    ax2.add_geometries(nine, ccrs.PlateCarree(), facecolor='none', edgecolor='r', linewidth=2)
    ca2 = ax2.imshow(value, extent=[x_min, x_max, y_max, y_min], transform=ccrs.PlateCarree(),
                     interpolation='bilinear', cmap=cmap)
    clip_boundary(ax2, ca2, outer_geometries, ccrs.PlateCarree())
    position = fig.add_axes([0.1, 0.08, 0.8, 0.02])
    plt.colorbar(mappable=ca, ax=ax, cax=position, orientation='horizontal',
                 shrink=0.2, label=bar_label, pad=0.1)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_china_level(x, y, value, proj, outer_geometries, nine, inner_geometries, level2color, level2text=None,
                     title=None, add_grid_line=True, legend_title=None, legend_loc='lower left',
                     save_path=None, texts=None, text_size=7, text_color='m'):
    """
    画全国地图的等级图
    :param x: 经度范围，从小到大排列，最少需要两个经度
    :param y: 维度范围，从小到大排列，最少需要两个维度
    :param value: 一个list，包含等级数据，格式为List[Tuple[Point,int]],需要与inter_geometries一起使用，比如[Point(成都),1],那么在
    inter_geometries中，包含成都这个经纬度点的几何区域就为1等级，至于1等级的颜色，由level2color决定
    :param proj: 投影类,一般为ccrs.LambertConformal(central_longitude=107.5,
                                                    central_latitude=36.0,standard_parallels=(25, 47)
    :param nine: 南海九段线
    :param inner_geometries: 内部轮廓，一般为List
    :param level2color: 等级到颜色的映射
    :param level2text: 等级到文字的映射，如果不为None，则会添加相应图例
    :param outer_geometries: 外部轮廓
    :param title: 标题
    :param add_grid_line: 是否添加经纬度线
    :param legend_title: 图例标题
    :param legend_loc: 图例位置
    :param save_path: 若不为None，则会保存图片到指定路径
    :param texts: 要添加的文字，比如[(Point,'a'),(Point,'b')]
    :param text_size: 文字大小
    :param text_color: 文字颜色
    :return:
    """
    x_max, x_min = np.max(x), np.min(x)
    y_max, y_min = np.max(y), np.min(y)
    fig = plt.figure()
    ax = plt.axes([0.1, 0.1, 0.8, 0.8], projection=proj)
    if title:
        ax.set_title(title)
    ax.set_extent([80, 130, 18, 52.5], crs=ccrs.PlateCarree())
    if not isinstance(outer_geometries, list):
        outer_geometries = [outer_geometries]

    def colorize_geometry(geometry):
        for point, level in value:
            if geometry.contains(point):
                color = level2color[level]
                return {'facecolor': color, 'edgecolor': 'k', "linewidth": 0.5}
        else:
            return {'facecolor': 'none', 'edgecolor': 'none'}

    ax.add_geometries(outer_geometries, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5)
    ax.add_geometries(nine, ccrs.PlateCarree(), facecolor='none', edgecolor='r', linewidth=2)
    ax.add_geometries(inner_geometries, crs=ccrs.PlateCarree(), styler=colorize_geometry)

    if add_grid_line:
        n_ticks = 6
        add_ball_grid_line(ax, x_min, y_min, x_max, y_max, n_ticks, n_ticks)

    if texts is not None:
        add_tag(ax, texts, ccrs.PlateCarree(), text_size, text_color)

    ax2 = fig.add_axes([0.75, 0.12, 0.16, 0.18], projection=proj)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_extent([105, 120, 3, 26], crs=ccrs.PlateCarree())
    if add_grid_line:
        ax2.gridlines(draw_labels=None, x_inline=False, y_inline=False,
                      linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    ax2.add_geometries(outer_geometries, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.5)
    ax2.add_geometries(nine, ccrs.PlateCarree(), facecolor='none', edgecolor='r', linewidth=2)
    ax2.add_geometries(inner_geometries, crs=ccrs.PlateCarree(), styler=colorize_geometry)
    if level2text:
        add_level_text(ax, level2text, level2color, legend_loc, legend_title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


__all__ = [
    "clip_boundary",
    "add_rectangular_grid_line",
    "add_ball_grid_line",
    "add_level_text",
    "add_tag",
    "plot_roi_image",
    "plot_roi_contourf",
    "plot_roi_level",
    "plot_china_image",
    "plot_china_level"
]

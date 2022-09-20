from vis.core import *
from vis.utils import *
import cartopy.crs as ccrs
import random


def test_plot_roi_image(province='四川省'):
    dataset = xr.open_dataset(r"example_data/example_data.nc")
    dataset = dataset.rename({"LON": "lon", "LAT": "lat"})
    province_geo = get_province_geometry("shp",province)  # 返回省份的轮廓
    city_info = get_city_geometries("shp",province)  # 返回省会所有的城市信息，为一个字典，键为城市名字，值为城市轮廓

    texts = []
    inner_geometries = []

    for city_name, city_geo in city_info.items():
        inner_geometries.append(city_geo)  # 积攒所有的城市轮廓
        texts.append((city_name, get_text_position_from_geometry(city_geo)))  # 为每个城市添加名称
    roi = extract_roi_from_xr(dataset, box=province_geo.bounds, size=(128, 128))  # 提取省会所需数据
    plot_roi_image(roi.lon.values, roi.lat.values, roi.PRCP.values, ccrs.PlateCarree(), province_geo,
                   inter_geometries=inner_geometries, texts=texts, crop=True,cmap='Greens')


def test_plot_roi_level(province='四川省'):
    province_geo = get_province_geometry("shp",province)  # 返回省份的轮廓
    city_info = get_city_geometries("shp",province)  # 返回省会所有的城市信息，为一个字典，键为城市名字，值为城市轮廓

    texts = []
    inner_geometries = []
    values = []
    for city_name, city_geo in city_info.items():
        inner_geometries.append(city_geo)  # 积攒所有的城市轮廓
        texts.append((city_name, get_text_position_from_geometry(city_geo)))  # 为每个城市添加名称
        values.append((get_text_position_from_geometry(city_geo),random.randint(1,4)))  # 准备等级数据

    level2color = {1: "blue", 2: "yellow", 3: "orange", 4: "red"}
    level2text = {1: "一般", 2: "较严重", 3: "严重", 4: "特别严重"}
    lon_min, lat_min, lon_max, lat_max = province_geo.bounds
    lon_range = [lon_min - 0.5,lon_max + 0.5]
    lat_range = [lat_min - 0.5,lat_max + 0.5]
    plot_roi_level(lon_range, lat_range,values,ccrs.PlateCarree(), inner_geometries=inner_geometries,
                   outer_geometries=province_geo, texts=texts, level2color=level2color, level2text=level2text)


def test_plot_china_level():
    china_geo = get_china_geometry("shp")
    nine_geo = get_nine_geometry("shp")
    provinces_geo = get_province_geometries("shp")
    texts = []
    values = []
    inner_geometries = []
    for name, geometry in provinces_geo.items():
        values.append([get_text_position_from_geometry(geometry), random.randint(1, 4)])
        inner_geometries.append(geometry)
        texts.append((name, get_text_position_from_geometry(geometry)))
    level2color = {1: "blue", 2: "yellow", 3: "orange", 4: "red"}
    level2text = {1: "一般", 2: "较严重", 3: "严重", 4: "特别严重"}
    lon_min, lat_min, lon_max, lat_max = china_geo.bounds
    lon_range = [lon_min - 0.5,lon_max + 0.5]
    lat_range = [lat_min - 0.5,lat_max + 0.5]

    plot_china_level(lon_range, lat_range, values, ccrs.LambertConformal(central_longitude=107.5,
                                                                                   central_latitude=36.0,
                                                                                   standard_parallels=(25, 47)),
                     inner_geometries=inner_geometries, outer_geometries=china_geo,
                     texts=texts, level2color=level2color, level2text=level2text, nine=nine_geo)


def test_plot_china_image():
    dataset = xr.open_dataset(r"example_data/example_data.nc")
    dataset = dataset.rename({"LON": "lon", "LAT": "lat"})
    province_geo = get_province_geometries("shp")
    china = get_china_geometry("shp")
    nine = get_nine_geometry("shp")
    texts = []
    for name, geometry in province_geo.items():
        texts.append((name, get_text_position_from_geometry(geometry)))
    roi = extract_roi_from_xr(dataset, box=china.bounds, size=(512, 512))
    plot_china_image(roi.lon.values, roi.lat.values, value=roi.PRCP.values,
                     proj=ccrs.LambertConformal(central_longitude=107.5,
                                                central_latitude=36.0, standard_parallels=(25, 47)),
                     outer_geometries=china, nine=nine, texts=texts, bar_label="降水",cmap="BuGn")


if __name__ == '__main__':
    test_plot_china_image()
    test_plot_china_level()
    test_plot_roi_level()
    test_plot_roi_image()
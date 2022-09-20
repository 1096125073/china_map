from shapely.geometry.base import BaseGeometry
import xarray as xr
import numpy as np
import os
import pickle
from typing import Tuple,Optional


def extract_roi_from_xr(xr_dataset: xr.Dataset,
                        box: Tuple[float, float, float, float],
                        size: Optional[Tuple[int, int]] = None,
                        interpolate_method: str = "linear", pad: float = 0.2) -> xr.Dataset:
    """
    根据一个方框边界从xarry中提取出数据，xarry的坐标为经纬度,坐标名称为lon,lat
    :param pad: 将box往外扩充pad经纬度，扩充裁减一部分区域,主要是为了避免画图的时候边框刚好在图的边界上
    :param size: 如果要对数据进行resize，则需要提供宽度和高度(width,height)
    :param interpolate_method: str,optional
            {"linear", "nearest"} for multidimensional array,
            {"linear", "nearest", "zero", "slinear", "quadratic", "cubic"}
            for 1-dimensional array. "linear" is used by default.
    :param xr_dataset:xarray dataset
    :param box:[lon_min,lat_min,lon_max,lat_max]
    :return: 经过裁减的xr_dataset，坐标经纬度范围在box内
    """
    assert len(box) == 4, "border should be a list with [lon_min,lat_min,lon_max,lat_max]"
    raw_lon, raw_lat = xr_dataset["lon"].values, xr_dataset['lat'].values
    lon_min, lat_min, lon_max, lat_max = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    if size is None:
        cond_lon = np.bitwise_and(raw_lon <= lon_max, raw_lon >= lon_min)
        select_lon = raw_lon[cond_lon]
        cond_lat = np.bitwise_and(raw_lat <= lat_max, raw_lat >= lat_min)
        select_lat = raw_lat[cond_lat]
        dataset = xr_dataset.sel(lon=sorted(select_lon.tolist()), lat=sorted(select_lat.tolist()))
        return dataset
    else:
        assert len(size) == 2 and isinstance(size[0], int), "size should be int (width,height)"
        width, height = size
        lon_unit = (lon_max - lon_min) / width
        lat_unit = (lat_max - lat_min) / height
        select_lon = [lon_min + i * lon_unit for i in range(width)]
        select_lat = [lat_min + i * lat_unit for i in range(height)]
        return xr_dataset.interp(lon=select_lon, lat=select_lat, method=interpolate_method)


def read_pickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)


def get_china_geometry(shp_dir):
    """
    得到中国的轮廓
    :param shp_dir:
    :return: BaseGeometry
    """
    path = os.path.join(shp_dir,"china.bin")
    return read_pickle(path)


def get_province_geometries(shp_dir):
    """
    获得全部省份的轮廓
    :param shp_dir:
    :return:Dict[name:BaseGeometry]
    """
    path = os.path.join(shp_dir,"province","province.bin")
    province = read_pickle(path)
    return {key:value['geometry'] for key,value in province.items()}


def get_province_geometry(shp_dir,province_name):
    """
    获得某个省份的轮廓
    :param shp_dir:
    :param province_name:
    :return:BaseGeometry
    """
    province_dict = get_province_geometries(shp_dir)
    if province_dict.get(province_name) is not None:
        return province_dict[province_name]
    else:
        for key,value in province_dict.items():
            if province_name[:2] in key:
                return value
    return None


def get_city_geometries(shp_dir,province_name):
    """
    获得某个省份里所有的城市轮廓
    :param shp_dir:
    :param province_name:
    :return:Dict[city_name:BaseGeometry]
    """
    path = os.path.join(shp_dir,"city","city.bin")
    city_dict = read_pickle(path)
    if city_dict.get(province_name) is not None:
        return {key:value['geometry'][0] for key,value in city_dict.get(province_name).items()}
    else:
        for key,value in city_dict.items():
            if province_name[:2] in key:
                return {key:value['geometry'][0] for key,value in value.items()}
    return None


def get_nine_geometry(shp_dir):
    """
    南海九段线geometry
    :param shp_dir:
    :return:BaseGeometry
    """
    return read_pickle(os.path.join(shp_dir,"nine","nine.bin"))


def get_text_position_from_geometry(geometry):
    """
    给与一个geometry，返回具有代表性的一个点，可以作为文字标注点
    :param geometry:
    :return:Point
    """
    assert isinstance(geometry,BaseGeometry),"need type {},but got {}".format(BaseGeometry,type(geometry))
    return geometry.representative_point()


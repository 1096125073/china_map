import xarray as xr
import matplotlib.pyplot as plt
xr_data = xr.open_dataset("example_data.nc")
xr_data.PRCP.plot()
plt.show()
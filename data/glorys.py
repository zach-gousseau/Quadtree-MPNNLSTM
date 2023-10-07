from pydap.cas.get_cookies import setup_session
from pydap.client import open_url
import xarray as xr
import rioxarray
import numpy as np
from requests.sessions import Session


USERNAME = 'your-cmems-username'
PASSWORD = 'your-cmems-password'
DATASET_ID = 'cmems_mod_glo_phy_my_0.083_P1D-m'

def copernicusmarine_datastore(dataset, username, password):
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
    
    # Set timeout for the session
    session_with_timeout = Session()
    session_with_timeout.verify = session.verify
    session_with_timeout.auth = session.auth
    session_with_timeout.cookies = session.cookies
    session_with_timeout.headers = session.headers
    
    database = ['my', 'nrt']
    url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
    try:
        data_store = xr.backends.PydapDataStore(open_url(url, session=session_with_timeout, timeout=600))  
    except:
        url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
        data_store = xr.backends.PydapDataStore(open_url(url, session=session_with_timeout, timeout=600))
    return data_store

def get_glorys_values(ds, dt, var_='siconc', lats=(51, 70), lons=(-95, -65)):
    ds_sub = ds.sel(
        latitude=lats,
        longitude=lons,
        time=dt
    )
    
    if var_ == 'sivol':
        ds_sub['sivol'] = ds_sub.siconc * ds_sub.sithick

    arr = ds_sub[var_]
    arr.values = np.nan_to_num(arr.values)
    arr.values[np.isnan(ds_sub.zos.values)] = np.nan
    return arr

def get_glorys(username, password):
    data_store = copernicusmarine_datastore(DATASET_ID, username, password)
    glorys = xr.open_dataset(data_store)
    glorys['time'] = np.array([dt - 12 * 3600000000000 for dt in glorys.time.values])
    glorys = glorys.rio.write_crs(4326)
    return glorys
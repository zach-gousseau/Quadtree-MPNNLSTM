import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import argparse
import cdsapi
import os
from tqdm import tqdm

# Set region of interest
lats = (51, 70)
lons = (-95, 65)

def download_era5(start_year, end_year, region, CDS_key):
    """
    Download ERA5 files
    :param start_year: Integer. Start year in YYYY.
    :param end_year: Integer. End year in YYYY.
    :param region: Latitude/longitude extents as format: (e.g.) "90/-180/40/180"
    :param CDS_key: Secret key for the Climate Data Store: https://cds.climate.copernicus.eu/#!/home
    """

    output_dir = os.getcwd()
    base = "ERA5_"
    url = "https://cds.climate.copernicus.eu/api/v2"

    features = ['10m_u_component_of_wind',
                '10m_v_component_of_wind',
                #'10m_wind_gust_since_previous_post_processing',
                #'2m_dewpoint_temperature',
		        '2m_temperature',
                #'evaporation',
                #'mean_sea_level_pressure', 'mean_wave_direction',
                #'mean_wave_period',
                'sea_ice_cover', #'sea_surface_temperature',
                #'significant_height_of_combined_wind_waves_and_swell',
                #'snowfall', 'snowmelt', 
		        # 'surface_latent_heat_flux',
                'surface_sensible_heat_flux',]
                # 'total_cloud_cover',
                #'total_precipitation',
                #'surface_solar_radiation_downwards']

    for year in range(end_year, start_year, -1):
        print(year)
        os.chdir(output_dir)

        for month in range(1, 13):  # up to 12
            os.chdir(output_dir)

            print(month)
            # '01' instead of '1'
            month = str(month).rjust(2, '0')

            # eg. 1979-01
            subdirectory = "{}-{}".format(year, month)
            if not os.path.isdir(subdirectory):
                os.mkdir(subdirectory)

            os.chdir(subdirectory)

            # _197901.nc
            extension = "_{}{}.nc".format(year, str(month).rjust(2, '0'))

            for feature in features:
                print(feature)

                # eg. ERA5_10m_u_component_of_wind_197901.nc
                filename = base + feature + extension

                if not os.path.isfile(filename):
                    print("Downloading file {}".format(filename))

                    downloaded = False

                    while not downloaded:
                        try:
                            client = cdsapi.Client(url=url, key=CDS_key, retry_max=5)
                            client.retrieve(
                                'reanalysis-era5-single-levels',
                                {
                                    'product_type': 'reanalysis',
                                    'format': 'netcdf',
                                    'variable': feature,
                                    'area': f'{lats[1]}/{lons[0]}/{lats[0]}/{lons[1]}',
                                    'time': [
                                        '00:00', '03:00', '06:00',
                                        '09:00', '12:00', '15:00',
                                        '17:00', '21:00'
                                    ],
                                    'day': [
                                        '01', '02', '03',
                                        '04', '05', '06',
                                        '07', '08', '09',
                                        '10', '11', '12',
                                        '13', '14', '15',
                                        '16', '17', '18',
                                        '19', '20', '21',
                                        '22', '23', '24',
                                        '25', '26', '27',
                                        '28', '29', '30', '31'
                                    ],
                                    # API ignores cases where there are less than 31 days
                                    'month': month,
                                    'year': str(year)
                                },
                                filename)

                        except Exception as e:
                            print(e)

                            # Delete the partially downloaded file.
                            if os.path.isfile(filename):
                                os.remove(filename)

                        else:
                            # no exception implies download was complete
                            downloaded = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 files to current working directory.")
    parser.add_argument("--start_year", type=int, default=1979,
                        help="Year to start in YYYY format.")
    parser.add_argument("--end_year", type=int, default=2019,
                        help="Year to end in YYYY format.")
    parser.add_argument("--region", type=str, default='90/-180/40/180',
                        help="Region of interest in lat0/lon0/lat1/lon1 format")
    parser.add_argument("--key", type=str,
                        help="CDS key")

    args = parser.parse_args()

    download_era5(start_year=args.start_year, end_year=args.end_year, region=args.region, CDS_key=args.key)

    # 120937:07970bca-a158-4ec5-9198-93d82b859818
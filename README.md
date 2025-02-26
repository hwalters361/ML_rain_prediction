# ML_rain_prediction

## Data

## Setup

It's recommended to use a python virtual environment for this project using python 3.9

`python -m env venv` to create a blank virtual environment
or `python3.9 -m env venv` to specify the python version without needing to configure it later.

Activate the virtual environment with `source env/bin/activate`

Then `pip install -r requirements.txt`

## Troubleshooting

"WARNING: There was an error checking the latest version of pip."
If running into problems installing requirements, run the following commands:

`pip install --upgrade pip`
`pip cache info`
to view the cache info. Then
`pip cache purge`

Then attempt to install requirements again into the virtual environment.

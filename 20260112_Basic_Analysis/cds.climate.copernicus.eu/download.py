import cdsapi

dataset = "derived-reanalysis-energy-moisture-budget"
request = {
    "variable": "vertical_integral_of_northward_total_energy_flux",
    "year": ["2022"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()

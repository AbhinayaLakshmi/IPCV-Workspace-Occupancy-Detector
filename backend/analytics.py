import pandas as pd
import os


def save_analytics(person_count, occupancy, zones):

    data = {
        "People Count":[person_count],
        "Occupancy %":[occupancy],
        "Zones Active":[",".join(zones)]
    }

    df = pd.DataFrame(data)

    file_path = "data/analytics.csv"

    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)
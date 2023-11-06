import pandas as pd
import motive_client

if __name__ == "__main__":
    markers_df = pd.read_csv('Data/markers.csv')
    rigid_bodies_df = pd.read_csv('Data/rigid_bodies.csv')

    for pose in range(1, 9):
        markers_df_ = markers_df[markers_df["pose"]
                                 == pose].drop('pose', axis=1)
        rigid_bodies_df_ = rigid_bodies_df[rigid_bodies_df["pose"] == pose].drop(
            'pose', axis=1)

        markers = {}
        rigid_bodies = {}

        for index, row in markers_df_.iterrows():
            marker = row.to_dict()
            markers[marker['marker_id']] = marker

        for index, row in rigid_bodies_df_.iterrows():
            rigid_body = row.to_dict()
            rigid_bodies[rigid_body['id']] = rigid_body

        motive_client.getWheelsCoords(markers, rigid_bodies)

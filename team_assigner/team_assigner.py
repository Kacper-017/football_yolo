from sklearn.cluster import KMeans
import numpy as np


class TeamAssigner:
    """Assigns team colors to the players based on their jersey color using KMeans clustering
    """
    def __init__(self) -> None:
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image: np.ndarray) -> KMeans:
        """Get the KMeans clustering model

        Args:
            image (np.ndarray): Image to be clustered

        Returns:
            KMeans: KMeans clustering model
        """
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters
        # kmeans++ is greedy initialization
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Get the color of the player from the bounding box

        Args:
            frame (np.ndarray): video frame
            bbox (np.ndarray): bounding box of the player

        Returns:
            np.ndarray: Color of the player
        """
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        top_half_image = image[0:int(image.shape[0]/2), :]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1],
                           clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters),
                                 key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame: np.ndarray, player_detections: dict) -> None:
        """Assign team colors to the players

        Args:
            frame (np.ndarray): video frame
            player_detections (dict): player detections
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame: np.ndarray, player_bbox: np.ndarray, player_id: int) -> int:
        """Get the team id of the player

        Args:
            frame (np.ndarray): video frame
            player_bbox (np.ndarray): bounding box of the player
            player_id (int): player id

        Returns:
            int: Team id of the player
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1

        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id

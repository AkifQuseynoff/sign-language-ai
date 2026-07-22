from feature_extraction.hand_geometry import HandGeometry
class FeatureExtractor:
    def __init__(self):
        self.HAND_BONES = [
            (0,1),
            (1,2),
            (2,3),
            (3,4),
            (0,5),
            (5,6),
            (6,7),
            (7,8),
            (0,9),
            (9,10),
            (10,11),
            (11,12),
            (0,13),
            (13,14),
            (14,15),
            (15,16),
            (0,17),
            (17,18),
            (18,19),
            (19,20)]

        self.JOINT_ANGLES = [
            (1,2,3),
            (2,3,4),
            (5,6,7),
            (6,7,8),
            (9,10,11),
            (10,11,12),
            (13,14,15),
            (14,15,16),
            (17,18,19),
            (18,19,20)]

        self.FINGERTIPS = [
            4,
            8,
            12,
            16,
            20]

        self.FINGER_VECTORS = [
            (1, 4),
            (5, 8),
            (9, 12),
            (13, 16),
            (17, 20)]

    def extract_angles(self, points):
        angles = []
        for a, b, c in self.JOINT_ANGLES:
            theta = HandGeometry.angle(points[a], points[b], points[c])
            if theta is not None:
                angles.append(theta)
            else:
                angles.append(0.0)
        return angles

    def extract_bone_lengths(self, points):
        lengths = []
        for a, b in self.HAND_BONES:
            length = HandGeometry.distance(points[a], points[b])
            lengths.append(length)
        return lengths
    
    def extract_tip_to_tip_distances(self, points):
        distances = []
        for i in range(len(self.FINGERTIPS)):
            for j in range(i + 1, len(self.FINGERTIPS)):
                a = self.FINGERTIPS[i]
                b = self.FINGERTIPS[j]
                dist = HandGeometry.distance(points[a], points[b])
                distances.append(dist)
        return distances
    
    def extract_wrist_to_tip_distances(self, points):
        distances = []
        wrist = points[0]
        for tip in self.FINGERTIPS:
            dist = HandGeometry.distance(wrist, points[tip])
            distances.append(dist)
        return distances
    
    def extract_finger_vectors(self, points):
        vectors = []
        for a, b in self.FINGER_VECTORS:
            vec = HandGeometry.vector(points[a], points[b] , normalize=True)
            vectors.extend(vec)
        return vectors
    
    def extract_features(self, points):
        features = []
        features.extend(self.extract_angles(points))
        features.extend(self.extract_bone_lengths(points))
        features.extend(self.extract_tip_to_tip_distances(points))
        features.extend(self.extract_wrist_to_tip_distances(points))
        features.extend(self.extract_finger_vectors(points))

        return features
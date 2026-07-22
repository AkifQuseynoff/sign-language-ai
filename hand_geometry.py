import numpy as np

class HandGeometry:

    @staticmethod
    def distance(a, b):
        """Calculates distance between two points a and b."""
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(b - a)
    @staticmethod
    def vector(a, b , normalize=False):
        """Calculates vector from point a to point b."""
        a = np.array(a)
        b = np.array(b)
        vec= b - a
        if normalize:
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                return np.zeros(3)
            return vec / norm
        else:
            return vec
    
    @staticmethod
    def angle(a, b, c):
        """Calculates angle between three points a, b, and c with b as the vertex."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denominator < 1e-8:
            return None
        cos_theta = np.dot(ba, bc) / denominator
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_theta))
        return theta
    
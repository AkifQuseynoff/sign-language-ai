import cv2

class Display:
    def show(self, frame, result, prediction):
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                for landmark in hand:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )
        cv2.putText(
            frame,
            prediction,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )
        cv2.imshow("Gesture Detection", frame)

    def should_close(self):
        return cv2.waitKey(1) & 0xFF == ord("q")
    
    def close(self):
        cv2.destroyAllWindows()
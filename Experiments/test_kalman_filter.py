class KalmanFilter:
    def __init__(self, initial_temp, process_variance, measurement_variance):
        self.estimate = initial_temp
        self.estimate_error = 1.0
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def update(self, measurement):
        # Prediction
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance

        # Update
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error

        return self.estimate
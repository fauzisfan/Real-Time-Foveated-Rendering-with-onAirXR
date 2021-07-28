import csv
from ._types import MotionData, PredictedData
from ._writer import PredictionOutputWriter


class MotionPredictSimulator:
    def __init__(self, module, input_motion_data, prediction_output):
        self.module = module
        self.input_motion_data = input_motion_data
        self.prediction_output = PredictionOutputWriter(
            prediction_output
        ) if prediction_output is not None else None

    def run(self):
        with open(self.input_motion_data, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                motion_data = MotionData(
                    float(row["timestamp"]),
                    [float(row["input_left_eye_position_x"]),
                     float(row["input_left_eye_position_y"]),
                     float(row["input_left_eye_position_z"])],
                    [float(row["input_right_eye_position_x"]),
                     float(row["input_right_eye_position_y"]),
                     float(row["input_right_eye_position_z"])],
                    [float(row["input_head_orientation_x"]),
                     float(row["input_head_orientation_y"]),
                     float(row["input_head_orientation_z"]),
                     float(row["input_head_orientation_w"])],
                    [float(row["input_head_acceleration_x"]),
                     float(row["input_head_acceleration_y"]),
                     float(row["input_head_acceleration_z"])],
                    [float(row["input_head_angular_vec_x"]),
                     float(row["input_head_angular_vec_y"]),
                     float(row["input_head_angular_vec_z"])],
                    [float(row["input_camera_projection_left"]),
                     float(row["input_camera_projection_top"]),
                     float(row["input_camera_projection_right"]),
                     float(row["input_camera_projection_bottom"])],
                    [float(row["input_right_hand_position_x"]),
                     float(row["input_right_hand_position_y"]),
                     float(row["input_right_hand_position_z"])],
                    [float(row["input_right_hand_orientation_x"]),
                     float(row["input_right_hand_orientation_y"]),
                     float(row["input_right_hand_orientation_z"]),
                     float(row["input_right_hand_orientation_w"])],
                    [float(row["input_right_hand_acceleration_x"]),
                     float(row["input_right_hand_acceleration_y"]),
                     float(row["input_right_hand_acceleration_z"])],
                    [float(row["input_right_hand_angular_vec_x"]),
                     float(row["input_right_hand_angular_vec_y"]),
                     float(row["input_right_hand_angular_vec_z"])],
                     0
                )

                prediction_time, left_eye_position, right_eye_position, \
                    head_orientation, camera_projection, \
                    right_hand_position, right_hand_orientation = \
                    self.module.predict(motion_data)
                
                predicted_data = PredictedData(motion_data.timestamp,
                                               prediction_time,
                                               left_eye_position,
                                               right_eye_position,
                                               head_orientation,
                                               camera_projection,
                                               right_hand_position,
                                               right_hand_orientation,
                                               0,
                                               0,
                                               0)

                if self.prediction_output is not None:
                    self.prediction_output.write(motion_data, predicted_data)

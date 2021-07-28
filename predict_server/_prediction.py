import math

class BufferedNoPrediction:
    def __init__(self, bufferCount, prediction_time):
        self.buffer = []
        self.bufferLen = bufferCount
        self.prediction_time = prediction_time

    def put_motion_data(self, motion_data):
        self.buffer.append(motion_data)

        if len(self.buffer) > self.bufferLen:
            self.buffer.pop(0)

    def get_predicted_result(self):
        motion_data = self.buffer[0]

        camera_projection = self.overfill_camera_projection(motion_data.camera_projection, [0.1745, 0.1745, 0.1745, 0.1745])
        foveation_inner_radius = 0.25
        foveation_middle_radius = 0.35

        return self.prediction_time, \
               motion_data.left_eye_position, \
               motion_data.right_eye_position, \
               motion_data.head_orientation, \
               camera_projection, \
               foveation_inner_radius, \
               foveation_middle_radius, \
               motion_data.right_hand_position, \
               motion_data.right_hand_orientation

    def overfill_camera_projection(self, camera_projection, overfilling):
        return [
            math.tan(math.atan(camera_projection[0]) - overfilling[0]),
            math.tan(math.atan(camera_projection[1]) + overfilling[1]),
            math.tan(math.atan(camera_projection[2]) + overfilling[2]),
            math.tan(math.atan(camera_projection[3]) - overfilling[3])
        ]

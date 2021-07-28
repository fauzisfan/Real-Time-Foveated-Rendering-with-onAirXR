import zmq
import numpy as np
from ._types import MotionData, PredictedData, ExternalInputData

class MotionDataTransport:
    def __init__(self, owner):
        self.ann_pred_rt = np.zeros((0,3), dtype= float)
        self.cap_pred_rt = np.zeros((0,3), dtype= float)
        self.crp_pred_rt = np.zeros((0,3), dtype= float)
        self.nop_pred_rt = np.zeros((0,3), dtype= float)
        self.ori_rt = np.zeros((0,3), dtype= float)
        
        self.owner = owner
        self.accept_client_buttons = False

    def configure(self, context, poller, port_recv, port_send, accept_client_buttons):
        self.socket_recv = context.socket(zmq.PULL)
        self.socket_recv.bind("tcp://*:" + str(port_recv))

        self.socket_send = context.socket(zmq.PUSH)
        self.socket_send.bind("tcp://*:" + str(port_send))

        poller.register(self.socket_recv, zmq.POLLIN)

        self.accept_client_buttons = accept_client_buttons

    async def process_events(self, events, external_input, overstyle, sess):
        if self.socket_recv not in dict(events):
            return

        frame = await self.socket_recv.recv(0, False)
        motion_data = MotionData.from_bytes(frame.bytes)

        self.owner.pre_predict_motion(motion_data.timestamp)

        prediction_time, \
            predicted_left_eye_position, \
            predicted_right_eye_position, \
            predicted_head_orientation, \
            predicted_camera_projection, \
            predicted_foveation_inner_radius, \
            predicted_foveation_middle_radius, \
            predicted_right_hand_position, \
            predicted_right_hand_orientation, \
            input_euler, ann, cap, crp, flag = \
            self.owner.predict_motion(motion_data, overstyle, sess)
            
        if (flag == 1):
            self.ori_rt = np.concatenate((self.ori_rt, np.array(input_euler).reshape(1,-1)), axis =0)
            self.ann_pred_rt = np.concatenate((self.ann_pred_rt, ann), axis =0)
            self.crp_pred_rt = np.concatenate((self.crp_pred_rt, crp), axis =0)
            self.cap_pred_rt = np.concatenate((self.cap_pred_rt, cap), axis =0)
            self.nop_pred_rt = np.concatenate((self.nop_pred_rt, np.array(input_euler).reshape(1,-1)), axis =0)
            
        if self.accept_client_buttons:
            external_input.set_input(ExternalInputData(
                motion_data.timestamp,
                0,
                motion_data.right_hand_primary_button_press,
                motion_data.right_hand_primary_button_press
            ))

        # TODO: add all inputs to predicted data
        input_data = external_input.get_input(0)

        predicted_data = PredictedData(motion_data.timestamp,
                                        prediction_time,
                                        motion_data.left_eye_position,
                                        motion_data.right_eye_position,
                                        motion_data.head_orientation,
                                        motion_data.camera_projection,
                                        motion_data.right_hand_position,
                                        motion_data.right_hand_orientation,
                                        predicted_left_eye_position,
                                        predicted_right_eye_position,
                                        predicted_head_orientation,
                                        predicted_camera_projection,
                                        predicted_foveation_inner_radius,
                                        predicted_foveation_middle_radius,
                                        predicted_right_hand_position,
                                        predicted_right_hand_orientation,
                                        input_data.id if input_data != None else 0,
                                        input_data.actual_press if input_data != None else False,
                                        input_data.predicted_press if input_data != None else False)

        self.owner.post_predict_motion(motion_data.timestamp)

        self.socket_send.send(predicted_data.pack())

        self.owner.write_prediction_output(motion_data, predicted_data)
        
    def perfResult (self):
        
        ann_mae = np.nanmean(np.abs(self.ann_pred_rt[7:-(22+22)] - self.ori_rt[22+7:-22]), axis=0)
        crp_mae = np.nanmean(np.abs(self.crp_pred_rt[7:-(22+22)] - self.ori_rt[22+7:-22]), axis=0)
        cap_mae = np.nanmean(np.abs(self.cap_pred_rt[7:-(22+22)] - self.ori_rt[22+7:-22]), axis=0)
        nop_mae = np.nanmean(np.abs(self.nop_pred_rt[7:-(22+22)] - self.ori_rt[22+7:-22]), axis=0)
        
        final_ann_rt_99 = np.nanpercentile(np.abs(self.ann_pred_rt[7:-(22+22)] - self.ori_rt[22+7:-22]),99, axis = 0)
        final_crp_rt_99 = np.nanpercentile(np.abs(self.crp_pred_rt[7:-(22+22)] - self.ori_rt[22+7:-22]),99, axis = 0)
        final_cap_rt_99 = np.nanpercentile(np.abs(self.cap_pred_rt[7:-(22+22)] - self.ori_rt[22+7:-22]),99, axis = 0)
        final_nop_rt_99 = np.nanpercentile(np.abs(self.nop_pred_rt[7:-(22+22)] - self.ori_rt[22+7:-22]),99, axis = 0)
        
        print('\nFinal Result of MAE:')
        print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae[0], ann_mae[1], ann_mae[2]))
        print('CRP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(crp_mae[0], crp_mae[1], crp_mae[2]))
        print('CAP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(cap_mae[0], cap_mae[1], cap_mae[2]))
        print('NOP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(nop_mae[0], nop_mae[1], nop_mae[2]))
        
        print('\nFinal Result of 99% Error:')
        print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_ann_rt_99[0], final_ann_rt_99[1], final_ann_rt_99[2]))
        print('CRP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_crp_rt_99[0], final_crp_rt_99[1], final_crp_rt_99[2]))
        print('CAP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_cap_rt_99[0], final_cap_rt_99[1], final_cap_rt_99[2]))
        print('NOP [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_nop_rt_99[0], final_nop_rt_99[1], final_nop_rt_99[2]))
import sys
import getopt
from predict_server import PredictModule, MotionPredictServer
from predict_server.simulator import MotionPredictSimulator
import numpy as np
import math
import quaternion
import tensorflow as tf
from realPred_train import pred
from train import train_nn
import os

# from predict_server import BufferedNoPrediction


class App(PredictModule):
    def __init__(self):
        # self.prediction = BufferedNoPrediction(20, 100)
        self.sum_overall_latency = 0
        self.count_overall_latency = 0
        pass

    def parse_command_args(self):
        port = 5555
        feedback = 5554
        input_file = output = metric_output = game_event_output = None
        accept_client_buttons = False
        
        try:
            opts, _args = getopt.getopt(sys.argv[1:], "p:f:m:o:i:g:s:", ["accept-client-buttons"])
        except getopt.GetoptError as err:
            print(err)
            sys.exit(1)

        for opt, arg in opts:
            if opt == "-p":
                port = int(arg)
            elif opt == "-f":
                feedback = int(arg)
            elif opt == "-i":
                input_file = arg
            elif opt == "-o":
                output = arg
            elif opt == "-m":
                metric_output = arg
            elif opt == "-g":
                game_event_output = arg
            elif opt == "-s":
                overstyle = arg
            elif opt == "--accept-client-buttons":
                accept_client_buttons = True
            else:
                assert False, "unhandled option"
                
        return port, feedback, input_file, output, metric_output, game_event_output, overstyle, accept_client_buttons

    def run(self):
        self.rt_counter = 0
        self.x_seq_rt = np.zeros(shape=(7, 9))
        self.timestamp_rt = np.zeros(shape=(2, 1), dtype= float)
        self.gyro_rt = np.zeros(shape=(2, 3), dtype= float)
        self.eul_rt = np.zeros(shape=(2, 3), dtype= float)

        self.flag = 0
        self.trainTime = 0.1*4320 #Number of minutes (72FPS)
        self.train_counter = 0
        self.eul_train = np.zeros((0,3), dtype= float)
        self.gyro_train = np.zeros((0,3), dtype= float)
        self.time_stamp = np.zeros((0,1), dtype= float)
        self.input_nm = 9
        
        '''For Step-Wise'''
        self.err_win=50
        self.tracked_std = np.zeros((0,3), dtype=float)
        self.tracked_mean = np.zeros((0,3), dtype=float)
        self.ellips_values = np.empty((0,4), dtype=np.float32)
        self.ido_values = np.empty((0,4), dtype=np.float32)
        self.pred_overhead = np.empty((0,1), dtype=np.float32)
        
        # [Left, Top, Right, Bottom]
        self.I = np.zeros(4, dtype = np.float32)
        self.D = np.zeros(4, dtype = np.float32)
        
        self.max_count = 10
        self.icr = 0.05
        
        self.d99 = np.sqrt(-2*np.log(1-0.99))
        self.tracked_err = np.zeros(shape=(self.err_win, 3))
        self.rt_tracked = 0
        
        port_input, port_feedback, input_file, output, metric_output, game_event_output, overstyle, accept_client_buttons = \
            self.parse_command_args()
        if input_file is None:
            assert(port_input is not None and port_feedback is not None)
            
            server = MotionPredictServer(
                self, port_input, port_feedback, output, metric_output, game_event_output, overstyle, accept_client_buttons
            )

            server.run()
                
        else:
            assert(output is not None)

            simulator = MotionPredictSimulator(self, input_file, output)

            try:                
                simulator.run()
            except KeyboardInterrupt:
                pass
    
    # implements PredictModule
    def FoveationPatternUpdate(self, proj_pred, proj_input, innerRadii, middleRadii):
    	
    	videoWidth = 4
    	videoHeight = 2
    	
    	eyeTextureWidth = 0.5
    	eyeTextureHeight = 0.25
    	
    	originalProj = proj_input[1]-proj_input[3]
    	overfillWidth = proj_pred[2]-proj_pred[0]
    	overfillHeight = proj_pred[1]-proj_pred[3]
    	
    	widthScale = videoWidth/2/eyeTextureWidth
    	heightScale = videoHeight/eyeTextureHeight
    	
    	encodingSizeWidth = overfillWidth*widthScale
    	encodingSizeHeight = overfillHeight*heightScale
    	
    	videoAspect = videoWidth/2/videoHeight
    	
    	scale = min(overfillWidth,overfillHeight)/originalProj
    	
    	Aspect = encodingSizeHeight/encodingSizeWidth*videoAspect
    	
    	return [innerRadii*scale*Aspect, middleRadii*scale*Aspect]

    def quat_to_euler(self, quat):        
        siny_cosp = 2 * (quat[3] * quat[1] - quat[2] * quat[0])
        cosy_cosp = 1 - 2 * (quat[1] * quat[1] + quat[2] * quat[2])
        yaw = math.atan2(siny_cosp, cosy_cosp)

        sinp = 2 * (quat[3] * quat[2] + quat[0] * quat[1])
        if abs(sinp) >= 1:
            roll = math.copysign(math.pi / 2, sinp)
        else:
            roll = math.asin(sinp)

        sinx_cosp = 2 * (quat[3] * quat[0] - quat[1] * quat[2])
        cosx_cosp = 1 - 2 * (quat[2] * quat[2] + quat[0] * quat[0])
        pitch = math.atan2(sinx_cosp, cosx_cosp)

        return [yaw, pitch, roll]

    def euler_to_quat (self, eul):
        # Convert Euler [oculus] to Quaternion [oculus]
        eul = np.deg2rad(eul)
        X_eul = eul[:,0]
        Y_eul = eul[:,1]
        Z_eul = eul[:,2]

        cos_pitch, sin_pitch = np.cos(X_eul/2), np.sin(X_eul/2)
        cos_yaw, sin_yaw = np.cos(Y_eul/2), np.sin(Y_eul/2)
        cos_roll, sin_roll = np.cos(Z_eul/2), np.sin(Z_eul/2)

        # order: w,x,y,z
        # quat = unit_q(quat)
        quat = np.nan * np.ones( (eul.shape[0],4) )
        quat[:,0] = cos_pitch * cos_yaw * cos_roll + sin_pitch * sin_yaw * sin_roll
        quat[:,1] = cos_pitch * cos_yaw * sin_roll - sin_pitch * sin_yaw * cos_roll
        quat[:,2] = cos_pitch * sin_yaw * cos_roll + sin_pitch * cos_yaw * sin_roll
        quat[:,3] = sin_pitch * cos_yaw * cos_roll - cos_pitch * sin_yaw * sin_roll
        return (quat[:,[3,1,2,0]])
        
    
    def Robust_overfilling(self, input_orientation, prediction, input_projection, offset = 1, fixed_param = 1):
        #Get Input Projection Distance for each side in x,y coordinate
        IPDx = input_projection[2]-input_projection[0]
        IPDy = input_projection[1]-input_projection[3]
    	
    	#Define projection distance to the user
        h = 1
    	
    	#Get the corner point distance to the rotation center for each side in x,y coordinate
        rx = np.sqrt(h**2+1/4*IPDx**2)
        ry = np.sqrt(h**2+1/4*IPDy**2)
    	
    	#Get initial input angle to the rotational center
        input_anglex = np.arctan(IPDx/(2*h))
        input_angley = np.arctan(IPDx/(2*h))
    	
    	#Get user's direction based on prediction motion
        pitch_diff = (prediction[0][0]-input_orientation[0])*np.pi/180
        roll_diff = (prediction[0][1]-input_orientation[1])*np.pi/180
        yaw_diff = (prediction[0][2]-input_orientation[2])*np.pi/180
    	
    	#Calculate predicted margin based on translation movement
        x_r = max(input_projection[2],rx*abs(np.sin(input_anglex-yaw_diff)))
        x_l = min(input_projection[0],-rx*abs(np.sin(input_anglex+yaw_diff)))
        y_t = max(input_projection[1],ry*abs(np.sin(input_angley+pitch_diff)))
        y_b = min(input_projection[3],-ry*abs(np.sin(input_angley-pitch_diff)))
    
    	#Calculate predicted margin based on rotational movement
        x_rr = rx*abs(np.sin(input_anglex+abs(roll_diff)))
        x_ll = -rx*abs(np.sin(input_anglex+abs(roll_diff)))
        y_tt = ry*abs(np.sin(input_angley+abs(roll_diff)))
        y_bb = -ry*abs(np.sin(input_angley+abs(roll_diff)))
    	
    	#Calculate final movement
        p_r = x_r+x_rr-IPDx/2
        p_l = x_l+x_ll+IPDx/2
        p_t = y_t+y_tt-IPDy/2
        p_b = y_b+y_bb+IPDy/2
    
        'Enhancement'
    ##	#get largest point
    #	z_yaw = abs(h-rx*np.cos(input_anglex+abs(yaw_diff)))
    #	z_pitch = abs(h-ry*np.cos(input_angley+abs(pitch_diff)))
    #	
    #	if (yaw_diff<0):
    #		p_r = np.sqrt(abs(p_r-p_l)**2+z_yaw**2)+p_l
    #	else: 
    #		p_l = p_r-np.sqrt(abs(p_l-p_r)**2+z_yaw**2)
    #	if (pitch_diff>0):
    #		p_t = np.sqrt(abs(p_t-p_b)**2+z_pitch**2)+p_b
    #	else :
    #		p_b = p_t-np.sqrt(abs(p_b-p_t)**2+z_pitch**2)
    	
    	#Calculate margin based on genrated area
        margin = np.sqrt((p_r-p_l)*(p_t-p_b)*offset)/2
        p_l = -(margin+np.sin(yaw_diff))+(input_projection[0]+1)
        p_t = margin+np.sin(pitch_diff)+(input_projection[1]-1)
        p_r = margin-np.sin(yaw_diff)+(input_projection[2]-1)
        p_b = -(margin-np.sin(pitch_diff))+(input_projection[3]+1)
    	
        'Enhancement ver 2'
    ##	Get dilation on high velocity
        p_r = max(p_r*(np.sin(abs(yaw_diff))*(fixed_param-1)+1),p_r*(np.sin(abs(pitch_diff))*(fixed_param-1)+1))
        p_l = min(p_l*(np.sin(abs(yaw_diff))*(fixed_param-1)+1),p_l*(np.sin(abs(pitch_diff))*(fixed_param-1)+1))
        p_t = max(p_t*(np.sin(abs(pitch_diff))*(fixed_param-1)+1),p_t*(np.sin(abs(yaw_diff))*(fixed_param-1)+1))
        p_b = min(p_b*(np.sin(abs(pitch_diff))*(fixed_param-1)+1), p_b*(np.sin(abs(yaw_diff))*(fixed_param-1)+1))
    	
    
    	#Shifting to predicted point
    #	return [-(abs(p_l)+np.sin(yaw_diff)),abs(p_t)+np.sin(pitch_diff),abs(p_r)-np.sin(yaw_diff),-(abs(p_b)-np.sin(pitch_diff))]
    
        return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]
        # margins = np.max(np.abs([p_l, p_t, p_r, p_b]))
        # return [-margins, margins, margins, -margins]
    
    
    def calc_optimal_overhead(self, hmd_orientation, frame_orientation, hmd_projection):
        q_d = np.matmul(
                np.linalg.inv(quaternion.as_rotation_matrix(hmd_orientation)),
                quaternion.as_rotation_matrix(frame_orientation)
                )
        #Projection Orientation:
            #hmd_projection[0] : Left (Negative X axis)
            #hmd_projection[1] : Top (Positive Y axis)
            #hmd_projection[2] : Right (Positive X axis)
            #hmd_projection[3] : Bottom (Negative Y axis)
            
        lt = np.matmul(q_d, [hmd_projection[0], hmd_projection[1], 1])
        p_lt = np.dot(lt, 1 / lt[2])
        
        rt = np.matmul(q_d, [hmd_projection[2], hmd_projection[1], 1])
        p_rt = np.dot(rt, 1 / rt[2])
        
        rb = np.matmul(q_d, [hmd_projection[2], hmd_projection[3], 1])
        p_rb = np.dot(rb, 1 / rb[2])
        
        lb = np.matmul(q_d, [hmd_projection[0], hmd_projection[3], 1])
        p_lb = np.dot(lb, 1 / lb[2])
        
        p_l = min(p_lt[0], p_rt[0], p_rb[0], p_lb[0])
        p_t = max(p_lt[1], p_rt[1], p_rb[1], p_lb[1])
        p_r = max(p_lt[0], p_rt[0], p_rb[0], p_lb[0])
        p_b = min(p_lt[1], p_rt[1], p_rb[1], p_lb[1])
        
    # 	size = max(p_r - p_l, p_t - p_b)
    # 	a_overfilling = size * size
        
    # 	a_hmd = (hmd_projection[2] - hmd_projection[0]) * (hmd_projection[1] - hmd_projection[3])
        
        return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]
        # margins = np.max(np.abs([p_l, p_t, p_r, p_b]))
        # return [-margins, margins, margins, -margins]
        
    def predict_overfilling(self, predict_orientation, predict_orientation_min, predict_orientation_max, hmd_projection):
    	# calc_optimal_overhead(input_orientation, predict_orientation, input_projection)
    	q_d_min = np.matmul(
    			np.linalg.inv(quaternion.as_rotation_matrix(predict_orientation)),
    			quaternion.as_rotation_matrix(predict_orientation_min)
    			)
    	q_d_max = np.matmul(
    			np.linalg.inv(quaternion.as_rotation_matrix(predict_orientation)),
    			quaternion.as_rotation_matrix(predict_orientation_max)
    			)
    	
    	lt_min = np.matmul(q_d_min, [hmd_projection[0], hmd_projection[1], 1])
    	p_lt_min = np.dot(lt_min, 1 / lt_min[2])
    	
    	rt_min = np.matmul(q_d_min, [hmd_projection[2], hmd_projection[1], 1])
    	p_rt_min = np.dot(rt_min, 1 / rt_min[2])
    	
    	rb_min = np.matmul(q_d_min, [hmd_projection[2], hmd_projection[3], 1])
    	p_rb_min = np.dot(rb_min, 1 / rb_min[2])
    	
    	lb_min = np.matmul(q_d_min, [hmd_projection[0], hmd_projection[3], 1])
    	p_lb_min = np.dot(lb_min, 1 / lb_min[2])
    	
    	lt_max = np.matmul(q_d_max, [hmd_projection[0], hmd_projection[1], 1])
    	p_lt_max = np.dot(lt_max, 1 / lt_max[2])
    	
    	rt_max = np.matmul(q_d_max, [hmd_projection[2], hmd_projection[1], 1])
    	p_rt_max = np.dot(rt_max, 1 / rt_max[2])
    	
    	rb_max = np.matmul(q_d_max, [hmd_projection[2], hmd_projection[3], 1])
    	p_rb_max = np.dot(rb_max, 1 / rb_max[2])
    	
    	lb_max = np.matmul(q_d_max, [hmd_projection[0], hmd_projection[3], 1])
    	p_lb_max = np.dot(lb_max, 1 / lb_max[2])
    	
    	p_l1 = min(p_lt_min[0], p_rt_min[0], p_rb_min[0], p_lb_min[0])
    	p_l2 = min(p_lt_max[0], p_rt_max[0], p_rb_max[0], p_lb_max[0])
    	p_t1 = max(p_lt_min[1], p_rt_min[1], p_rb_min[1], p_lb_min[1])
    	p_t2 = max(p_lt_max[1], p_rt_max[1], p_rb_max[1], p_lb_max[1])
    	p_r1 = max(p_lt_min[0], p_rt_min[0], p_rb_min[0], p_lb_min[0])
    	p_r2 = max(p_lt_max[0], p_rt_max[0], p_rb_max[0], p_lb_max[0])
    	p_b1 = min(p_lt_min[1], p_rt_min[1], p_rb_min[1], p_lb_min[1])
    	p_b2 = min(p_lt_max[1], p_rt_max[1], p_rb_max[1], p_lb_max[1])
    	
    	p_l = (p_l1+p_l2)/2
    	p_t = (p_t1+p_t2)/2
    	p_r = (p_r1+p_r2)/2
    	p_b = (p_b1+p_b2)/2
    	
    	
    # 	return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]
    	margins = np.max(np.abs([p_l, p_t, p_r, p_b]))
    	return [-margins, margins, margins, -margins]
    
    def predict(self, motion_data, overstyle, sess):
        # self.prediction.put_motion_data(motion_data)
        # return self.prediction.get_predicted_result()

        # no prediction
        prediction_time = 100  # ms
        
        input_euler = self.quat_to_euler(motion_data.head_orientation)
        input_euler = np.array([input_euler[1],input_euler[2],input_euler[0]])*180/np.pi
        self.input_gyro = np.array(motion_data.head_angular_velocity)*180/np.pi
        input_time = motion_data.timestamp/ 705600000
        # self.input_gyro = (input_euler/input_time).reshape(1,-1) * (np.pi/180)
        
        if (self.flag == 0):
            predicted_head_orientation = motion_data.head_orientation
            self.overfilling = motion_data.camera_projection
            ann = cap = crp = np.array(input_euler).reshape(1,-1)
        
        if (self.flag == 0 and self.train_counter >= 1300):
            self.eul_train = np.concatenate((self.eul_train, np.array(input_euler).reshape(1,-1)), axis =0)
            self.gyro_train = np.concatenate((self.gyro_train, np.array(self.input_gyro).reshape(1,-1)), axis =0)
            self.time_stamp = np.concatenate((self.time_stamp, np.array(input_time).reshape(1,-1)), axis =0)
            print(self.train_counter)
            if (self.train_counter == 1300+self.trainTime):
                self.flag, self.model, self.DELAY_SIZE, self.anticipation_size, self.midVel, self.avgVel = train_nn(self.eul_train, self.gyro_train, self.time_stamp)
                # print(self.DELAY_SIZE)
                # print(self.avgVel)
                self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.DELAY_SIZE, self.input_nm))
                self.y = self.model(self.x)
                self.pred_rt = np.zeros(shape=(self.anticipation_size,3), dtype= float)
        elif (self.flag == 1):# get the data after 17 seconds (72 FPS is assumed)
            
            ann, cap, crp, nop, self.x_seq_rt = pred(input_euler, self.input_gyro, input_time, self.x_seq_rt, self.timestamp_rt, self.eul_rt, self.gyro_rt, self.rt_counter, sess, self.model, self.DELAY_SIZE, self.y, self.x, self.midVel, self.avgVel)
            
            # self.pred_rt = np.concatenate((self.pred_rt, ann), axis =0)
            # print(self.pred_rt)
            # print(ann)
            
            self.pred_rt[:-1] = self.pred_rt[1:]
            self.pred_rt[-1] = ann[0]
            
            # print(ann)
            # print(input_euler)
            quat_quat_predict = self.euler_to_quat(ann)
            quat_quat_data = self.euler_to_quat(input_euler.reshape(1,-1))
            projection_input = motion_data.camera_projection
            
            diff = (ann - input_euler)*np.pi/180
            
            if (self.rt_counter<self.DELAY_SIZE+self.anticipation_size):
                margin = projection_input
            else:
                '''Ellipsoidal Overfilling'''
                if (overstyle == 'ellipse' or overstyle == 'step'):
                    idx = [0,3,1,2]
                    
                    if self.rt_counter >= self.DELAY_SIZE+self.anticipation_size:
                        if self.rt_tracked < self.err_win:
                            self.tracked_err[:-1] = self.tracked_err[1:]
                            self.tracked_err[-1] = self.pred_rt[0]-input_euler[0]
                                        
                            self.tracked_std = np.concatenate((self.tracked_std, [np.std(self.tracked_err[self.err_win-1-self.rt_tracked::], axis=0)]), axis = 0)
                            self.tracked_mean = np.concatenate((self.tracked_mean, [np.mean(self.tracked_err[self.err_win-1-self.rt_tracked::], axis=0)]), axis = 0)
                                        
                            self.rt_tracked += 1
                        else:
                            self.tracked_err[:-1] = self.tracked_err[1:]
                            self.tracked_err[-1] = self.pred_rt[0]-input_euler[0]
                                        
                            self.tracked_std  = np.concatenate((self.tracked_std, [np.std(self.tracked_err, axis=0)]), axis = 0)
                            self.tracked_mean = np.concatenate((self.tracked_mean, [np.mean(self.tracked_err, axis=0)]), axis = 0)
                    
                        ellipsoid_radius_99 = np.std(self.tracked_err, axis=0)*self.d99
                    
                        euler_pred_min = ann - ellipsoid_radius_99
                        euler_pred_max = ann + ellipsoid_radius_99
                        
                        quat_quat_data = self.euler_to_quat(input_euler.reshape(1,-1))
                        quat_quat_predict_max = self.euler_to_quat(euler_pred_max)
                        quat_quat_predict_min = self.euler_to_quat(euler_pred_min)
                    
                        hmd_orientation = quat_quat_data[:,idx]
                        hmd_max = quat_quat_predict_max[:, idx]
                        hmd_min = quat_quat_predict_min[:, idx]
                                    
                        input_orientation = np.quaternion(
                                        hmd_orientation[:,0],
                                        hmd_orientation[:,1],
                                        hmd_orientation[:,2],
                                        hmd_orientation[:,3]
                                    )
                                    
                        input_quat_max = np.quaternion(
                                        hmd_max[:,0],
                                        hmd_max[:,1],
                                        hmd_max[:,2],
                                        hmd_max[:,3]
                                        )
                    
                        input_quat_min = np.quaternion(
                                        hmd_min[:,0],
                                        hmd_min[:,1],
                                        hmd_min[:,2],
                                        hmd_min[:,3]
                                        )
                                    
                        pred_overhead_values = self.predict_overfilling(input_orientation, input_quat_min, input_quat_max, projection_input)
                    	
                        pred_overhead_values = np.array([pred_overhead_values[0]-np.sin(diff[0,2]),pred_overhead_values[1]+np.sin(diff[0,0]),pred_overhead_values[2]-np.sin(diff[0,2]),pred_overhead_values[3]+np.sin(diff[0,0])])
                    	
                        # self.ellips_values = np.vstack((self.ellips_values,pred_overhead_values))
		
                        '''IDO Mechanism'''
                        if (overstyle == 'step'):
                            for m in range (4):
                            	if ((abs(projection_input[m])-abs(pred_overhead_values[m]))>0):# Get error margin
                            		self.I[m] += 1
                            		if (self.I[m] > self.max_count): # Increase Margin
                            			if (m == 0 or m == 3):# Left and Bottom
                            				pred_overhead_values[m] -= self.icr
                            			else: # Top and Right
                            				pred_overhead_values[m] += self.icr
                            #					I[m] = 0
                            		self.D[m] = 0
                            	else:
                            		self.D[m] += 1
                            		if (self.D[m] > self.max_count): # Decrease margin
                            			if (m == 0 or m == 3):# Left and Bottom
                            				pred_overhead_values[m] += self.icr
                            			else: # Top and Right
                            				pred_overhead_values[m] -= self.icr
                            #					D[m] = 0
                            		self.I[m] = 0
                                
                            	# Clip on Maximum Value
                            	if (pred_overhead_values[m] < projection_input[m]-0.2):
                            		pred_overhead_values[m] = projection_input[m]-0.2
                            	elif (pred_overhead_values[m] > projection_input[m]+0.2):
                            		pred_overhead_values[m] = projection_input[m]+0.2
                            
                            # self.ido_values = np.vstack((self.ido_values,pred_overhead_values))
                        
                        margin = pred_overhead_values
                    
                    '''Optimal Overfilling'''
                elif (overstyle == 'optimal'):
                    idx = [0,3,1,2]
                    
                    hmd_orientation = quat_quat_data[:,idx]
                    frame_orientation = quat_quat_predict[:,idx]
                    
                    #Calculate overfiling
                    input_orientation = np.quaternion(
                        hmd_orientation[:,0],
                        hmd_orientation[:,1],
                        hmd_orientation[:,2],
                        hmd_orientation[:,3]
                        )
                    predict_orientation = np.quaternion(
                        frame_orientation[:,0],
                        frame_orientation[:,1],
                        frame_orientation[:,2],
                        frame_orientation[:,3]
                        )
                    
                    margin = self.calc_optimal_overhead(input_orientation, predict_orientation, projection_input)

                    '''Model-Based Overfilling'''
                elif (overstyle == 'model'):
                    margin = self.Robust_overfilling(input_euler, ann, projection_input, offset = 0.75, fixed_param = 1.6)
                else:
                    margin = self.make_camera_projection(motion_data, [0,0,0,0])

            
            print('\nInput [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(input_euler[0],input_euler[1],input_euler[2]))
            print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann[0][0],ann[0][1],ann[0][2]))
            print('Overfilling: {:.2f},{:.2f},{:.2f},{:.2f}'.format(margin[0],margin[1], margin[2], margin[3]))
            print('method: ' + overstyle)
            self.overfilling = [margin[0], margin[1], margin[2], margin[3]]

            self.rt_counter += 1
            
            predicted_head_orientation = quat_quat_predict[0]
            # predicted_head_orientation = quat_quat_data[0]
            # predicted_head_orientation = motion_data.head_orientation
            
            # print(quat_quat_predict[0])
            # print(quat_quat_data[0])
            # print(motion_data.head_orientation)
        
        self.train_counter += 1
        
        predicted_left_eye_pos = motion_data.left_eye_position
        predicted_right_eye_pos = motion_data.right_eye_position

        # overfilling delta in radian (left, top, right, bottom)
        #overfilling = [0, 0, 0, 0]
        #predicted_camera_projection = self.make_camera_projection(motion_data, overfilling)
        
        predicted_camera_projection = self.overfilling

        predicted_right_hand_pos = motion_data.right_hand_position
        predicted_right_hand_ori = motion_data.right_hand_orientation
        
        [foveation_inner_radius, foveation_middle_radius] = self.FoveationPatternUpdate(predicted_camera_projection, motion_data.camera_projection, innerRadii=0.25, middleRadii=0.35)
        
        # foveation_inner_radius = 0.25 #1.06
        # foveation_middle_radius = 0.35 #1.42

        return prediction_time, \
               predicted_left_eye_pos, \
               predicted_right_eye_pos, \
               predicted_head_orientation, \
               predicted_camera_projection, \
               foveation_inner_radius, \
               foveation_middle_radius, \
               predicted_right_hand_pos, \
               predicted_right_hand_ori,\
               input_euler, ann, cap, crp, self.flag

    def feedback_received(self, feedback):
        # see PrefMetricWriter.write_metric() to understand feedback values
        # (motion_prediction_server.py:320)
        
        # example : calculate overall latency
        #
        self.sum_overall_latency += feedback['endClientRender'] - feedback['gatherInput']
        self.count_overall_latency += 1

        if self.count_overall_latency >= 72:
            print("avg. overall latency: " + str(self.sum_overall_latency / self.count_overall_latency), flush=True)
            self.sum_overall_latency = 0
            self.count_overall_latency = 0
            
        # print("avg. overall latency: " + str(self.sum_overall_latency / self.count_overall_latency), flush=True)

        pass

    def external_input_received(self, input_data):
        #print(input_data, flush=True)
        pass
    
    def game_event_received(self, event):
        print(event, flush=True)
        pass
    
def main():
    app = App()
    app.run()

    
if __name__ == "__main__":
    main()

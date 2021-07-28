import math
import asyncio
import zmq
from abc import abstractmethod, ABCMeta
from zmq.asyncio import Context, Poller
import tensorflow as tf
from ._types import MotionData, PredictedData
from ._motion_data_transport import MotionDataTransport
from ._feedback_analyser import FeedbackAnalyser
from ._external_input import ExternalInput
from ._writer import PredictionOutputWriter, PerfMetricWriter, GameEventWriter
from ._prediction import BufferedNoPrediction


class PredictModule(metaclass=ABCMeta):
    @abstractmethod
    def predict(self, motion_data):
        pass

    @abstractmethod
    def feedback_received(self, feedback):
        pass

    @abstractmethod
    def external_input_received(self, input_data):
        pass

    @abstractmethod
    def game_event_received(self, event):
        pass

    def make_camera_projection(self, motion_data, overfilling):
        return [
            math.tan(math.atan(motion_data.camera_projection[0]) - overfilling[0]),
            math.tan(math.atan(motion_data.camera_projection[1]) + overfilling[1]),
            math.tan(math.atan(motion_data.camera_projection[2]) + overfilling[2]),
            math.tan(math.atan(motion_data.camera_projection[3]) - overfilling[3])
        ]


class MotionPredictServer:
    def __init__(self, module, port_input, port_feedback, prediction_output, metric_output, game_event_output, overstyle, accept_client_buttons):
        self.module = module
        self.port_input = port_input
        self.port_feedback = port_feedback
        self.accept_client_buttons = accept_client_buttons

        self.external_input = ExternalInput(self)
        self.motion_data_transport = MotionDataTransport(self)
        self.feedback_analyser = FeedbackAnalyser(self)        

        self.prediction_output = PredictionOutputWriter(
            prediction_output
        ) if prediction_output is not None else None
        
        self.metric_writer = PerfMetricWriter(
            metric_output
        ) if metric_output is not None else None

        self.game_event_writer = GameEventWriter(
            game_event_output
        ) if game_event_output is not None else None
        
        self.overstyle = overstyle

    def run(self):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        context = Context.instance()
        self.event_loop = asyncio.get_event_loop()

        print("Starting server on port {}...".format(self.port_input), flush=True)

        try:
            self.event_loop.run_until_complete(self.loop(context))
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        self.event_loop.close()
        
        self.motion_data_transport.perfResult()

        if self.prediction_output is not None:
            self.prediction_output.close()

        if self.metric_writer is not None:
            self.metric_writer.close()

        if self.game_event_writer is not None:
            self.game_event_writer.close()

    async def loop(self, context):
        poller = Poller()
        self.external_input.configure(context, poller, self.port_input + 2)
        self.motion_data_transport.configure(context, poller, self.port_input, self.port_input + 1, self.accept_client_buttons)
        self.feedback_analyser.configure(context, poller, self.port_feedback)

        with tf.compat.v1.Session() as sess:
            while True:
                events = await poller.poll(100)
                
                await self.external_input.process_events(events)
                await self.motion_data_transport.process_events(events, self.external_input, self.overstyle, sess)
                await self.feedback_analyser.process_events(events)

    # for motion data transport
    def pre_predict_motion(self, session):
        self.feedback_analyser.start_prediction(session)

    def predict_motion(self, motion_data, overstyle, sess):
        return self.module.predict(motion_data, overstyle, sess)

    def post_predict_motion(self, session):
        self.feedback_analyser.end_prediction(session)

    def write_prediction_output(self, motion_data, predicted_data):
        if self.prediction_output is None:
            return

        self.prediction_output.write(motion_data, predicted_data)

    def feedback_received(self, feedback):
        self.module.feedback_received(feedback)

        if self.metric_writer is not None:
            self.metric_writer.write_metric(feedback)

    def external_input_received(self, input_data):
        self.module.external_input_received(input_data)

    def game_event_received(self, event):
        self.module.game_event_received(event)

        if self.game_event_writer is not None:
            self.game_event_writer.write(event)


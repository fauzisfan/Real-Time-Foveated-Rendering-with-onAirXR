import zmq
import time
import cbor2

class FeedbackAnalyser:
    def __init__(self, owner):
        self.owner = owner
        self.feedbacks = {}

    def configure(self, context, poller, port):
        self.socket = context.socket(zmq.PULL)
        self.socket.bind("tcp://*:" + str(port))
        
        poller.register(self.socket, zmq.POLLIN)

    async def process_events(self, events):
        if self.socket not in dict(events):
            return

        data = await self.socket.recv()
        self.process_feedback(cbor2.loads(data))

    def start_prediction(self, session):
        assert(session not in self.feedbacks)
        self.feedbacks[session] = {
            'srcmask': 0,
            'startPrediction': time.process_time()
        }

    def end_prediction(self, session):
        self.feedbacks[session]['stopPrediction'] = time.process_time()

    def process_feedback(self, feedback):
        if not 'source' in feedback:
            return

        if feedback['source'] == 'gevt':
            self.owner.game_event_received(feedback)
        else:
            self.merge_feedback(feedback)

    def merge_feedback(self, feedback):
        if not all(key in feedback for key in ('session', 'source')):
            return

        if not feedback['session'] in self.feedbacks:
            return
        
        session = feedback['session']
        entry = self.feedbacks[session]
        
        if feedback['source'] == 'acli':
            entry['srcmask'] |= 0b01
        elif feedback['source'] == 'asrv':
            entry['srcmask'] |= 0b10
        else:
            return

        del feedback['source']
        self.feedbacks[session] = {**entry, **feedback}

        if entry['srcmask'] == 0b11:
            self.owner.feedback_received(self.feedbacks[session])
                
            self.feedbacks = {
                s: self.feedbacks[s] for s in self.feedbacks if s > session
            }
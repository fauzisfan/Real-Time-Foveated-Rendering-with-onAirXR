import zmq
from ._types import ExternalInputData

class ExternalInput:
    def __init__(self, owner):
        self.owner = owner
        self.states = {}

    def configure(self, context, poller, port):
        self.socket = context.socket(zmq.PULL)
        self.socket.bind("tcp://*:" + str(port))

        poller.register(self.socket, zmq.POLLIN)

    async def process_events(self, events):
        if self.socket not in dict(events):
            return
        
        frame = await self.socket.recv(0, False)
        input_data = ExternalInputData.from_bytes(frame.bytes)

        self.set_input(input_data)
            
    def get_input(self, input_id):
        if not input_id in self.states:
            return None

        return self.states[input_id]

    def set_input(self, input_data):
        if input_data.id not in self.states and \
            (not input_data.actual_press and not input_data.predicted_press):
            return
        
        if input_data.id in self.states and \
            self.states[input_data.id].StateEquals(input_data):
            return

        self.states[input_data.id] = input_data

        self.owner.external_input_received(input_data)

from dobot.lib.interface import Interface
from time import sleep
from typing import List, Optional, Tuple, Union

class MovementException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Controller:

    def __init__(self, port: Optional[str]):
        if port is None:
            port = '/dev/ttyACM0'
        self.bot = Interface(port)
        print('Bot status:', 'connected' if self.bot.connected() else 'not connected')

    
        
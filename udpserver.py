import socket
import threading


class UDPServer:
    """
    Simple UDP listener.

    Change from original: the data callback now receives TWO arguments:
        callback(data: bytes, addr: tuple[str, int])

    This lets the caller identify which device sent each packet using the
    sender's IP address — essential for routing packets to per-device state
    when multiple SensaGram devices are streaming simultaneously.

    The buffer size has also been raised from 1 024 B to 8 192 B.  A typical
    SensaGram JSON packet with full accelerometer stats is ~500–900 B, but
    GPS packets and packets with many sensors can approach 2 KB.  8 KB gives
    comfortable headroom while staying well below typical OS socket limits.
    """

    DEFAULT_BUFFER = 8192   # raised from 1 024

    def __init__(self, address, buffer_size=DEFAULT_BUFFER):
        self.sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.address     = address
        self.buffer_size = buffer_size
        self.callback    = None
        self._stop_flag  = False

    def setDataCallBack(self, callback):
        """
        Register the packet handler.

        Signature expected by server.py:
            callback(data: bytes, addr: tuple[str, int])
        """
        self.callback = callback

    def start(self):
        self._stop_flag = False
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop_flag = True
        # Unblock recvfrom by closing the socket.
        try:
            self.sock.close()
        except OSError:
            pass

    def _listen(self):
        self.sock.bind(self.address)
        while not self._stop_flag:
            try:
                data, addr = self.sock.recvfrom(self.buffer_size)
            except OSError:
                # Socket was closed by stop() — exit cleanly.
                break
            if self.callback and data:
                # addr is (ip_string, port_int) — e.g. ('192.168.1.5', 54321)
                self.callback(data, addr)

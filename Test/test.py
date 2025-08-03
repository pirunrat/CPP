import socket
import struct
import cv2
import numpy as np

def recvall(conn, count):
    """ Receive exact number of bytes """
    buf = b''
    while count:
        chunk = conn.recv(count)
        if not chunk:
            return None
        buf += chunk
        count -= len(chunk)
    return buf

sock = socket.socket()
sock.bind(('0.0.0.0', 5001))
sock.listen(1)
print("🟡 Waiting for C++ connection on port 5001...")
conn, addr = sock.accept()
print(f"✅ Connected by {addr}")

while True:
    # Receive 4 bytes (frame size)
    size_data = recvall(conn, 4)
    if not size_data:
        print("❌ Failed to receive frame size. Exiting.")
        break

    size = struct.unpack('!I', size_data)[0]
    print(f"📦 Expecting frame of size: {size} bytes")

    # Receive the actual image bytes
    img_data = recvall(conn, size)
    if not img_data:
        print("❌ Failed to receive frame data.")
        break

    print(f"✅ Received {len(img_data)} bytes")

    # Decode image
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        cv2.imwrite("received_frame.jpg", frame)
        print("🖼 Frame saved as received_frame.jpg")
    else:
        print("❌ Failed to decode frame")

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
conn.close()

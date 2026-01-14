import zmq
from datetime import datetime

# ==============================
# CONFIG
# ==============================

ZMQ_RAWTX = "tcp://127.0.0.1:28332"

# ==============================
# INIT ZMQ
# ==============================

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(ZMQ_RAWTX)

# Subscribe to all raw tx messages
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("\nCP1 ZMQ RAWTX LISTENER (MAINNET)")
print("Listening on:", ZMQ_RAWTX)
print("Press Ctrl+C to stop\n")

# ==============================
# MAIN LOOP
# ==============================

while True:
    try:
        raw_msg = socket.recv()

        # rawtx message is raw binary tx
        raw_hex = raw_msg.hex()
        size_bytes = len(raw_msg)

        print(
            f"[{datetime.utcnow().isoformat()}] "
            f"Received TX | size={size_bytes} bytes"
        )

    except KeyboardInterrupt:
        print("\n🛑 ZMQ listener stopped cleanly.")
        break

    except Exception as e:
        print("ZMQ runtime error:", str(e))

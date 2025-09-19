import asyncio, re, sys, time
from wakeonlan import send_magic_packet
from aiowebostv import WebOsClient

# ==== EDIT THESE ====
TV_MAC = "AA:BB:CC:DD:EE:FF"         # TV's MAC address (from TV network info)
BROADCAST_IP = "192.168.1.255"       # your LAN broadcast address
TV_IP = "192.168.1.123"              # TV's IP (static/DHCP reservation recommended)
# =====================

def extract_yt_id(s: str) -> str:
    # Accept a full URL, shorts, share links, or a bare 11-char ID
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)
    s = s.strip()
    return s[-11:] if re.fullmatch(r"[A-Za-z0-9_-]{11}", s) else s

async def bringup_and_play(video_arg: str):
    vid = extract_yt_id(video_arg)
    # 1) Wake the TV
    send_magic_packet(TV_MAC, ip_address=BROADCAST_IP)
    # 2) Wait for the TV to boot & accept websocket connections
    client = None
    for _ in range(45):  # up to ~45s
        try:
            client = await WebOsClient.create(TV_IP, ping_interval=0)
            break
        except Exception:
            await asyncio.sleep(1)
    if not client:
        print("❌ TV did not come online in time.")
        return
    # First run will prompt a pairing popup on the TV; accept it with the remote.

    # 3) Launch YouTube deep-linked to the video
    # App id 'youtube.leanback.v4', deep-link with contentId=<VIDEO_ID>
    await client.launch_app("youtube.leanback.v4", params={"contentId": vid})
    await asyncio.sleep(2)
    await client.disconnect()
    print(f"✅ Asked TV to open YouTube video: {vid}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lg_youtube.py <YouTube URL or VIDEO_ID>")
        sys.exit(1)
    asyncio.run(bringup_and_play(sys.argv[1]))

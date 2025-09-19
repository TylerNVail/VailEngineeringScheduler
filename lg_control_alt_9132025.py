import json, time, socket, os, traceback, re, sys
from contextlib import closing
from pywebostv.connection import WebOSClient

CONFIG  = "config.json"
KEYFILE = "client_key_pywebostv.json"   # created after first pairing

def _log(s: str): print(f"[LG] {s}")

def _cfg():
    with open(CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)

def _send_wol(mac: str):
    try:
        mac_bytes = bytes.fromhex(mac.replace(":", "").replace("-", ""))
        pkt = b"\xff"*6 + mac_bytes*16
        with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.sendto(pkt, ("255.255.255.255", 9))
        _log("Sent Wake-on-LAN magic packet.")
    except Exception as e:
        _log(f"WoL send failed: {e}")

def _ports_up(ip: str, timeout=0.6) -> bool:
    for port in (3000, 3001):
        try:
            with closing(socket.create_connection((ip, port), timeout=timeout)):
                return True
        except OSError:
            pass
    return False

def _load_store():
    if os.path.exists(KEYFILE):
        try:
            return json.load(open(KEYFILE, "r", encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_store(store):
    json.dump(store, open(KEYFILE, "w", encoding="utf-8"))

def _register_with_retries(ip: str, tries=12, sleep_s=3.0) -> WebOSClient:
    last = None
    for attempt in range(1, tries + 1):
        try:
            client = WebOSClient(ip)
            store  = _load_store()
            if not store:
                _log("No pairing key found — will request pairing.")
            for status in client.register(store):
                if status == WebOSClient.PROMPTED:
                    _log("Please accept the pairing popup on the TV…")
                elif status == WebOSClient.REGISTERED:
                    _save_store(store)
                    _log("Registered with TV and saved key.")
            return client
        except OSError as e:
            last = e
            _log(f"register() attempt {attempt}/{tries} failed: {e}")
            time.sleep(sleep_s)
        except Exception as e:
            last = e
            _log(f"register() attempt {attempt}/{tries} error: {e}")
            time.sleep(sleep_s)
    raise RuntimeError(f"Unable to register with TV after retries: {last}")

def _extract_video_id_and_time(arg: str):
    if not arg:
        return None, None
    url = arg.strip()
    t = None
    m = re.search(r"[?&#]t=([0-9]+)s?\b", url)
    if not m:
        m2 = re.search(r"[?&#]t=(?:(\d+)m)?(?:(\d+)s)?", url)
        if m2:
            mins = int(m2.group(1) or 0)
            secs = int(m2.group(2) or 0)
            t = mins * 60 + secs
    else:
        t = int(m.group(1))
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1), t
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url, t
    return None, t

def _open_youtube_deeplink(client: WebOSClient, video_id: str, start_s: int | None) -> bool:
    try:
        payload = {
            "id": "youtube.leanback.v4",
            "params": {"contentId": video_id}
        }
        if start_s:
            payload["params"]["startTime"] = int(start_s)
        client.request("ssap://system.launcher/launch", payload)
        _log("Deep-linked via youtube.leanback.v4")
        return True
    except Exception as e:
        _log(f"Leanback deep-link failed: {e}")
    try:
        payload = {
            "id": "com.webos.app.youtube",
            "params": {"contentId": video_id}
        }
        if start_s:
            payload["params"]["startTime"] = int(start_s)
        client.request("ssap://system.launcher/launch", payload)
        _log("Deep-linked via com.webos.app.youtube")
        return True
    except Exception as e:
        _log(f"com.webos.app.youtube deep-link failed: {e}")
    try:
        target = f"https://www.youtube.com/tv#/watch?v={video_id}"
        if start_s:
            target += f"&t={int(start_s)}s"
        client.request("ssap://system.launcher/open", {"target": target})
        _log("Opened in browser as fallback.")
        return True
    except Exception as e:
        _log(f"Browser fallback failed: {e}")
        return False

def run_sequence(override_url: str | None = None):
    cfg = _cfg()
    ip      = cfg["tv_ip"]
    mac     = cfg["tv_mac"]
    wake_first         = bool(cfg.get("wake_first", True))
    post_wake_delay    = float(cfg.get("post_wake_delay_sec", 12.0))
    pre_deeplink_wait  = float(cfg.get("pre_deeplink_wait_sec", 2.0))

    url = override_url if override_url else cfg.get("video_url")
    vid, start_s = _extract_video_id_and_time(url or "")

    try:
        if wake_first:
            _send_wol(mac)
        start = time.monotonic()
        while time.monotonic() - start < 120:
            if _ports_up(ip):
                _log("WebOS ports responding (3000/3001).")
                break
            time.sleep(1.2)
        else:
            raise RuntimeError("TV did not come online (ports closed).")

        _log(f"Waiting {post_wake_delay:.1f}s after wake…")
        time.sleep(post_wake_delay)

        client = _register_with_retries(ip)

        if vid:
            if pre_deeplink_wait > 0:
                time.sleep(pre_deeplink_wait)
            ok = _open_youtube_deeplink(client, vid, start_s)
        else:
            app_id = cfg.get("youtube_app_id", "com.webos.app.youtube")
            client.request("ssap://system.launcher/open", {"id": app_id})
            ok = True
            _log(f"Opened {app_id}")

        if not ok:
            _log("Could not deep-link to YouTube video.")
        else:
            _log("✅ Done. (YouTube should be in the foreground.)")

    except Exception as e:
        _log(f"EXCEPTION: {e}")
        _log(traceback.format_exc())
        raise

if __name__ == "__main__":
    override = sys.argv[1] if len(sys.argv) > 1 else None
    run_sequence(override)

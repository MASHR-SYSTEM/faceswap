"""FaceSwap desktop launcher; device ownership belongs to the child server."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


def probe(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/capabilities', timeout=1) as response:
            return json.load(response).get('application') == 'mashr-faceswap'
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Start and stop without opening a window or devices')
    args = parser.parse_args()
    if args.server:
        # The distributed entrypoint always binds to loopback.
        os.environ['FACESWAP_HOST'] = '127.0.0.1'
        from faceswap.app import main as serve
        serve()
        return
    port = int(os.environ.get('FACESWAP_PORT', '7865'))
    url = f'http://127.0.0.1:{port}'
    child = None
    log = None
    try:
        if not probe(port):
            from faceswap.settings import application_data_dir
            logs = application_data_dir() / '.cache/faceswap'
            logs.mkdir(parents=True, exist_ok=True)
            log = (logs / 'launcher.log').open('ab')
            prefix = [sys.executable] if getattr(sys, 'frozen', False) else [sys.executable, str(Path(__file__).resolve())]
            child = subprocess.Popen(prefix + ['--server'], stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            for _ in range(120):
                if child.poll() is not None:
                    raise RuntimeError(f'FaceSwap could not start. See {logs / "launcher.log"}. Port {port} may be in use.')
                if probe(port):
                    break
                time.sleep(.25)
            else:
                raise RuntimeError(f'FaceSwap did not become ready. See {logs / "launcher.log"}')
        if args.smoke:
            print('FaceSwap startup smoke passed')
            return
        import tkinter as tk
        root = tk.Tk()
        root.title('MASHr FaceSwap')
        root.geometry('380x170')
        tk.Label(root, text='FaceSwap runs on this computer.\nClosing the browser does not stop it.').pack(pady=15)
        tk.Button(root, text='Open FaceSwap', command=lambda: webbrowser.open(url)).pack(pady=5)
        def exit_app():
            try:
                request = urllib.request.Request(url + '/api/session/stop-all', data=b'{}',
                    headers={'Content-Type': 'application/json'}, method='POST')
                urllib.request.urlopen(request, timeout=15).close()
            except OSError:
                pass
            root.destroy()
        tk.Button(root, text='Exit' if child else 'Close launcher', command=exit_app if child else root.destroy).pack(pady=5)
        root.protocol('WM_DELETE_WINDOW', exit_app if child else root.destroy)
        webbrowser.open(url)
        root.mainloop()
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if log:
            log.close()


if __name__ == '__main__':
    main()

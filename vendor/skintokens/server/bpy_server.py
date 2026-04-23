from bottle import request, response

import bottle
import queue
import threading

from .spec import bytes_to_object, object_to_bytes, BPY_PORT

from ..rig_package.parser.bpy import BpyParser, transfer_rigging

def run():
    path_queue = queue.Queue()
    result_queue = queue.Queue()
    
    app = bottle.Bottle()
    
    @app.route('/load', method='GET') # type: ignore
    def load():
        data = request.body.read() # type: ignore
        path_queue.put(('load', data))
        res = result_queue.get()
        payload = object_to_bytes(res)
        response.content_type = 'application/octet-stream'  # type: ignore
        return payload
    
    @app.route('/ping', method='GET') # type: ignore
    def ping():
        return 'pong'
    
    @app.route('/export', method='post') # type: ignore
    def export():
        data = request.body.read() # type: ignore
        path_queue.put(('export', data))
        res = result_queue.get()
        payload = object_to_bytes(res)
        response.content_type = 'application/octet-stream'  # type: ignore
        return payload
    
    @app.route('/transfer', method='post') # type: ignore
    def transfer():
        data = request.body.read() # type: ignore
        path_queue.put(('transfer', data))
        res = result_queue.get()
        payload = object_to_bytes(res)
        response.content_type = 'application/octet-stream'  # type: ignore
        return payload
    
    def run_server(): bottle.run(app, host='0.0.0.0', port=BPY_PORT, server='tornado')
    threading.Thread(target=run_server, daemon=False).start()
    
    while True:
        d = path_queue.get()
        op = d[0]
        data = bytes_to_object(d[1])
        if op == 'load':
            print("[SERVER] received load path:", data)
            asset = BpyParser.load(data)
            result_queue.put(asset)
        elif op == 'export':
            print("[SERVER] received export path:", data['filepath'])
            BpyParser.export(**data)
            result_queue.put('ok')
        elif op == 'transfer':
            print("[SERVER] received transfer path:", data['target_path'])
            transfer_rigging(**data)
            result_queue.put('ok')
        else:
            result_queue.put(f"unsupported op: {str(op)}")
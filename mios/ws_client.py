import json
import websockets
import asyncio
import socket
from concurrent.futures import TimeoutError as ConnectionTimeoutError
import websockets.exceptions


class Client:
    def __init__(self, hostname, port, endpoint):
        uri = "ws://" + hostname + ":" + str(port) + "/" + endpoint
        self.connection = websockets.connect(uri).ws_client

    def send(self, request):
        message = json.dumps(request)
        self.connection.ws_client.send(message)
        response = asyncio.wait_for(websocket.recv(), timeout=timeout)
        return json.loads(response)


async def send(hostname, port=12000, endpoint="mios/core", request=None, timeout=100, silent=False):
    uri = "ws://" + hostname + ":" + str(port) + "/" +endpoint
    try:
        async with websockets.connect(uri, close_timeout=1000) as websocket:
            message = json.dumps(request)
            await websocket.send(message)
            response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            return json.loads(response)
    except ConnectionRefusedError as e:
        if silent is False:
            print("ConnectionRefusedError: ")
            print(e)
            print("Hostname: " + hostname + ", port: " + str(port) + ", endpoint: " + endpoint)
        return None
    except ConnectionResetError as e:
        if silent is False:
            print("ConnectionResetError: ")
            print(e)
            print("Hostname: " + hostname + ", port: " + str(port) + ", endpoint: " + endpoint)
        return None
    except ConnectionAbortedError as e:
        if silent is False:
            print("ConnectionAbortedError: ")
            print(e)
            print("Hostname: " + hostname + ", port: " + str(port) + ", endpoint: " + endpoint)
        return None
    except websockets.ConnectionClosedError as e:
        if silent is False:
            print("ConnectionClosedError: ")
            print(e)
            print("Hostname: " + hostname + ", port: " + str(port) + ", endpoint: " + endpoint)
        return None
    except ConnectionTimeoutError as e:
        if silent is False:
            print("ConnectionTimeoutError: ")
            print(e)
            print("Hostname: " + hostname + ", port: " + str(port) + ", endpoint: " + endpoint)
        return None
    except websockets.exceptions.InvalidMessage as e:
        if silent is False:
            print("InvalidMessage: ")
            print(e)
            print("Hostname: " + hostname + ", port: " + str(port) + ", endpoint: " + endpoint)
        return None


def call_server(hostname, port, endpoint, request, timeout):
    asyncio.set_event_loop(asyncio.new_event_loop())
    return asyncio.get_event_loop().run_until_complete(send(hostname, request=request, port=port,
                                                            endpoint=endpoint, timeout=timeout))


def call_method(hostname: str, port: int, method, payload=None, endpoint="mios/core", timeout=100, silent=False):
    try:
        request = {
            "method": method,
            "request": payload
        }
        asyncio.set_event_loop(asyncio.new_event_loop())
        return asyncio.get_event_loop().run_until_complete(send(hostname, request=request, port=port,
                                                                endpoint=endpoint, timeout=timeout, silent=silent))
    except socket.gaierror as e:
        print(e)
        print("Hostname: " + hostname + ", port:" + str(port) + ", endpoint: " + endpoint)
        return None


def start_task(hostname: str, task: str, parameters={}, queue=False):
    payload = {
        "task": task,
        "parameters": parameters,
        "queue": queue
    }
    return call_method(hostname, 12000, "start_task", payload)


def stop_task(hostname: str, raise_exception=False, recover=False, empty_queue=False):
    payload = {
        "raise_exception": raise_exception,
        "recover": recover,
        "empty_queue": empty_queue
    }
    return call_method(hostname, 12000, "stop_task", payload)


def wait_for_task(hostname: str, task_uuid: str):
    payload = {
        "task_uuid": task_uuid
    }
    return call_method(hostname, 12000, "wait_for_task", payload)


def start_task_and_wait(hostname, task, parameters, queue=False):
    response = start_task(hostname, task, parameters, queue)
    response = wait_for_task(hostname, response["result"]["task_uuid"])
    return response


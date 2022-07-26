# import http.server


# http.server.test(http.server.SimpleHTTPRequestHandler, port=8000)



#!/usr/bin/env python


import asyncio
import websockets
from websockets import serve
from threading import Thread



async def handler(websocket):
    async for message in websocket:
        print(message)


async def main():
    async with websockets.serve(handler, "", 8500):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())


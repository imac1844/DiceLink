import asyncio
import websockets

async def connect_stop():
    async with websockets.connect("ws://localhost:8000") as websocket:
        await websocket.send('/stop')
        print(await websocket.recv())

asyncio.run(connect_stop())
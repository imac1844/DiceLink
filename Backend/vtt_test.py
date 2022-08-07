import asyncio
import websockets

async def connect_register_read():
	async with websockets.connect("ws://localhost:8000") as websocket:
		await websocket.send('/reg_vtt')
		print(await websocket.recv())
		await websocket.send('/read')
		print(await websocket.recv())
		print(await websocket.recv())


asyncio.run(connect_register_read())
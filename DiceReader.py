
"""
DiceLink
Version PA.1.2
by imac1844



This application serves as the main functionality of DiceLink. It takes a raw video input from a webcam, and starts a video stream.
Each frame creates an object of the DiceConnector class, which is analyzed for dice. A websocket is opened as well, for future 
integration with Foundry VTT. A tool for assessing contrast in image processing, and an optional GUI interface are also provided.

Some parts should be replaced with asynchronos functions

"""
import time
import numpy as np
import cv2
import imutils
import math
from threading import Thread
import tkinter as tk
import os, sys
import asyncio
from websockets import connect, serve
import websockets
import subprocess


class DiceLink:
	''' 
	DiceLink provides a wrapper to run the processes that make up the full application.

	'''
	def __init__(self):

		# Control Flags
		self.FLAG_stop = False
		self.FLAG_read = False

		# The video feed and websocket are initialized and started as threads. 
		self.Webcam = self.VideoCapture(self)
		self.Socket = self.DiceLinkSocketHost(self)
		self.videoThread = Thread(target = self.Webcam.VCdisplay)
		self.serverThread = Thread(target = self.Socket.start)
		self.controlThread = Thread(target = self.main)
		self.videoThread.start()
		self.serverThread.start()
		self.controlThread.start()

		# self.TK_interface() # TKinter GUI Interface. Todo: integrate as an option.
		self.controlThread.join()
		# self.main() # Main thread runs in this loop

	def main(self):
		# Simple loop to check flags. Todo: integrate flag systems all into DiceLink, reorder classes to be subclasses of this class.
		while True:
			if self.FLAG_stop:
				self.quit_funct()
				break
			# if self.FLAG_read:
			# 	self.read_funct()
			# 	self.Socket.dice = self.Webcam.connection.dice
			# 	self.Socket.read_flag = False
			time.sleep(1)

	def TK_interface(self):
		# TKinter GUI settings. Sets up a very simple GUI to trigger reads or shutdown.
		self.interface_TK = tk.Tk()
		self.interface_TK.title('DiceLink Interface')
		self.DiceButtonWindow = tk.Canvas(self.interface_TK, width=250, height=100, bd = 15, bg = 'cyan')
		self.DiceButtonWindow.grid(columnspan=2, rowspan = 1)

		self.button_quit = tk.Button(width = 10, height = 2, text = 'Exit', command = lambda: setattr(self, 'FLAG_stop', True))
		self.button_quit.grid(row = 0, column = 0)

		self.button_read = tk.Button(width = 10, height = 2, text = 'Read', command = lambda: setattr(self, 'FLAG_read', True))
		self.button_read.grid(row = 0, column = 1)

		self.interface_TK.mainloop()

	def quit_funct(self):
		print('Stopping...')
		try: self.videoThread.join()
		except RuntimeError: pass
		print('Video stopped')
		try: self.serverThread.join()
		except (RuntimeError): pass
		print('Server closed')
		try: self.interface_TK.destroy()
		except: pass
		print('Interface closed')
		print('Shutdown Successful')


	class DiceConnector:
		'''
		Main functionality. Called every cycle of the loop in the Video thread of DiceLink. Takes an image (as a numpy array) 
		and the integer frame number, though the frame number is unused. The image_processor method finds the dice in the image, 
		then the list_dice method creates a list of Dice objects. The Video Feed is then updated with bounding boxes drawn on the dice.

		To Add: correction method
		'''
		def __init__(self, Capture, frame_i, ThreshTest=False, ShowSteps=False):
			self.frame_i = frame_i
			self.Capture = Capture 						# Todo: rotate as an option #cv2.rotate(Capture, cv2.ROTATE_180) 		
			self.ThreshTest = ThreshTest				# Set True to enable the Thresholding test
			self.show = ShowSteps 						# Set True to show working images
			self.defaultThresh = 200					# CHANGE ME: If your results are bad, to the thresh test and set this value
			self.max_dice = 5 							# Temp, but sets the number at which to stop looking.

			self.threshold_a = 0 						# Used to initialize the thresholing test
			self.wrk_img = 0 							# Used in image_processor to make iterative changes. Maybe convert to local?
			self.diceFindSize = (8000, 30000)			# Magic number, bounds the area (in pixels) of "good" dice during image processing
			self.diceOverlapSearchRadius = 10 			# Stops doubling by removing bounding boxes that overlap too closely.
			self.diceList = []
			self.dice = []								# dice as Dice objects

			self.image_processor()						# coords of die bounding boxsbes, in form ([coords], center). 

		def drawboxes(self):
			for die, center in (self.diceList):
				cv2.drawContours(self.Capture.img, [die], -1, (0,255,0), 3)

		def on_change(self, value=100):
			# Controls the slider value of the threshold test. Should be nested deeper in a thresh test class
			image_copy = self.wrk_img.copy()
			self.threshold_a = value
			_, image_copy = cv2.threshold(image_copy, self.threshold_a, 255, cv2.THRESH_BINARY)
			cv2.imshow('Threshold Test(a)', image_copy)

		def image_processor(self):
			# Takes the raw captured image and finds the location of any dice, using canny edge detection.
			# First, the image is processed for binary thresholding
			raw_img = self.Capture.img
			self.wrk_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
			self.wrk_img = cv2.GaussianBlur(self.wrk_img, (11,11), 1000)

			# Runs the manual threshold set slider for calibration.
			if self.ThreshTest:
				cv2.imshow('Threshold Test(a)', self.wrk_img)
				a = cv2.createTrackbar("Thresh a", "Threshold Test(a)", 0, 255, self.on_change)
				cv2.waitKey(0)
			else:
				self.threshold_a = self.defaultThresh
			_, self.wrk_img = cv2.threshold(self.wrk_img, self.threshold_a, 255, cv2.THRESH_BINARY)

			# Edge Detection
			canny_img = cv2.Canny(self.wrk_img, 1, 50, 2)
			contours, _ = cv2.findContours(canny_img, 1, 2)

			# Finding the minimum area rotated rectangle for each contour
			minRect = [None]*len(contours)
			for i, c in enumerate(contours):
				minRect[i] = cv2.minAreaRect(c)

			#variables declared, all empty. These are used for sorting.
			pyBox = []
			sortedbox = []
			centersList = []
			self.diceList = []

			# Sorts contours from largest area to smallest area
			SortedContours = sorted(contours, key=cv2.contourArea, reverse=True)

			for i, c in enumerate(SortedContours):
				# Creates a rotated rectangle, not kept between iterations
				box = cv2.boxPoints(minRect[i])
				box = np.intp(box)
				center = (int(minRect[i][0][0]),int(minRect[i][0][1]))
				centersList.append(center)
				
				# Removes duplicates
				inList = False
				for k in range(0, len(centersList)-1):
					if ((centersList[k][0] - center[0])**2 + (centersList[k][1] - center[1])**2) < self.diceOverlapSearchRadius**2:
						inList = True
						centersList.pop()
						break

				# Sorts a copy of box, and keeps the sorted boxes in a list
				pyBox.append([])
				for j in range(0,4):
					pyBox[i].append([])
					pyBox[i][j].append(box[j][0])
					pyBox[i][j].append(box[j][1])
				sortedbox.append(sorted(pyBox[i]))

				# Determines size and ratio of sides for each box
				dx = sortedbox[i][2][0] - sortedbox[i][0][0]
				dy = sortedbox[i][1][1] - sortedbox[i][0][1]
				boxArea = abs(dx*dy)

				# Saves boxes that meet the criteria for size and ratio
				if 0.75 < abs(dy/(dx+0.001)) < 1.5: #+0.01 is for 0 catching
					if self.diceFindSize[0] < boxArea < self.diceFindSize[1]:
						if not inList:
							self.diceList.append([box, center])
							if self.show:
								cv2.drawContours(drawing, [box], 0, green)
								cv2.circle(drawing, center, 0, green, -1)	



			# Does some trig to turn the list of dice coordinates/rotations into separate images of the dice themselves.

			croppedDice = []
			ogImgCenter = self.Capture.centerPoint

			for i, (coords, center) in enumerate(self.diceList):
				# The image is rotated so the die bounding box is parallel with the window.
				# WARNING: 
				# <TRIGONOMETRY> 
				dieRotRatio = (coords[1][0]-coords[0][0])/(coords[1][1]-coords[0][1]+0.000001) # The +0.000001 prevents div-by-zero.
				dieRotRad = math.atan(dieRotRatio)
				dieRotDeg = math.degrees(dieRotRad)
				die = imutils.rotate_bound(self.Capture.img, dieRotDeg)
				newImgCenter = [(len(die)/2), ((len(die[0])/2))]

				# The coordinate system has changed now that the image is rotated, so the bounding box is converted to the new coordinates
				newCorners = []
				for j in 1,3:
					newx = int(((coords[j][0]-ogImgCenter[0])*math.cos(dieRotRad)) - (coords[j][1]-ogImgCenter[1])*math.sin(dieRotRad) + newImgCenter[1])
					newy = int(((coords[j][1]-ogImgCenter[1])*math.cos(dieRotRad)) + (coords[j][0]-ogImgCenter[0])*math.sin(dieRotRad) + newImgCenter[0])
					newCorners.append([newx,newy])

				# The new coordinates are put in order, then the die image is cut out and added to the self.dice local variable, which is then returned
				# when all the dice are done.
				arrangedCorners = [sorted([newCorners[0][1],newCorners[1][1]]),sorted([newCorners[0][0],newCorners[1][0]])] #[[y,y1],[x,x1]]
				croppedDie = die[arrangedCorners[0][0]:arrangedCorners[0][1], arrangedCorners[1][0]:arrangedCorners[1][1]]

				self.dice.append(self.Dice(croppedDie, center))
				# </TRIGONOMETRY>

		def feature_match(self):
			# Matches dice to reference images using cv2 feature matching
			dice = []
			for die in self.dice:
				GrayImg = cv2.cvtColor(die.img, cv2.COLOR_BGR2GRAY)

				# <Mostly a shameless rip from stackoverflow>
				orb = cv2.ORB_create()
				bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
				kp_target, des_target = orb.detectAndCompute(GrayImg,None)

				# Compares the image to every reference image, and returns the file name and path
				for root, dirs, files in os.walk(".\\DiceSets\\", topdown=True):
		   			for name in files:
		   				pathname = os.path.join(root, name)
		   				ref_img = cv2.imread(pathname, cv2.COLOR_BGR2GRAY)
		   				# cv2.imshow()
		   				try:
		   					kp_ref, des_ref = orb.detectAndCompute(ref_img,None)
		   				except cv2.error:
		   					continue

		   				# print(type(des_target), "\n", type(des_ref))
		   				matches = bf.match(des_target, des_ref)
		   				matches = sorted(matches, key = lambda x:x.distance)

		   				top5 = 0
		   				for i in range (0,4):
		   					top5 += matches[i].distance
		   				die.matchlist.append((pathname, top5))

		   			# </Mostly a shameless rip from stackoverflow>

				die.matchlist = sorted(die.matchlist, key = lambda x:x[1])
				die.results() # die is a Dice object
				dice.append(die)
			return(dice)

		class Dice:
			'''
			Contains information about a die
			'''
			def __init__(self, raw_img, center):
				self.center = center
				self.img = raw_img
				self.frameWidth = len(self.img[0])
				self.frameHeight = len(self.img)
				self.matchlist = []


				self.value = None
				self.type = None

			def results(self):
				#  self.matchlist[0][0] is the pathname to the best matching image
				raw_result = self.matchlist[0][0].split('\\')
				self.value = raw_result[-1][:-4]
				self.type = raw_result[-2]

				print(self.type, "rolled", self.value )

	class VideoCapture:
		def __init__(self, parentObject):
			'''
			Runs the video feed, which creates a DiceConnector object every frame to find and box the dice
			'''
			self.parentObject = parentObject

			self.frameWidth = 640
			self.frameHeight = 480
			self.centerPoint = (self.frameWidth/2, self.frameHeight/2)
			self.capture = cv2.VideoCapture(1)
			self.capture.set(3, self.frameWidth)
			self.capture.set(4, self.frameHeight)
			self.img = 0 							# Defined in the "VCdisplay" method every frame

			self.connection = None 					# Defined in the "display" method every nth frames, as per connection_timer
			self.connection_timer = 10 				# 

		async def send_read_data(self, raw_data):
			async with websockets.connect("ws://localhost:{}".format(self.parentObject.Socket.port)) as WebSocLocalClient:
				pkg = '!'
				await WebSocLocalClient.send('/reg_local')
				await WebSocLocalClient.recv()
				for i, die in enumerate(raw_data):
					pkg = pkg + 'die{}: '.format(i) + '{' + 'type: {}, value: {}'.format(die.type, die.value) + "}"
					if i != len(raw_data)-1:
						pkg = pkg + ', '
				if pkg == '!':
					await WebSocLocalClient.send('No dice to read!')
				await WebSocLocalClient.send(pkg)
				await WebSocLocalClient.close()
				self.parentObject.FLAG_read = False

		def VCdisplay(self):
			# Main Loop. Todo: integrate the flags.
			print("Display loading...")
			frame = 0
			while True:
				# time.sleep(1)
				frame += 1
				_, img = self.capture.read()
				self.img = img
				if self.parentObject.FLAG_stop:
					break
				if frame%self.connection_timer == 1:
					self.connection = self.parentObject.DiceConnector(self, frame)
				self.connection.drawboxes()
				if self.parentObject.FLAG_read:
					output = self.connection.feature_match()
					asyncio.run(self.send_read_data(output))
				cv2.imshow("Dice Link", img)
				cv2.waitKey(1)

			self.capture.release()
			cv2.destroyAllWindows()

	class DiceLinkSocketHost:
		def __init__(self, parentObject, port=8000):
			'''
			Runs the Websocket for passing commands and information between this application and the foundryVTT side.
			'''
			self.parentObject = parentObject
			self.port = port

			self.websocket = None # defined in handler()

			self.VTTClient = None
			self.LocalClient = None
			
		def start(self):
			print("Websocket running at ws://localhost:{}/".format(self.port))
			asyncio.run(self.main())

		async def shutdown(self):
			while not self.parentObject.FLAG_stop:
				await asyncio.sleep(1)
			try: await self.websocket.send("Socket shutting down...")
			except: 
				pass

		async def handler(self, websocket):
			self.websocket = websocket
			async for message in self.websocket:
				""" Command Handler Block """
				if message == "":
					continue
				if message[0] == "/":
					match message:
						case '/reg_vtt':
							self.VTTClient = websocket
							await self.websocket.send('Registered as VTT Client')
						case '/reg_local':
							self.LocalClient = websocket
							await self.websocket.send('Registered as Local Client')
						case "/read":
							self.parentObject.FLAG_read = True
							try: await self.websocket.send("READ recieved")
							except websockets.ConnectionClosedError:
								pass
						case '/stop':
							self.parentObject.FLAG_stop = True
							await self.websocket.send("STOP recieved")
						case _:
							await self.websocket.send(message)

				if message[0] == "!":
					try: await self.VTTClient.send(message)
					except websockets.ConnectionClosedError:
						pass
				else: print(message)
				


		async def main(self):
			async with websockets.serve(self.handler, "", self.port):
				await self.shutdown()  # run for a time




if __name__ == '__main__':
	DiceLink()

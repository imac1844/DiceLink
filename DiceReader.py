
"""

Important to work on:

 - self.diceOverlapSearchRadius and self.diceFindSize in DiceConnector __init__ are manually assigned and should be dynamic

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



# Debugging info
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
MAGENTA = (255,0,255)
CYAN = (255,255,0)
YELLOW = (0,255,255)
WHITE = (255,255,255)


class DC:
	def __init__(self):
		self.Webcam = VideoCapture()
		self.Socket = DiceLinkSocketHost(self)
		self.videoThread = Thread(target = self.Webcam.display)
		self.serverThread = Thread(target = self.Socket.start)

		self.videoThread.start()
		self.serverThread.start()



		# self.buttons()

		self.main()

	def main(self):
		while True:
			if self.Socket.stop_flag == True:
				self.quit_funct()
				break
			if self.Socket.read_flag == True:
				self.read_funct()
				self.Socket.dice = self.Webcam.connection.dice
				self.Socket.read_flag = False
			time.sleep(1)



	def buttons(self):

		self.interface_TK = tk.Tk()
		self.interface_TK.title('DiceLink Interface')
		self.DiceButtonWindow = tk.Canvas(self.interface_TK, width=250, height=100, bd = 15, bg = 'cyan')
		self.DiceButtonWindow.grid(columnspan=2, rowspan = 1)

		self.button_quit = tk.Button(width = 10, height = 2, text = 'Exit', command = self.quit_funct)
		self.button_quit.grid(row = 0, column = 0)

		self.button_read = tk.Button(width = 10, height = 2, text = 'Read', command = self.read_funct)
		self.button_read.grid(row = 0, column = 1)

		self.interface_TK.mainloop()

	def read_funct(self):
		self.Webcam.read = True


	def quit_funct(self):
		self.Webcam.exit = True
		# self.ServerShutdown()
		try: self.videoThread.join()
		except RuntimeError: pass
		self.Socket.stop()
		try: self.serverThread.join()
		except (RuntimeError): pass
		try: self.interface_TK.destroy()
		except AttributeError: pass	

class DiceConnector:

	def __init__(self, Capture, frame_i, ThreshTest=False, ShowSteps=False):
		self.threshold_a = 0
		self.wrk_img = 0
		self.Capture = Capture #cv2.rotate(Capture, cv2.ROTATE_180)
		self.ThreshTest = ThreshTest
		self.defaultThresh = 200
		self.max_dice = 5

		self.diceFindSize = (8000, 30000)
		self.diceOverlapSearchRadius = 3

		self.show = ShowSteps # Set True to show working images


		self.diceList = self.image_processor()		#coords of die bounding boxsbes, in form ([coords], center). 
		self.dice = self.list_dice()				#dice as Dice objects
		self.display()

	def list_dice(self):
		croppedDice = []
		ogImgCenter = self.Capture.centerPoint
		allDice = []

		for i, (coords, center) in enumerate(self.diceList):
			dieRotRatio = (coords[1][0]-coords[0][0])/(coords[1][1]-coords[0][1]+0.000001)
			dieRotRad = math.atan(dieRotRatio)
			dieRotDeg = math.degrees(dieRotRad)
			die = imutils.rotate_bound(self.Capture.img, dieRotDeg)
			newImgCenter = [(len(die)/2), ((len(die[0])/2))]
			
			newCorners = []
			for j in 1,3:
				newx = int(((coords[j][0]-ogImgCenter[0])*math.cos(dieRotRad)) - (coords[j][1]-ogImgCenter[1])*math.sin(dieRotRad) + newImgCenter[1])
				newy = int(((coords[j][1]-ogImgCenter[1])*math.cos(dieRotRad)) + (coords[j][0]-ogImgCenter[0])*math.sin(dieRotRad) + newImgCenter[0])
				newCorners.append([newx,newy])

			arrangedCorners = [sorted([newCorners[0][1],newCorners[1][1]]),sorted([newCorners[0][0],newCorners[1][0]])] #[[y,y1],[x,x1]]
			croppedDie = die[arrangedCorners[0][0]:arrangedCorners[0][1], arrangedCorners[1][0]:arrangedCorners[1][1]]

			allDice.append(Dice(croppedDie, center))
		return (allDice)

	def on_change(self, value=100):
		image_copy = self.wrk_img.copy()
		self.threshold_a = value
		_, image_copy = cv2.threshold(image_copy, self.threshold_a, 255, cv2.THRESH_BINARY)
		cv2.imshow('Threshold Test(a)', image_copy)

	def image_processor(self):
		#Process image to a Binary Threshold
		raw_img = self.Capture.img
		self.wrk_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
		self.wrk_img = cv2.GaussianBlur(self.wrk_img, (11,11), 1000)


		if self.show:
			cv2.imshow("1 - Blurred", self.wrk_img)
			cv2.moveWindow("1 - Blurred", -1275,0)

		
		if self.ThreshTest:
			cv2.imshow('Threshold Test(a)', self.wrk_img)
			a = cv2.createTrackbar("Thresh a", "Threshold Test(a)", 0, 255, self.on_change)
			cv2.waitKey(0)
		else:
			self.threshold_a = self.defaultThresh
		_, self.wrk_img = cv2.threshold(self.wrk_img, self.threshold_a, 255, cv2.THRESH_BINARY)


		if self.show:
			cv2.imshow("2 -Threshold", self.wrk_img)
			cv2.moveWindow("2 -Threshold", -1275,515)


		canny_img = cv2.Canny(self.wrk_img, 1, 50, 2)
		contours, _ = cv2.findContours(canny_img, 1, 2)

		#minimum area rotated rectangle for each contour
		minRect = [None]*len(contours)
		for i, c in enumerate(contours):
			minRect[i] = cv2.minAreaRect(c)

		#Drawing tools init, for debugging
		if self.show:
			drawing = np.zeros((canny_img.shape[0], canny_img.shape[1], 3), dtype=np.uint8)


		#variables declared, all empty. These are used for sorting
		pyBox = []
		sortedbox = []
		boxKeep = []
		centersList = []

		SortedContours = sorted(contours, key=cv2.contourArea, reverse=True)

		for i, c in enumerate(SortedContours):
			# Drawings for debugging
			if self.show:
				cv2.drawContours(drawing, contours, i, RED)


			# Creates a rotated rectangle, not kept between iterations
			box = cv2.boxPoints(minRect[i])
			box = np.intp(box)
			center = (int(minRect[i][0][0]),int(minRect[i][0][1]))
			centersList.append(center)
			

			#removes duplicates
			inList = False
			for k in range(0, len(centersList)-1):
				if ((centersList[k][0] - center[0])**2 + (centersList[k][1] - center[1])**2) < self.diceOverlapSearchRadius**2:
					inList = True
					centersList.pop()
					break

			#sorts a copy of box, and keeps the sorted boxes in a list
			pyBox.append([])
			for j in range(0,4):
				pyBox[i].append([])
				pyBox[i][j].append(box[j][0])
				pyBox[i][j].append(box[j][1])
			sortedbox.append(sorted(pyBox[i]))

			#determines size and ratio of sides for each box
			dx = sortedbox[i][2][0] - sortedbox[i][0][0]
			dy = sortedbox[i][1][1] - sortedbox[i][0][1]
			boxArea = abs(dx*dy)


			#saves (non duplicate) boxes that meet the criteria for size and ratio
			if 0.5 < abs(dy/(dx+0.01)) < 2: #+0.01 is for 0 catching
				if self.diceFindSize[0] < boxArea < self.diceFindSize[1]:
					if not inList:

						boxKeep.append([box, center])
						if self.show:
							cv2.drawContours(drawing, [box], 0, green)
							cv2.circle(drawing, center, 0, green, -1)	

				elif self.show:
					# cv2.drawContours(drawing, [box], 0, MAGENTA)
					pass
			elif self.show:
				# cv2.drawContours(drawing, [box], 0, CYAN)
				pass
		if self.show:
			cv2.imshow("3 - Contours and boxes", drawing)
			cv2.moveWindow("3 - Contours and boxes", -1935,515)
		return(boxKeep)

	def display(self):
		for die, center in (self.diceList):
			cv2.drawContours(self.Capture.img, [die], -1, GREEN, 3)
		cv2.imshow("DiceLink Video Feed", self.Capture.img)
		# cv2.moveWindow("0 - Live Feed", -1935,0)

		if self.show:
			displacement = 0
			for i, d in enumerate(self.dice):
				cv2.imshow("Die "+str(i), d.img)
				cv2.moveWindow("Die "+ str(i), -650,(displacement+(i*30)))
				displacement += d.frameHeight

			for j in range(len(self.dice), self.max_dice):
				try:
					cv2.destroyWindow("Die "+ str(j))
				except cv2.error:
					pass

	def feature_match(self):
		for die in self.dice:
			GrayImg = cv2.cvtColor(die.img, cv2.COLOR_BGR2GRAY)
			orb = cv2.ORB_create()
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			kp_target, des_target = orb.detectAndCompute(GrayImg,None)


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

					# img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
					# plt.imshow(img3),plt.show()


			die.matchlist = sorted(die.matchlist, key = lambda x:x[1])
			die.results()

		# print(matches[0].distance)

class Dice:
	def __init__(self, raw_img, center):
		self.center = center
		self.img = raw_img
		self.frameWidth = len(self.img[0])
		self.frameHeight = len(self.img)
		self.matchlist = []


		self.value = None
		self.type = None

	def results(self):
		raw_result = self.matchlist[0][0].split('\\')
		self.value = raw_result[-1][:-4]
		self.type = raw_result[-2]

		print(self.type, "rolled", self.value )

class VideoCapture:

		def __init__(self):
			self.frameWidth = 640
			self.frameHeight = 480
			self.centerPoint = (self.frameWidth/2, self.frameHeight/2)
			self.capture = cv2.VideoCapture(1)
			self.capture.set(3, self.frameWidth)
			self.capture.set(4, self.frameHeight)
			self.img = 0 #Defined in the "display" method every frame
			self.connection = None #Defined in the "display" method every frame

			self.exit = False
			self.read = False

		def display(self):
			print("Display loading...")
			frame = 0
			while True:
				# time.sleep(1)
				frame += 1
				_, img = self.capture.read()
				self.img = img
				if self.exit:
					break
				self.connection = DiceConnector(self, frame)
				if self.read:
					self.read = False
					self.connection.feature_match()
				# cv2.imshow("Dice Link", img)
				cv2.waitKey(1)

				

			self.capture.release()
			cv2.destroyAllWindows()

class DiceLinkSocketHost:
	def __init__(self, parent, port=8000):
		self.DCWrapper = parent
		self.stop_flag = False
		self.read_flag = False
		self.websocket = None # defined in handler()
		self.pkg = '' #Defined in passthrough
		self.port = port

	def start(self):
		print("Initializing server on port :{}...".format(self.port))
		asyncio.run(self.main())

	def stop(self):
		self.stop_flag = True

	def read(self):
		self.read_flag = True
		try: len(self.DCWrapper.Webcam.connection.dice)
		except AttributeError: self.read() #RECURSION lol

		for i, die in enumerate(self.DCWrapper.Webcam.connection.dice):
			self.pkg = self.pkg + 'die{}: '.format(i) + '{' + 'type: {}, value: {}'.format(die.type, die.value) + "}"
			if i != len(self.DCWrapper.Webcam.connection.dice)-1:
				self.pkg = self.pkg + ', '


		# self.pkg = '({}, {})'.format(self.DCWrapper.Webcam.connection.dice.type, self.DCWrapper.Webcam.connection.dice.value)

	async def shutdown(self):
		while not self.stop_flag:
			await asyncio.sleep(1)
		print("Socket shutting down...")
		try: await self.websocket.send("Socket shutting down...")
		except: 
			pass
		raise Exception("Burn, baby, burn")




	async def handler(self, websocket):
		self.websocket = websocket
		async for message in self.websocket:
			""" Command Handler Block """
			if message == "":
				continue
			if message[0] == "/":
				match message:
					case "/read":
						self.read()
						# self.DCWrapper.Webcam.connection.dice
						await websocket.send(self.pkg)
					case '/stop':
						self.stop()
						await self.shutdown()
					case _:
						await self.websocket.send(message)

			else: print(message)


	async def main(self):
		async with websockets.serve(self.handler, "", self.port):
			await self.shutdown()  # run for a time
		print('Shutdown Successful')


if __name__ == '__main__':
	DC()

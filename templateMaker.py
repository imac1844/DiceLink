"""
____________________________________________________________________________________________
Dice Connection 
- Template Maker v1.0

This is a tool designed to capture reference photos of dice sets, to be used by DiceReader. 




_____________________________________________________________________________________________
"""



import cv2
import tkinter as tk
from threading import Thread
import os
import sys


class TM:
	""" Wrapper class, 
	"""
	def __init__(self):
		
		self.frameWidth = 640      # 
		self.frameHeight = 480
		self.centerPoint = (int(self.frameWidth/2), int(self.frameHeight/2))
		self.capture = cv2.VideoCapture(1)   							#change the value if it loads the wrong webcam
		self.capture.set(3, self.frameWidth)
		self.capture.set(4, self.frameHeight)


		self.guideSquareSize = 200
		self.guideSquareP1 = (int(self.centerPoint[0] + self.guideSquareSize/2), int(self.centerPoint[1] + self.guideSquareSize/2))
		self.guideSquareP2 = (int(self.centerPoint[0] - self.guideSquareSize/2), int(self.centerPoint[1] - self.guideSquareSize/2))
		self.cropSquareX1 = int(self.frameWidth /2-self.guideSquareSize/2)
		self.cropSquareX2 = int(self.frameWidth /2+self.guideSquareSize/2)
		self.cropSquareY1 = int(self.frameHeight/2-self.guideSquareSize/2)
		self.cropSquareY2 = int(self.frameHeight/2+self.guideSquareSize/2)

		self.button_list = []
		self.GUI_coords = [  # format is (diceType, row, column). information is for TKinter.
		(4,   0, 0), 
		(6,   0, 1), 
		(8,   0, 2), 
		(10,  1, 0), 
		(12,  1, 1), 
		(20,  1, 2),
		(100, 2, 0)]
		self.remainingValues = 0


		self.displayThread = Thread(target = self.display)
		

		self.setname = None #User Entered in "check_name" method
		self.img = 0 #Defined in the "display" method every frame

		# Internal Control Flags
		self.trigger_save = False
		self.show_template = True
		self.exit = False

		self.lastDieValue = None
		self.lastDieType = None

		if __name__ == '__main__':
			self.main()
			cv2.waitKey()
		
	# def help(self):
	# 	print("Help Message")

	def main(self):
		self.name_entry_funct()
		if self.setname == None:
			print("Window closed, exiting...")
			self.quit_funct()
			return
		self.dice_choice_buttons()

	def name_entry_funct(self):
		self.NameEntry_TK = tk.Tk()
		self.NameEntry_TK.title("New Dice")
		self.NameEntryWindow = tk.Canvas(self.NameEntry_TK, width=300, height=200, bd = 15, bg = 'green')
		self.NameEntryWindow.grid(columnspan=4, rowspan = 5)

		self.UE_dice_label = tk.Label(text = "Dice Set Name:")
		self.UE_dice_label.grid(row = 1, column = 1, columnspan = 2)

		self.name_box = tk.Entry()
		self.name_box.grid(row = 2, column = 1, columnspan = 2)

		self.button_GO = tk.Button(width = 10, height = 2, text = 'Go!', command = self.check_name)
		self.button_GO.grid(row = 3, column = 1, columnspan = 2)

		self.NameEntry_TK.mainloop()

	def check_name(self):
		try: os.mkdir('./DiceSets/{}'.format(self.name_box.get()))
		except FileExistsError:
			self.feedback_label = tk.Label()
			self.feedback_label.grid(row = 4, column = 1, columnspan = 2)
			self.feedback_label['text'] = 'Name Taken! \n Enter New Name & GO! or Overwrite'

			self.button_OW = tk.Button(width = 10, height = 2, text = 'Overwrite', command = self.config_name)
			self.button_OW.grid(row = 3, column = 2, columnspan = 1)

			self.button_GO.grid(row = 3, column = 1, columnspan = 1)
			return
		self.config_name()

	def config_name(self):
		self.setname = self.name_box.get()
		self.NameEntry_TK.destroy()

	def quit_funct(self):
		self.exit = True
		try: self.displayThread.join()
		except RuntimeError: pass
		try: self.DiceChoice_TK.destroy()
		except AttributeError: pass	
		try: self.Face_Val.destroy()
		except AttributeError: pass	

	def mem_clear(self):
		try: 
			self.button_1.destroy()
			self.button_2.destroy()
			self.button_3.destroy()
			self.button_4.destroy()
			self.button_5.destroy()
			self.button_6.destroy()
			self.button_7.destroy()
			self.button_8.destroy()
			self.button_9.destroy()
			self.button_10.destroy()
			self.button_11.destroy()
			self.button_12.destroy()
			self.button_13.destroy()
			self.button_14.destroy()
			self.button_15.destroy()
			self.button_16.destroy()
			self.button_17.destroy()
			self.button_18.destroy()
			self.button_19.destroy()
			self.button_20.destroy()
		except AttributeError: 
			pass

	def on_press(self, value):
		if value > 20:
			if value > 100:
				value = int((value-100)/10)
			else:
				value = int(value/10)
		self.lastDieValue = value

		statechange_str = 'self.button_{}["state"] = tk.DISABLED'.format(value)
		exec(statechange_str)
		bg_str = 'self.button_{}["bg"] = "cyan"'.format(value)
		exec(bg_str)
		self.trigger_save = True
		self.remainingValues -= 1


		if self.remainingValues == 0:
			statechange_str = 'self.button_d{}["state"] = tk.DISABLED'.format(self.lastDieType)
			exec(statechange_str)
			bg_str = 'self.button_d{}["bg"] = "green"'.format(self.lastDieType)
			exec(bg_str)

			if self.lastDieType == 100:
				bound = 10
			else:
				bound = self.lastDieType


			for i in range (1, bound+1):
				bg_str = 'self.button_{}["bg"] = "green"'.format(i)
				exec(bg_str)

	def save(self):
		
		self.trigger_save = False
		cropped_die = self.img[self.cropSquareY1:self.cropSquareY2, self.cropSquareX1:self.cropSquareX2]
		cv2.imwrite('./DiceSets/{}/d{}/{}.jpg'.format(self.setname, self.lastDieType, self.lastDieValue), cropped_die)

	def DieChecklist(self):
		try: os.mkdir('./DiceSets/{}/d{}/'.format(self.setname, self.lastDieType))
		except FileExistsError: pass
		self.remainingValues = 0
		if self.lastDieType == 100:
			self.remainingValues = 10
		else: self.remainingValues = int(self.lastDieType)

	def dice_choice_buttons(self):
		self.displayThread.start()
		self.DiceChoice_TK = tk.Tk()
		self.DiceChoice_TK.title(self.setname)

		self.DiceButtonWindow = tk.Canvas(self.DiceChoice_TK, width=220, height=150, bd = 15, bg = 'cyan')
		self.DiceButtonWindow.grid(columnspan=3, rowspan = 5)
		self.spacer = tk.Canvas(width = 220, height = 5, bg = 'green', bd = 15)

		self.spacer.grid(row = 3, column = 0, columnspan = 3)

		# for x, i, j in self.GUI_coords:
		# 	newButton = tk.Button(width = 10, height = 2, text = 'd{}'.format(x), command = self.die_val_funct(x))
		# 	newButton.grid(row = i, column = j)
		# 	self.button_list.append((x, newButton))



		self.button_d4 = tk.Button(width = 10, height = 2, text = 'd4', command = lambda: self.die_val_funct(4))
		self.button_d4.grid(row = 0, column = 0)

		self.button_d6 = tk.Button(width = 10, height = 2, text = 'd6', command = lambda: self.die_val_funct(6))
		self.button_d6.grid(row = 0, column = 1)
		
		self.button_d8 = tk.Button(width = 10, height = 2, text = 'd8', command = lambda: self.die_val_funct(8))
		self.button_d8.grid(row = 0,column = 2)

		self.button_d10 = tk.Button(width = 10, height = 2, text = 'd10', command = lambda: self.die_val_funct(10))
		self.button_d10.grid(row = 1, column = 0)

		self.button_d100 = tk.Button(width = 10, height = 2, text = 'd100', command = lambda: self.die_val_funct(100))
		self.button_d100.grid(row = 2, column = 0)

		self.button_d12 = tk.Button(width = 10, height = 2, text = 'd12', command = lambda: self.die_val_funct(12))
		self.button_d12.grid(row = 1, column = 1)
		
		self.button_d20 = tk.Button(width = 10, height = 2, text = 'd20', command = lambda: self.die_val_funct(20))
		self.button_d20.grid(row = 1,column = 2)

		self.button_quit = tk.Button(width = 10, height = 2, text = 'Done', command = self.quit_funct)
		self.button_quit.grid(row = 2,column = 2)

		self.DiceChoice_TK.mainloop()

	def die_val_funct(self, die_type=int):
		self.mem_clear()
		self.lastDieType = die_type
		self.DieChecklist()

		if die_type == 100:
			self.button_1 = tk.Button(width = 5, height = 2, text = 10, command = lambda: self.on_press(110))
			self.button_1.grid(row = 0, column = 5)
			self.button_2 = tk.Button(width = 5, height = 2, text = 20, command = lambda: self.on_press(120))
			self.button_2.grid(row = 0, column = 6)
			self.button_3 = tk.Button(width = 5, height = 2, text = 30, command = lambda: self.on_press(30))
			self.button_3.grid(row = 0, column = 7)
			self.button_4 = tk.Button(width = 5, height = 2, text = 40, command = lambda: self.on_press(40))
			self.button_4.grid(row = 0, column = 8)
			self.button_5 = tk.Button(width = 5, height = 2, text = 50, command = lambda: self.on_press(50))
			self.button_5.grid(row = 0, column = 9)
			self.button_6 = tk.Button(width = 5, height = 2, text = 60, command = lambda: self.on_press(60))
			self.button_6.grid(row = 1, column = 5)
			self.button_7 = tk.Button(width = 5, height = 2, text = 70, command = lambda: self.on_press(70))
			self.button_7.grid(row = 1, column = 6)
			self.button_8 = tk.Button(width = 5, height = 2, text = 80, command = lambda: self.on_press(80))
			self.button_8.grid(row = 1, column = 7)
			self.button_9 = tk.Button(width = 5, height = 2, text = 90, command = lambda: self.on_press(90))
			self.button_9.grid(row = 1, column = 8)
			self.button_10 = tk.Button(width = 5, height = 2, text = '00', command = lambda: self.on_press(100))
			self.button_10.grid(row = 1, column = 9)
			return

		self.button_1 = tk.Button(width = 5, height = 2, text = 1,  command = lambda: self.on_press(1))
		self.button_1.grid(row = 0, column = 5)
		self.button_2 = tk.Button(width = 5, height = 2, text = 2, command = lambda: self.on_press(2))
		self.button_2.grid(row = 0, column = 6)
		self.button_3 = tk.Button(width = 5, height = 2, text = 3, command = lambda: self.on_press(3))
		self.button_3.grid(row = 0, column = 7)
		self.button_4 = tk.Button(width = 5, height = 2, text = 4, command = lambda: self.on_press(4))
		self.button_4.grid(row = 0, column = 8)
		if die_type == 4:
			return

		self.button_5 = tk.Button(width = 5, height = 2, text = 5, command = lambda: self.on_press(5))
		self.button_5.grid(row = 0, column = 9)
		self.button_6 = tk.Button(width = 5, height = 2, text = 6, command = lambda: self.on_press(6))
		self.button_6.grid(row = 1, column = 5)
		if die_type == 6:
			return

		self.button_7 = tk.Button(width = 5, height = 2, text = 7, command = lambda: self.on_press(7))
		self.button_7.grid(row = 1, column = 6)
		self.button_8 = tk.Button(width = 5, height = 2, text = 8, command = lambda: self.on_press(8))
		self.button_8.grid(row = 1, column = 7)
		if die_type == 8:
			return

		self.button_9 = tk.Button(width = 5, height = 2, text = 9, command = lambda: self.on_press(9))
		self.button_9.grid(row = 1, column = 8)
		self.button_10 = tk.Button(width = 5, height = 2, text = 10, command = lambda: self.on_press(10))
		self.button_10.grid(row = 1, column = 9)
		if die_type == 10:
			return

		self.button_11 = tk.Button(width = 5, height = 2, text = 11, command = lambda: self.on_press(11))
		self.button_11.grid(row = 2, column = 5)
		self.button_12 = tk.Button(width = 5, height = 2, text = 12, command = lambda: self.on_press(12))
		self.button_12.grid(row = 2, column = 6)
		if die_type == 12:
			return
		self.button_13 = tk.Button(width = 5, height = 2, text = 13, command = lambda: self.on_press(13))
		self.button_13.grid(row = 2, column = 7)
		self.button_14 = tk.Button(width = 5, height = 2, text = 14, command = lambda: self.on_press(14))
		self.button_14.grid(row = 2, column = 8)
		self.button_15 = tk.Button(width = 5, height = 2, text = 15, command = lambda: self.on_press(15))
		self.button_15.grid(row = 2, column = 9)
		self.button_16 = tk.Button(width = 5, height = 2, text = 16, command = lambda: self.on_press(16))
		self.button_16.grid(row = 3, column = 5)
		self.button_17 = tk.Button(width = 5, height = 2, text = 17, command = lambda: self.on_press(17))
		self.button_17.grid(row = 3, column = 6)
		self.button_18 = tk.Button(width = 5, height = 2, text = 18, command = lambda: self.on_press(18))
		self.button_18.grid(row = 3, column = 7)
		self.button_19 = tk.Button(width = 5, height = 2, text = 19, command = lambda: self.on_press(19))
		self.button_19.grid(row = 3, column = 8)
		self.button_20 = tk.Button(width = 5, height = 2, text = 20, command = lambda: self.on_press(20))
		self.button_20.grid(row = 3, column = 9)		

	def display(self):
		print("Template Maker loading... \n \n Place die square, centered and oriented upright. Follow prompts \n")
		frame = 0

		_, img = self.capture.read()
		cv2.imshow("Template Maker", img)

		while True:
			# time.sleep(1)
			frame += 1
			_, img = self.capture.read()
			self.img = img
			if cv2.waitKey(1) & 0xFF == ord('q'):
				self.quit_funct()

			cv2.rectangle(img, self.guideSquareP1, self.guideSquareP2, (30,30,30), 5)
			cv2.imshow("Template Maker", img)

			if self.exit:
				break

			if self.trigger_save:
				self.save()
				

		self.capture.release()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	TM()
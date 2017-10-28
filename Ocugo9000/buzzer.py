import RPi.GPIO as GPIO
import time
import threading

class Buzzer:

	def __init__(self, gpin, frequency, beep_duration, pause_duration):
		self.gpin = gpin
		self.frequency = frequency
		self.speed = 1/frequency
		self.beep_duration = beep_duration
		self.pause_duration = pause_duration
		self.is_buzzing = False
		self.lock = threading.Lock()

		GPIO.setmode(GPIO.BCM)
		GPIO.setup(gpin, GPIO.OUT)
		GPIO.output(gpin, False)


	def loop(self):
		while True:
			with self.lock:
				if not self.is_buzzing:
					return

			beep_end = time.time() + self.beep_duration

			# for beep length
			while time.time() < beep_end:
				GPIO.output(self.gpin, True)
				time.sleep(self.speed)
				GPIO.output(self.gpin, False)
				time.sleep(self.speed)

			time.sleep(self.pause_duration)

	def start(self):
		self.is_buzzing = True
		
		thread = threading.Thread(target = self.loop, daemon = True)
		#thread.start()


	def stop(self):
		with self.lock:
			self.is_buzzing = False


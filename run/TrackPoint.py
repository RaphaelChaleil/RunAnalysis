import math
import json
import urllib.request
from datetime import datetime
from datetime import timedelta
from dataclasses import dataclass

@dataclass
class TrackPoint(object):
	def __init__(self,time: str):
		self.time = time
		#self.time = datetime.fromisoformat(time)
		#self.time = dateutil.parser.isoparse(time)
		print(self.time)
		self.HR = 0
		self.lat = 0.0
		self.lon = 0.0
		self.dist = 0.0
		self.alt = 0
		self.alt_SRTM250 = 0
		self.velocity = 0.0
		self.power = 0.0
		self.gradient = 0.0
		self.dt = 0.0

	def add_coords(self,lat: float,lon: float):
		self.lat = lat
		self.lon = lon

	def add_HR(self,HR: float):
		self.HR = HR

	def add_dist(self,dist: float):
		self.dist = dist

	def add_altitude(self,alt: int):
		self.alt = alt
		#print(self.alt)

	def add_altitude_SRTM(self,alt: int):
		self.alt_SRTM250 = alt

	def add_gradient(self,grad: float):
		self.gradient = grad

	def add_velocity(self,v: float):
		self.velocity = v

	def add_dt(self,dt):
		self.dt = dt

	def add_power(self,p: float):
		self.power = p

	def str(self):
		out = self.time
		if self.lat and self.lon:
			out += "\t("+str(self.lat)+","+str(self.lon)
			if self.alt:
				out += ","+str(self.alt)+")"
		if self.dist:
			out += "\t"+str(self.dist)
		if self.HR:
			out += "\tHR: "+str(self.HR)
		return out

	def data_dict(self):
		return {"latitute":self.lat,"longitude":self.lon,"HR":self.HR,"distance":self.dist,"date":self.time,"time":self.dt,"altitude":self.alt,"velocity":self.velocity,"power":self.power,"gradient":self.gradient}

	def is_complete(self):
		if self.lat and self.lon and self.dist>0: # and self.HR:
			return True
		return False

	def has_hr(self):
		if self.HR:
			return True
		return False

	def get_position(self):
		return {"latitude":self.lat,"longitude":self.lon}

	def get_altitude(self):
		if not self.alt:
			loclist = {"locations":[self.get_position()]}
			json_data=json.dumps(loclist,skipkeys=int).encode('utf8')
			url="https://api.open-elevation.com/api/v1/lookup"
			response = urllib.request.Request(url,json_data,headers={'Content-Type': 'application/json'})
			fp=urllib.request.urlopen(response)
			res_byte=fp.read()
			res_str=res_byte.decode("utf8")
			js_str=json.loads(res_str)
			fp.close()
			response_len=len(js_str['results'])
			self.alt = int(js_str['results'][0]['elevation'])
		return self.alt

	def time_delta(self,tp):
		return math.fabs((self.time-tp.time).total_seconds())

	def distance_delta(self,tp):
		return math.fabs(self.dist-tp.dist)


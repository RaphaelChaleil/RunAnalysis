from .TrackPoint import TrackPoint
from typing import List
import xml.etree.ElementTree as ET
import dateutil.parser
import math
import json
import folium
import urllib.request
from folium import plugins
import branca.colormap
import pandas as pd
from geopy import distance
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import gpxpy
import fitdecode
from dataclasses import dataclass
from datetime import timedelta

@dataclass
class Track(object):
	def __init__(self,name: str):
		self.name = name
		self.points = []
		self.avg_hr = 0
		self.stdev_hr = 0
		self.hr_min = 1000
		self.hr_max = 0
		self.avg_vel = 0
		self.stdev_vel = 0
		self.vel_min = 100
		self.vel_max = 0
		self.avg_power = 0
		self.stdev_power = 0
		self.mass = 90.0
		self.total_distance = 0
		self.total_time = 0
		self.cooper_dist = 0
		self.VO2max = 0
		self.cooper_dist_measured = 0
		self.VO2max_measured = 0
		self.pace = ""
		self.alt_min = 10000.0
		self.alt_max = -400.0
		self.min_gradient = 100.0
		self.max_gradient = -100.0
		self.DF = None
		self.grid = None
		self.alt_grid_smooth = None
		self.alt_grid_longitudes = None
		self.alt_grid_latitudes = None

	def getGPX_HR(self,waypoint):
		for extension in waypoint.extensions:
			#print(extension[0])
			if extension[0].tag == '{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}hr':
				return float(extension[0].text)

	def read_fit(self,fit_file: str):
		pos_ratio = 360/(2**32)
		trackpoints_list = List[TrackPoint]
		with fitdecode.FitReader(fit_file) as fit_file:
			for frame in fit_file:
				if isinstance(frame, fitdecode.records.FitDataMessage):
					if frame.name == "record":
						print(frame.name)
						tp = None
						if frame.has_field('position_lat') and frame.has_field('position_long'):
							lat = pos_ratio * float(frame.get_value("position_lat"))
							lon = pos_ratio * float(frame.get_value("position_long"))
							dist = float(frame.get_value('distance'))
							alt = float(frame.get_value('enhanced_altitude'))
							hr  = int(frame.get_value('heart_rate'))
							ts = frame.get_value('timestamp')
							print(ts)
							#time = dateutil.parser.isoparse(ts)
							#print(time)
							tp = TrackPoint(ts)
							tp.add_coords(lat,lon)
							tp.add_dist(dist)
							tp.add_HR(hr)
							tp.add_altitude(alt)
						if tp:
							trackpoints_list.append(tp)
			#print(len(trackpoints_list))
			for t in trackpoints_list:
				print((t.lat,t.lon,t.dist))
				if t.is_complete():
					self.add_point(t)
			self.total_distance = self.points[-1].dist
			self.total_time = self.points[-1].time_delta(self.points[0])
			self.calc_velocities_power()
			self.calc_hr_stats()
			self.calc_Cooper()
			self.calc_pace()
			self.calc_grid()
			self.calc_gradients()

	def read_tcx(self,tcx_file: str):
		tree = ET.parse(tcx_file)
		namespace = "{http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2}"
		ns = {'ns' : namespace}
		root = tree.getroot()
		trackpoints_list = List[TrackPoint]
		last_hr = 0
		for activities in root.findall(namespace+'Activities',ns):
			for activity in activities.findall(namespace+'Activity'):
				for lap in activity.findall(namespace+'Lap'):
					for track in lap.iter(namespace+'Track'):
						for trackpoint in track.iter(namespace+'Trackpoint'):
							tp = None
							for Time in trackpoint.findall(namespace+'Time'):
								time = dateutil.parser.isoparse(Time.text)
								tp = TrackPoint(time)
								#tp = TrackPoint(Time.text)
							for Position in trackpoint.findall(namespace+'Position'):
								lat = None
								lon = None
								for latitude in Position.findall(namespace+'LatitudeDegrees'):
									lat = float(latitude.text)
								for longitude in Position.findall(namespace+'LongitudeDegrees'):
									lon = float(longitude.text)
								if lat and lon:
									tp.add_coords(lat,lon)
							for DistanceMeters in trackpoint.findall(namespace+'DistanceMeters'):
								dist = DistanceMeters.text
								print("dist: {dist:f}".format(dist=float(dist)))
								tp.add_dist(float(dist))
							hr = None
							for HR in trackpoint.findall(namespace+'HeartRateBpm'):
								for value in HR.findall(namespace+'Value'):
									hr = float(value.text)
							if not hr:
								hr = last_hr
							tp.add_HR(hr)
							last_hr = hr
							if tp:
								trackpoints_list.append(tp)
		locations = []
		selection = []
		for t in trackpoints_list:
			if t.is_complete():
				self.add_point(t)
		self.total_distance = self.points[-1].dist
		self.total_time = self.points[-1].time_delta(self.points[0])
		self.retrieve_altitudes()
		self.calc_velocities_power()
		self.calc_hr_stats()
		self.calc_Cooper()
		self.calc_pace()
		self.calc_grid()
		self.calc_gradients()

	def read_gpx(self,gpxfilename:str):
		gpx_file = open(gpxfilename,'r')
		gpx = gpxpy.parse(gpx_file)
		data = gpx.tracks[0].segments[0].points
		start = data[0]
		end = data[-1]
		df = pd.DataFrame(columns=['lon','lat','alt','time','hr'])
		trackpoints_list = []
		for point in data:
			#print(pd.to_datetime(point.time))
			#df = df.append({'lon' : point.longitude, 'lat' : point.latitude, 'alt' : point.elevation,'time' : pd.to_datetime(point.time), 'hr' : getHR(point)},ignore_index=True)
			#print(pd.to_datetime(point.time))
			#print(type(point.time))
			#print(dateutil.parser.isoparse(point.time))
			tp = TrackPoint(point.time)
			tp.add_coords(point.latitude,point.longitude)
			tp.add_altitude(point.elevation)
			tp.add_HR(self.getGPX_HR(point))
			if tp:
				trackpoints_list.append(tp)
		index_step =1
		td = 0
		for index in range(len(data)):
			if index == 0:
				pass
			else:
				start = data[index-index_step]
				stop = data[index]
				distance_vin_2d = distance.geodesic((start.latitude,start.longitude),(stop.latitude,stop.longitude)).m
				td += distance_vin_2d
				print("dist: {dist:f}".format(dist=td))
				if distance_vin_2d == 0:
					index_step += 1
					pass
				else:
					index_step = 1
				trackpoints_list[index].add_dist(td)
		for t in trackpoints_list:
			if t.is_complete():
				#print(t.dist,t.time)
				self.add_point(t)
		self.total_distance = self.points[-1].dist
		self.total_time = self.points[-1].time_delta(self.points[0])
		#self.retrieve_altitudes()
		self.calc_velocities_power()
		if self.points[0].has_hr():
			self.calc_hr_stats()
		self.calc_Cooper()
		self.calc_pace()
		self.calc_grid()
		self.calc_gradients()

	def set_weight(self,w: float):
		self.mass=w

	def add_point(self,p: TrackPoint):
		self.points.append(p)

	def add_traj(self,points: List[TrackPoint]):
		self.points.extend(points)

	def get_open_elevation_data(self,loclist):
		json_data=json.dumps(loclist,skipkeys=int).encode('utf8')
		url="https://api.open-elevation.com/api/v1/lookup"
		response = urllib.request.Request(url,json_data,headers={'Content-Type': 'application/json'})
		fp=urllib.request.urlopen(response)
		res_byte=fp.read()
		res_str=res_byte.decode("utf8")
		js_str=json.loads(res_str)
		fp.close()
		return js_str

	def get_opentopo_elevation(self,point):
		urltemplate = "https://api.opentopodata.org/v1/eudem25m?locations="
		url = "{urlt:s}{lat:f},{lon:f}".format(urlt=urltemplate,lat=point.lat,lon=point.lon)
		result = urllib.request.urlopen(url).read()
		js_str=json.loads(result.decode("utf-8"))
		return js_str


	def calc_grid(self):
		lat_min = 90
		lat_max = -90
		lon_min = 180
		lon_max = -180
		lat_list = []
		lon_list = []
		for p in self.points:
			lat_list.append(p.lat)
			lon_list.append(p.lon)
			if p.lat < lat_min:
				lat_min = p.lat
			if p.lat > lat_max:
				lat_max = p.lat
			if p.lon < lon_min:
				lon_min = p.lon
			if p.lon > lon_max:
				lon_max = p.lon
		self.grid = ((lat_min,lon_min),(lat_max,lon_max))
		w1 = distance.geodesic((lat_min,lon_min),(lat_min,lon_max)).m
		w2 = distance.geodesic((lat_max,lon_min),(lat_max,lon_max)).m
		h =  distance.geodesic((lat_min,lon_min),(lat_max,lon_min)).m
		r = (w1+w2)/(2*h)
		n = 250
		if r > 1:
			n = int(math.ceil(h/250))
			latrange = np.linspace(lat_min,lat_max,n)
			lonrange = np.linspace(lon_min,lon_max,int(n/r))
		else:
			n = int(math.ceil(w1/250))
			latrange = np.linspace(lat_min,lat_max,int(n/r))
			lonrange = np.linspace(lon_min,lon_max,n)
		print("r = ",r," n = ",n)
		print("lat_linspace: ",latrange)
		print("lon_linspace: ",lonrange)
		alt_grid_points = []
		for lat in latrange:
			for lon in lonrange:
				alt_grid_points.append({"latitude":lat,"longitude":lon})
		loclist = {"locations":	alt_grid_points}
		js_str = self.get_open_elevation_data(loclist)
		lat_size = len(latrange)
		lon_size = len(lonrange)
		alt_grid = []
		for i in range(lat_size):
			lat_grid = []
			for j in range(lon_size):
				alt_grid.append(float(js_str['results'][i*lon_size+j]['elevation']))
				#lat_grid.append(float(js_str['results'][i*lon_size+j]['elevation']))
			#alt_grid.append(lat_grid)
		np_alt_grid = np.array(alt_grid)
		#print(np_alt_grid.shape)
		np_alt_grid_2d = np_alt_grid.reshape((lat_size,lon_size))
		#np_alt_grid_2d = np_alt_grid.reshape((lon_size,lat_size))
		#print(np_alt_grid_2d.shape,lat_size,lon_size)
		f = interpolate.interp2d(latrange, lonrange, np_alt_grid, kind='cubic')
		#f = interpolate.interp2d(lonrange, latrange, np_alt_grid, kind='cubic')
		n2 = 100
		m2 = 20
		if r > 1:
			m2 = int(n2/r)
			latrange2 = np.linspace(lat_min,lat_max,n2)
			lonrange2 = np.linspace(lon_min,lon_max,m2)
		else:
			m2 = int(n2/r)
			latrange2 = np.linspace(lat_min,lat_max,m2)
			lonrange2 = np.linspace(lon_min,lon_max,n2)
		alt_grid2 = np.flip(f(latrange2,lonrange2),1)
		#alt_grid2 = f(lonrange2,latrange2)
		#for row in alt_grid:
		#	print(row)
		#print(alt_grid2)
		#for row in alt_grid2:
		#	print(row)
		new_grid = np.flip(np_alt_grid_2d,0)
		fig, ax = plt.subplots(1, 2)
		ax[0].imshow(new_grid, cmap='terrain', interpolation='nearest')
		ax[1].imshow(alt_grid2.T, cmap='terrain', interpolation='nearest')
		#plt.imshow(alt_grid2, cmap='terrain', interpolation='nearest')
		plt.show()

		#print(self.grid,h,w1,w2)
		#print("shapes: ",latrange2.shape,lonrange2.shape,alt_grid2.shape,n2,m2)
		self.alt_grid_latitudes = latrange2
		self.alt_grid_longitudes = lonrange2
		self.alt_grid_smooth = alt_grid2
		smooth_srtm_altitudes = f(lat_list,lon_list)
		for i in range(len(self.points)):
			self.points[i].alt_SRTM250 = smooth_srtm_altitudes[i][i]


	def find_dist_time(self,dist: float):
		if dist <= self.points[-1].dist:
			i = 0
			i_min = -1
			min_diff = 10000
			dt = self.points[0].time
			for p in self.points:
				diff = math.fabs(p.dist-dist)
				if diff < min_diff:
					i_min = i
					min_diff = diff
					#print(i,diff,p.time,p.dist)
					dt = self.points[0].time_delta(p)
				i += 1
			return self.points[i_min].time - self.points[0].time
		return self.total_time

	def find_time_dist(self,time: int):
		if time < self.total_time:
			i = 0
			i_min = -1
			min_diff = 1000000
			dx = 0
			t = self.points[0].time
			for p in self.points:
				dt = math.fabs(self.points[0].time_delta(p)-time)
				if dt < min_diff:
					min_diff = dt
					dx = self.points[0].distance_delta(p)
					i_min = i
					i += 1
			return dx
		return self.total_distance


	def retrieve_altitudes(self):
		block_size = 1024
		altitudes = []
		for i in range(0,len(self.points),block_size):
			loclist = {"locations":[loc.get_position() for loc in self.points[i:i+block_size]]}
			js_str = self.get_open_elevation_data(loclist)
			response_len=len(js_str['results'])
			k = 0
			for j in range(i,i+response_len):
				self.points[j].add_altitude_SRTM(float(js_str['results'][k]['elevation']))
				if self.points[j].alt_SRTM250 < self.alt_min:
					self.alt_min = self.points[j].alt_SRTM250
				if self.points[j].alt_SRTM250 > self.alt_max:
					self.alt_max = self.points[j].alt_SRTM250
				k  += 1

	def calc_velocities_power(self):
		n = 0
		power2 = 0
		vel2 = 0
		for i in range(1,len(self.points)):
			dt = self.points[i].time_delta(self.points[i-1])
			if dt == 0:
				pass
			else:
				v = self.points[i].distance_delta(self.points[i-1])/dt
				#print(3.6*v)
				p = 0.5*self.mass*(v**2)/dt
				self.points[i].add_velocity(v)
				self.avg_vel += v
				if v > self.vel_max:
					self.vel_max = v
				if v < self.vel_min:
					self.vel_min = v
				vel2 += v**2
				self.points[i].add_power(p)
				self.points[i].add_dt(self.points[i].time_delta(self.points[0]))
				self.avg_power += p
				power2 += p**2
				n += 1
		self.avg_vel /= n
		self.avg_power /= n
		self.stdev_vel = math.sqrt((vel2/n)-self.avg_vel**2)
		self.stdev_power = math.sqrt((power2/n)-self.avg_power**2)

	def calc_gradients(self):
		for i in range(1,len(self.points)-1):
			#print(self.points[i].alt)
			dh = self.points[i+1].get_altitude() - self.points[i-1].get_altitude()
			dx = self.points[i+1].distance_delta(self.points[i-1])
			#print(self.points[i+1].alt,self.points[i-1].alt ,dh,dx,100*dh/dx)
			self.points[i].add_gradient(100*dh/self.points[i+1].distance_delta(self.points[i-1]))
			if self.points[i].gradient > self.max_gradient:
				self.max_gradient = self.points[i].gradient
			if self.points[i].gradient < self.min_gradient:
				self.min_gradient = self.points[i].gradient

	def check_altitudes(self):
		sum_diff = 0
		#sum_diff2 = 0
		#big_errs = 0
		for p in self.points:
			diff = p.alt - p.alt_SRTM250
			print(p.alt,p.alt_SRTM250,diff)
			#if math.fabs(diff) > 10:
			#	diff2 = p.alt-float(self.get_opentopo_elevation(p)["results"][0]["elevation"])
			#	sum_diff2 += diff2
			#	big_errs += 1
			sum_diff += diff
		print(sum_diff/len(self.points))
		#if big_errs > 0:
		#	print(sum_diff2/big_errs)

	def calc_hr_stats(self):
		hr2 = 0
		n = 0
		for p in self.points:
			self.avg_hr += p.HR
			hr2 += p.HR**2
			if p.HR < self.hr_min:
				self.hr_min = p.HR
			if p.HR > self.hr_max:
				self.hr_max = p.HR
			n += 1
		self.avg_hr /= n
		self.stdev_hr = math.sqrt((hr2/n)-self.avg_hr**2)

	def calc_Cooper(self):
		self.cooper_dist = 720*self.total_distance/self.total_time
		self.VO2max  = (self.cooper_dist - 504.9)/44.73
		self.cooper_dist_measured = self.find_time_dist(720)
		self.VO2max_measured = (self.cooper_dist_measured - 504.9)/44.73

	def calc_pace(self):
		raw_pace = 1000*self.total_time/self.total_distance
		pace_min = int(raw_pace/60)
		pace_sec = int(raw_pace)%60
		self.pace = "{minutes:d}:{seconds:d}".format(minutes=pace_min,seconds=pace_sec)

	def stats(self):
		stats = "#ROUTE#\nDistance: {dist:.3f} Km\tTime: {time:s}\tPace: {pace:s} min/Km\n".format(dist=self.total_distance/1000,time=str(timedelta(seconds=self.total_time)),pace=self.pace)
		if self.points[0].has_hr():
			stats += "\n#HEART RATE#\naverage: {avg_hr:d} BPM\tstandard deviation: {std_hr:f}\tmin: {min_hr:d} BPM\tmax: {max_hr:d} BPM\n".format(avg_hr=int(self.avg_hr),std_hr=self.stdev_hr,
			min_hr=int(self.hr_min),max_hr=int(self.hr_max))
		stats += "\n#COOPER TEST PROJECTED VALUES#\nDistance: {cooper_d:.2f} m\testimated VO2max: {vo2max:.1f} ml/min/Kg\tDistance in first 12 minutes: {cooper_dm:.2f} m\testimated VO2max measured: {vo2max_m:.1f}\n".format(cooper_d=self.cooper_dist,vo2max=self.VO2max,
				cooper_dm=self.cooper_dist_measured,vo2max_m=self.VO2max_measured)
		stats += "\n#POWER#\nAverage speed: {speed:f} Km/h\tAverage power: {power:f} W\tPower to weight: {power_to_weight:f} W/Kg\n".format(speed=3.6*self.avg_vel,power=self.avg_power,power_to_weight=self.avg_power/self.mass)
		#stats += "\n#ALTITUDE#\nmin: {altmin:d} m\tmax: {altmax:d} m\tmin gradient: {mingrad:f} % max gradient: {maxgrad:f} %".format(altmin=self.alt_min,altmax=self.alt_max,mingrad=self.min_gradient,maxgrad=self.max_gradient)
		return stats

	def get_dataframe(self,threshold=0.8):
		self.DF = pd.DataFrame([p.data_dict() for p in self.points if p.velocity > threshold])
		return self.DF

	def map_html(self,title: str):
		##if not self.DF:
		#	self.get_dataframe()
		m = folium.Map(location=[0.5*(self.grid[0][0]+self.grid[1][0]),0.5*(self.grid[0][1]+self.grid[1][1])],tiles='Openstreetmap',zoom_start=20)
		levels = 50
		if len(self.points) < levels:
			levels = len(self.points)
		linmap = getattr(branca.colormap.linear, 'viridis')
		#altmap = []
		#i = 0
		#j = 0
		#if self.alt_grid_smooth == None:
		#	self.calc_grid()
		#print("shapes: ",self.alt_grid_latitudes.shape,self.alt_grid_longitudes.shape,self.alt_grid_smooth.shape)
		#for lat in self.alt_grid_latitudes:
		#	j=0
		#	for lon in self.alt_grid_longitudes:
		#		#print(i,j,lat,lon,self.alt_grid_smooth[j][i])
		#		altmap.append([lat,lon,self.alt_grid_smooth[j][i]])
		#		j += 1
		#	i += 1
		#hm = folium.plugins.HeatMap(altmap,min_opacity=0.5,radius=5)
		line_options = {'weight': 6}

		if self.points[0].has_hr():
			colormap = linmap.scale(self.hr_min, self.hr_max).to_step(levels)
			line = folium.features.ColorLine(positions=[(p.lat,p.lon) for p in self.points],colormap=colormap,colors=[p.HR for p in self.points], control=False, **line_options)
		else:
			colormap = linmap.scale(self.vel_min, self.vel_max).to_step(levels)
			line = folium.features.ColorLine(positions=[(p.lat,p.lon) for p in self.points],colormap=colormap,colors=[p.velocity for p in self.points], control=False, **line_options)

		line.add_to(m)
		#m.add_child(hm)
		m.add_child(colormap)
		m.save(title+'.html')


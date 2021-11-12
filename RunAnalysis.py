#!/usr/bin/env python3

import sys
from run.Track import Track



RunTrack = Track("Run")
weight = 89.2

filename = sys.argv[1]
if len(sys.argv) > 2:
	weight = float(sys.argv[2])
RunTrack.set_weight(weight)
extension = filename.split('.')[-1]
if extension == 'tcx':
	RunTrack.read_tcx(filename)
elif extension == 'gpx':
	RunTrack.read_gpx(filename)
	#RunTrack.check_altitudes()
elif extension == 'fit':
	RunTrack.read_fit(filename)
	#RunTrack.check_altitudes()
else:
	exit
print(RunTrack.stats())
if RunTrack.total_time > 720:
	print("12 minutes: ",RunTrack.find_time_dist(720)," m")
if RunTrack.total_distance > 1609:
	print("1 mile: ",RunTrack.find_dist_time(1609))
if RunTrack.total_distance > 3218:
	print("2 mile: ",RunTrack.find_dist_time(3218))
if RunTrack.total_distance > 5000:
	print("5 Km: ",RunTrack.find_dist_time(5000))
if RunTrack.total_distance > 10000:
 print("10 Km: ",RunTrack.find_dist_time(10000))
if RunTrack.total_distance > 15000:
	print("15 Km: ",RunTrack.find_dist_time(15000))
if RunTrack.total_distance > 20000:
	print("20 Km: ",RunTrack.find_dist_time(20000))
if RunTrack.total_distance > 21098:
	print("Half marathon: ",RunTrack.find_dist_time(21098))
RunTrack.retrieve_altitudes()
df = RunTrack.get_dataframe()
print(df)
df.to_csv("run.csv")
RunTrack.map_html("run")

#print("Distance: ",selection[-1].dist/1000.0,"km\tTotal Time:",(selection[-1].time-selection[0].time)," Pace: ",pace_min,":",pace_sec," min/Km",
#"\nProjected Cooper test: ",cooper_dist,"m\testimated relative VO2max: ",vo2max,"ml/min/Kg\nAverage HR: ",avg_hr,"BPM\tStandard deviation HR: "
#,stdev_hr,"BPM\tHR range: [",hr_min,",",hr_max,"] Power average: ",avg_power,"W\tPower stdev: ",stdev_power,"W")

#m = folium.Map(location=[selection[0].lat,selection[0].lon],tiles='openstreetmap',zoom_start=20)
#list_points = []
#list_hr = []
#minz = 1000
#maxz = 0
#for point in selection[1:]:
#	p = (point.lat,point.lon)
#	list_hr.append(point.power)
#	if minz > point.power:
#		minz = point.power
#	if maxz < point.power:
#		maxz = point.power
#	list_points.append(p)
#uniq = len(selection)
#if uniq < 10:
#	levels = uniq + 1
#else:
#	levels = 10
#linmap = getattr(branca.colormap.linear, 'viridis')
#colormap = linmap.scale(minz, maxz).to_step(levels)
#line_options = {'weight': 2}
#line = folium.features.ColorLine(positions=list_points,colormap=colormap,colors=list_hr, control=False, **line_options)
#line.add_to(m)
#m.add_child(colormap)
#m.save('path.html')


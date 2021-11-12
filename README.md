# RunAnalysis
Analysis of run tracking data from GPX, TCX and FIT files 

This is a small personal project to play around with my running data extracted from my watch, code needs reorganising a bit.
What it can do is recalculate the distance, when altitude data is missing it can retrieve the data from NASA's SRTM250 with 2D interpolation to get more preceise altitudes (this is very experimental!). It uses the folium library to generate an HTML map with the trajectory plotted with colouring according to heart rate data.
It also tries to calculate gradients and velocities, and I'm planning to add a bit of ML with Pytorch to see whether it is possible to predict performance according to all the data gathered. 

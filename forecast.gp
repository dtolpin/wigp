data = "forecast.dat"

if (ARG1 ne "")
	data = ARG1

set ylabel "y"
set xlabel "x"
stats data using 3
plot data using 1:3:($3 + 1.96*$4) with filledcurves \
		  lc "#eeeeee" title "95%", \
	 "" using 1:3:($3 - 1.96*$4)  with filledcurves \
	      lc "#eeeeee" title "", \
	 "" using 1:3:($3 + $4) with filledcurves \
		  lc "#dddddd" title "68%", \
	 "" using 1:3:($3 - $4)  with filledcurves \
	      lc "#dddddd" title "", \
	 "" using 1:3 with lines lw 2 lc black title "predicted", \
	 "" using 1:2 with points ls 7 lc black title "observed", \
	 "" using 1:(STATS_min - 0.25) with points ls 7 lc black ps 0.67 title ""

# Script to partition the 'raw' cdnet dataset in 3 dirs: background, train, val.
# The ratio variable defines the size of the train set relative to the whole
# dataset. Ratio should be in the interval [0,100]

dataset=$1
destination=$2
ratio=$3

for videoType in $dataset/*; do
	videoTypeName=${videoType##*/}
	for  video in  $videoType/*; do
		videoName=${video##*/}

		mkdir -p $destination/train/$videoTypeName/$videoName/input/
		mkdir -p $destination/val/$videoTypeName/$videoName/input/

		mkdir -p $destination/train/$videoTypeName/$videoName/groundtruth/
		mkdir -p $destination/val/$videoTypeName/$videoName/groundtruth/

		mkdir -p $destination/background/$videoTypeName/$videoName/ROI/
		mkdir -p $destination/background/$videoTypeName/$videoName/reference/

		ROI=$(<$video/temporalROI.txt)
		eval x=($ROI)
		begin=${x[0]}
		end=${x[1]}

		if [[ ("$videoTypeName" = "badWeather") || ("$videoTypeName" = "PTZ") || ("$videoTypeName" = "turbulence") || ("$videoTypeName" = "nightVideos") || ("$videoTypeName" = "lowFramerate") ]]; then
			(( end = ($end + $begin)/2 - 1 ))
			echo "fixed roi for "$videoName
		fi

		(( endTrain = $begin + ( $ratio*( $end - $begin ) + 50 )/100 ))
		(( beginTest = $endTrain + 1))

		echo $videoTypeName"/"$videoName
		echo "begin - "$begin
		echo "endTrain - "$endTrain
		echo "beginTest - "$beginTest
		echo "end - "$end
		echo -e "\n"

		cp $video/ROI.jpg $destination/background/$videoTypeName/$videoName/ROI/

		for (( frame=1; frame<=150; frame++)); do
			printf -v name "%06d" $frame
			cp $video/input/in${name}.jpg $destination/background/$videoTypeName/$videoName/reference/
		done

		for (( frame=$begin; frame<=$endTrain; frame++)); do
			printf -v name "%06d" $frame
			cp $video/input/in${name}.jpg $destination/train/$videoTypeName/$videoName/input/
			cp $video/groundtruth/gt${name}.png $destination/train/$videoTypeName/$videoName/groundtruth/
		done

		for (( frame=$beginTest; frame<=$end; frame++)); do
			printf -v name "%06d" $frame
			cp $video/input/in${name}.jpg $destination/val/$videoTypeName/$videoName/input/
			cp $video/groundtruth/gt${name}.png $destination/val/$videoTypeName/$videoName/groundtruth/
		done

	done
done

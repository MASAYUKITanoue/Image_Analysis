for(i=1; i<=50;i=i+1){
	selectImage("crop16.tiff");
	setSlice(i);
	run("cvMatch_Template...", "image=crop16.tiff method=[Correlation coefficient] template=particle.tiff output show tolerence=0.10 threshold=0");
}
run("Images to Stack", "use");

run("TrackMate");
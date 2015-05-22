all: facedetect.o letter_recog.o

facedetect.o: facedetect.cpp
	g++ -ggdb `pkg-config --cflags opencv` -o `basename facedetect.cpp .cpp` facedetect.cpp `pkg-config --libs opencv`

letter_recog.o: letter_recog.cpp
	g++ -ggdb `pkg-config --cflags opencv` -o `basename letter_recog.cpp .cpp` letter_recog.cpp `pkg-config --libs opencv`

clean: rm *.o
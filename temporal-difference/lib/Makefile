UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)
LBITS := $(shell getconf LONG_BIT)

INCLUDE = -I. -I/usr/local/include -I$(HOME)/torch/install/include
LDFLAGS := -lavutil -lavformat -lavcodec -lssl
LIBOPTS = -shared -L/usr/local/lib/lua/5.1 -L/usr/local/lib
CFLAGS = -O3 -fopenmp -c -fpic -Wall -DINCLUDEOPENSSL -DUSE_SHA1
VPATH = fileio
VIDEODEC_FILES = video_decoder.o
LIBFILES = libvideo_decoder.so
CC = gcc

ifneq ($(filter arm%,$(UNAME_P)),)
	CFLAGS += -mfpu=neon
endif

ifeq ($(UNAME_S),Linux)
	VIDEODEC_FILES += videocap.o videocodec.o
	CFLAGS += -DDOVIDEOCAP
endif

ifeq ($(UNAME_S),Darwin)
	LDFLAGS += -lTH -lluajit -lluaT
	CFLAGS += -DDARWIN
endif


.PHONY : all
all : $(LIBFILES)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDE) $<

libvideo_decoder.so : $(VIDEODEC_FILES)
	$(CC) $(VIDEODEC_FILES) $(LIBOPTS) -o $@ $(LDFLAGS)

install : $(LIBFILES)
	sudo cp $(LIBFILES) /usr/local/lib/lua/5.1/

uninstall :
	sudo rm /usr/local/lib/lua/5.1/libvideo_decoder.so

.PHONY : clean
clean :
	rm -f *.o $(LIBFILES)

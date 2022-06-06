TARGET = gl
CC = nvcc
CFLRAGS =
SRCS = main.cpp
SRCS += app.cpp
SRCS += app_kernel.cu
SRCS += vec.cpp
OBJP = $(SRCS:.cu=.o)
OBJS = $(OBJP:.cpp=.o)
OBJDIR = ./obj
INCDIR = -I../../common/inc
LIBDIR = -L../../common/lib/linux/aarch64
LIBS = -lGL -lglut -lGLEW -lGLU

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LIBDIR) $(LIBS)

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCS)

all: clean $(OBJS) $(TARGET)

clean:
	-rm -f $(OBJS) $(TARGET) *.d

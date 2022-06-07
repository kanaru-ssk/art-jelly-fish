CC=nvcc
CFLRAGS=

TARGETDIR=bin
TARGET=$(TARGETDIR)/app

SRCDIR=src
SRCS=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')

OBJDIR=obj
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))

INCDIR = -I../../common/inc

LIBDIR = -L../../common/lib/linux/aarch64
LIBS = -lGL -lglut -lGLEW -lGLU

$(TARGET): $(OBJS)
	[ -d $(TARGETDIR) ] || mkdir $(TARGETDIR)
	$(CC) $(CFLRAGS) $+ -o $@ $(LIBDIR) $(LIBS)

$(SRCDIR)/%.cpp: $(SRCDIR)/%.cu
	$(CC) $(CFLRAGS) --cuda $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CC) $(CFLRAGS) $(INCDIR) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)

# TARGET = app
# CC = nvcc
# CFLRAGS =
# SRCS = main.cpp
# SRCS += app.cpp
# SRCS += app_kernel.cu
# SRCS += vec.cpp
# OBJP = $(SRCS:.cu=.o)
# OBJS = $(OBJP:.cpp=.o)
# OBJDIR = ./obj
# INCDIR = -I../../common/inc
# LIBDIR = -L../../common/lib/linux/aarch64
# LIBS = -lGL -lglut -lGLEW -lGLU

# $(TARGET): $(OBJS)
# 	$(CC) -o $@ $^ $(LIBDIR) $(LIBS)

# $(OBJS): $(SRCS)
# 	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCS)

# all: clean $(OBJS) $(TARGET)

# clean:
# 	-rm -f $(OBJS) $(TARGET) *.d

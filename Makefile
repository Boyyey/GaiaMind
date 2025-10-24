# Makefile for GaiaMind C Engine

CC = gcc
CFLAGS = -Wall -g
LDFLAGS = -lm  # Link math library if needed

# Source files
SRCS = core_c/main.c core_c/world.c core_c/agent.c core_c/energy.c

# Object files
OBJS = $(SRCS:.c=.o)

# Executable name
TARGET = core_c/simulator

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

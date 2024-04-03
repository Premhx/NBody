CC=gcc
CFLAGS=-Ofast -march=native -fopenmp -msse -mavx
LIBS=-lm

nbody: nbody.c
	$(CC) -o nbody nbody.c $(CFLAGS) $(LIBS)

clean:
	rm -f nbody

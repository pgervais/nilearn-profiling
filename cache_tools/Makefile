# Quick-and-dirty makefile for generating dontneed.so python extension.

CC=gcc

all: dontneed.so
clean: .PHONY
	python ./setup.py clean
	rm -f dontneed.c dontneed.so

dontneed.so:
	python ./setup.py build_ext --inplace


.PHONY:

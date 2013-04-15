# Quick-and-dirty makefile for generating dontneed.so python extension.
# linux only
CC=gcc

all: dontneed.so
clean: .PHONY
	python ./setup.py clean
	rm -f dontneed.c dontneed.so

dontneed.so: .PHONY
	python ./setup.py build_ext --inplace
	cp -f cache_tools/dontneed.so .

.PHONY:

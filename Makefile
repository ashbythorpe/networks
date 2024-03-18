Linux := $(findstring Linux, $(shell uname -s))
MacOS := $(findstring Darwin, $(shell uname -s))
Windows := $(findstring NT, $(shell uname -s))

default: test

ifdef Windows

%: %.c
	icx -Dtest_$@ -std=c11 -Wall -pedantic -g -qmkl $@.c -o $@.exe

else

%: %.c
	icx -Dtest_$@ -std=c11 -Wall -pedantic -g -qmkl $@.c -o $@ \
	    -fsanitize=undefined -fsanitize=address

endif

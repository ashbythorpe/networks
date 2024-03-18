Linux := $(findstring Linux, $(shell uname -s))
MacOS := $(findstring Darwin, $(shell uname -s))
Windows := $(findstring NT, $(shell uname -s))

default: feed_forward

ifdef Windows

%: src/%.c
	icx -Dtest_$@ -std=c11 -Wall -pedantic -g -qmkl $@.c -o $@.exe

else

%: src/%.c
	icx -Dtest_$@ -std=c11 -Wall -pedantic -g -qmkl -Iinclude src/$@.c -o target/$@ \
	    -fsanitize=undefined -fsanitize=address

endif

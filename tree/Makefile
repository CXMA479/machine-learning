CC = g++
LD_ADD = -lboost_python -lpython2.7 -lglog
IN_ADD = -I/usr/include/python2.7 -I/usr/include/boost

src = tree.cpp
target=node_test.so

$(target):$(src)
	$(CC)  -o $@ $^   $(LD_ADD) $(IN_ADD)  -fPIC  -shared

.PHONY:clean
clean:
	rm $(target)

